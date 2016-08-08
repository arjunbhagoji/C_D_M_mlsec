# Importing libraries
#from __future__ import print_function
import numpy as np
import matplotlib
import sys,os
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.regularization import l2
from sklearn.decomposition import PCA
import multiprocessing
import pandas as pd

# Defining required functions
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        #if not os.path.exists(filename):
        #    download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        #if not os.path.exists(filename):
        #    download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    script_dir=os.path.dirname(__file__)
    rel_path="Input_data/"
    abs_path=os.path.join(script_dir,rel_path)
    X_train = load_mnist_images(abs_path+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(abs_path+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(abs_path+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(abs_path+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def build_custom_mlp(input_var, depth, WIDTH, drop_input,
                     drop_hidden,rd):
    network = lasagne.layers.InputLayer(shape=(None, 1, rd),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.sigmoid
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, WIDTH, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network

def build_cnn(input_var,rd):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, rd),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # 2 Convolutional layers with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    # Another convolution with 64 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=64, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=64, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(network, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def pca_main(rd):
    rd=rd
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test=load_dataset()
    #X_adv=np.genfromtxt(abs_path_o+'adv_examples_cnn_9_layers_papernot.txt',delimiter=',')
    #Reshaping for PCA function
    PCA_in_train=X_train.reshape(50000,784)
    PCA_in_val=X_val.reshape(10000,784)
    PCA_in_test=X_test.reshape(10000,784)
    # PCA_in_adv_train=X_adv[:500000,:].reshape(50000,784,10)
    # PCA_in_adv_test=X_adv[500000:,:].reshape(10000,784,10)

    print("Doing PCA over the training data")
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd)
    pca_train=pca.fit(PCA_in_train)
    # X_adv_dr_train=np.zeros((50000,784,10))
    # X_adv_dr_test=np.zeros((10000,784,10))
    print ("Transforming the training, validation and test data")
    X_train_dr=pca.transform(PCA_in_train).reshape((50000,1,rd))
    X_test_dr=pca.transform(PCA_in_test).reshape((10000,1,rd))
    X_val_dr=pca.transform(PCA_in_val).reshape((10000,1,rd))
    # for i in range(1):
    #     X_adv_dr_train[:,:,i]=pca.transform(PCA_in_adv_train[:,:,i])
    #     X_adv_dr_test[:,:,i]=pca.transform(PCA_in_adv_test[:,:,i])


    ### Neural network learning with DR examples
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')
    # Create neural network model (depending on if hidden layers exist or not)
    print("Building model and compiling functions...")
    #depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
    # network=build_custom_mlp(input_var, int(DEPTH), int(WIDTH),
    #                            float(DROP_IN), float(DROP_HID),rd)
    network = build_cnn(input_var,rd)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.1, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    eval_fn=theano.function([input_var,target_var],loss)

    # Compile a function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    # Compile a function giving the network's prediction on any input
    predict_fn=theano.function([input_var],T.argmax(test_prediction, axis=1))
    #Probability vector for each output
    confidence=theano.function([input_var],test_prediction)

    # Loading the trained model
    # abs_path_m=os.path.join(abs_path_m,'model_FC10_'+str(DEPTH)
    #                             +'_'+str(WIDTH)+'_PCA_'+str(rd)+'_drop'+'.npz')
    rel_path_m="Models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    with np.load(abs_path_m+'model_cnn_9_layers_papernot'
                                +'_PCA_'+str(rd)+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test_dr, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    if adv_example_flag==0:
        ### To create adversarial exmaples according to fast sign gradient method
        #Theano object to find the gradient with respect to the input
        req_gradient=T.grad(loss,input_var)
        # Theano function to find the nuemrical value of
        #gradient for a particular input-output combination
        grad_function=theano.function([input_var,target_var],req_gradient)

        # Variables and loop for finding adversarial examples from traning set
        # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
        #                 +str(WIDTH)+'_'+str(rd)+'_drop'+'.txt','a')
        plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_9_layers_papernot_PCA_'
                        +str(rd)+'.txt','a')
        plotfile.write('rd,Dev.,Wrong,Conf.,Adv.,Conf.,Either,Conf.,Train \n')
        plotfile.close()
        #Loop for training data
        start_time=time.time()
        # advfile=open(abs_path_o+'adv_examples_FC10_'+str(DEPTH)+'_'+
        #                 str(WIDTH)+'_'+str(rd)+'_drop'+'.txt','a')
        # labelfile=open(abs_path_o+'adv_examples_labels_FC10_'+str(DEPTH)+'_'+
        #                 str(WIDTH)+'_'+str(rd)+'_drop'+'.txt','a')
        advfile=open(abs_path_o+'adv_examples_cnn_9_layers_papernot_PCA_'+str(rd)+
                        '.txt','a')
        labelfile=open(abs_path_o+'adv_examples_labels_cnn_9_layers_papernot_PCA_'
                        +str(rd)+'.txt','a')
        for DEV_MAG in np.linspace(0.05,0.5,10):
            adv_set_size=50000
            outer_loop=np.random.randint(0,50000,adv_set_size)
            count_tot=0.0
            count_wrong=0.0
            conf_wrong=0.0
            count_abs_wrong=0.0
            conf_abs=0.0
            count_adv=0.0
            conf_adv=0.0
            for old_class in outer_loop:
                #print ("Actual class is {}".format(y_old))
                input_curr=X_train_dr[old_class].reshape((1,1,rd))
                y_curr=y_train[old_class].reshape((1,))
                # Gradient w.r.t to input and current class
                delta_x=grad_function(input_curr,y_curr)
                # Sign of gradient
                delta_x_sign=np.sign(delta_x)
                #Perturbed image
                adv_x=input_curr+DEV_MAG*delta_x_sign
                # Predicted class for perturbed image
                prediction_curr=predict_fn(adv_x)
                #Saving the perturbed sample for use with other classifiers
                np.savetxt(advfile,adv_x.reshape(1,rd))
                #label_array=np.concatenate((y_curr,prediction_curr,
                #                            old_class.reshape((1,)),
                #                            DEV_MAG.reshape((1,))))
                #np.savetxt(labelfile,label_array.reshape(4,1).T)
                labelfile.write(str(float(y_curr[0]))+" "+
                                str(float(prediction_curr[0]))+" "+
                                str(old_class)+" "+str(DEV_MAG)+"\n")
                if prediction_curr!=predict_fn(input_curr):
                    count_wrong=count_wrong+1
                    conf_wrong=conf_wrong+confidence(adv_x)[0][prediction_curr[0]]
                if prediction_curr!=y_curr and y_curr==predict_fn(input_curr):
                    count_adv=count_adv+1
                    conf_adv=conf_adv+confidence(adv_x)[0][prediction_curr[0]]
                if prediction_curr!=y_curr:
                    count_abs_wrong=count_abs_wrong+1
                    conf_abs=conf_abs+confidence(adv_x)[0][prediction_curr[0]]
                count_tot=count_tot+1
            # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
            #                 +str(WIDTH)+'_'+str(rd)+'_drop'+'.txt','a')
            plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_9_layers_papernot_PCA_'
                            +str(rd)+'.txt','a')
            print("Deviation {} took {:.3f}s".format(
                DEV_MAG, time.time() - start_time))
            plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                            str(count_wrong/count_tot*100)+","+
                            str.format("{0:.3f}",conf_wrong/count_tot)+","+
                            str(count_adv/count_tot*100)+","+
                            str.format("{0:.3f}",conf_adv/count_tot)+","+
                            str(count_abs_wrong/count_tot*100)+","+
                            str.format("{0:.3f}",conf_abs/count_tot)+","+str(1)+
                            "\n")
            plotfile.close()
        # Loop for test data
        start_time=time.time()
        for DEV_MAG in np.linspace(0.05,0.5,10):
            start_time = time.time()
            adv_set_size=10000
            outer_loop=np.random.randint(0,10000,adv_set_size)
            count_tot=0.0
            count_wrong=0.0
            conf_wrong=0.0
            count_abs_wrong=0.0
            conf_abs=0.0
            count_adv=0.0
            conf_adv=0.0
            for old_class in outer_loop:
                #print ("Actual class is {}".format(y_old))
                input_curr=X_test_dr[old_class].reshape((1,1,rd))
                y_curr=y_test[old_class].reshape((1,))
                # Gradient w.r.t to input and current class
                delta_x=grad_function(input_curr,y_curr)
                # Sign of gradient
                delta_x_sign=np.sign(delta_x)
                #Perturbed image
                adv_x=input_curr+DEV_MAG*delta_x_sign
                # Predicted class for perturbed image
                prediction_curr=predict_fn(adv_x)
                #Saving adversarial examples
                np.savetxt(advfile,adv_x.reshape(1,rd))
                labelfile.write(str(float(y_curr[0]))+" "+
                                str(float(prediction_curr[0]))+" "+
                                str(old_class)+" "+str(DEV_MAG)+"\n")
                #label_array=np.concatenate((y_curr,prediction_curr,
                                            #old_class.reshape((1,)),
                                            #DEV_MAG.reshape((1,))))
                #np.savetxt(labelfile,label_array.reshape(4,1).T)
                if prediction_curr!=predict_fn(input_curr):
                    count_wrong=count_wrong+1
                    conf_wrong=conf_wrong+confidence(adv_x)[0][prediction_curr[0]]
                if prediction_curr!=y_curr and y_curr==predict_fn(input_curr):
                    count_adv=count_adv+1
                    conf_adv=conf_adv+confidence(adv_x)[0][prediction_curr[0]]
                if prediction_curr!=y_curr:
                    count_abs_wrong=count_abs_wrong+1
                    conf_abs=conf_abs+confidence(adv_x)[0][prediction_curr[0]]
                count_tot=count_tot+1
            # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
            #                 +str(WIDTH)+'_'+str(rd)+'_drop'+'.txt','a')
            plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_9_layers_papernot_PCA_'
                            +str(rd)+'.txt','a')
            print("Deviation {} took {:.3f}s".format(
                DEV_MAG, time.time() - start_time))
            plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                            str(count_wrong/count_tot*100)+","+
                            str.format("{0:.3f}",conf_wrong/count_tot)+","+
                            str(count_adv/count_tot*100)+","+
                            str.format("{0:.3f}",conf_adv/count_tot)+","+
                            str(count_abs_wrong/count_tot*100)+","+
                            str.format("{0:.3f}",conf_abs/count_tot)+","+str(0)+
                            "\n")
            plotfile.close()
        advfile.close()
        labelfile.close()
    elif adv_example_flag==1:
        plotfile=open(abs_path_o+'FSG_MNIST_cnn_9_layers_papernot_unaware_PCA_'
                        +str(rd)+'.txt','a')
        plotfile.write('rd,Dev.,Wrong,Conf.,Adv.,Conf.,Either,Conf.,Train \n')
        plotfile.close()
        DEV_MAG=0.05
        count_tot=0.0
        count_wrong=0.0
        conf_wrong=0.0
        count_abs_wrong=0.0
        conf_abs=0.0
        count_adv=0.0
        conf_adv=0.0
        adv_set_size=50000
        dev_count=0
        y_labels=np.loadtxt(abs_path_o+'adv_examples_labels_cnn_9_layers_papernot.txt')
        start_time=time.time()
        with open(abs_path_o+'adv_examples_cnn_9_layers_papernot.txt') as f:
            for line in f:
                if dev_count<10:
                    adv_x_l=list()
                    cleanline=line.rstrip('\n')
                    clean_list=cleanline.split(' ')
                    [adv_x_l.append(float(clean_list[i])) for i in range(784)]
                    adv_x=np.array(adv_x_l).reshape((1,784))
                    adv_x_dr=pca.transform(adv_x).reshape((1,1,rd))
                    curr_x_index=y_labels[int(count_tot)+dev_count*50000,2]
                    input_curr=X_train_dr[curr_x_index].reshape((1,1,rd))
                    y_curr=y_labels[int(count_tot)+dev_count*50000,0].reshape((1,))
                    prediction_curr=predict_fn(adv_x_dr)
                    if prediction_curr!=predict_fn(input_curr):
                        count_wrong=count_wrong+1
                        conf_wrong=conf_wrong+confidence(adv_x_dr)[0][prediction_curr[0]]
                    if prediction_curr!=y_curr and y_curr==predict_fn(input_curr):
                        count_adv=count_adv+1
                        conf_adv=conf_adv+confidence(adv_x_dr)[0][prediction_curr[0]]
                    if prediction_curr!=y_curr:
                        count_abs_wrong=count_abs_wrong+1
                        conf_abs=conf_abs+confidence(adv_x_dr)[0][prediction_curr[0]]
                    count_tot=count_tot+1
                    if int(count_tot)==50000:
                        dev_count=dev_count+1
                        plotfile=open(abs_path_o+'FSG_MNIST_cnn_9_layers_papernot_unaware_PCA_'
                                        +str(rd)+'.txt','a')
                        print("Deviation {} took {:.3f}s".format(
                            DEV_MAG, time.time() - start_time))
                        plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                                        str(count_wrong/count_tot*100)+","+
                                        str.format("{0:.3f}",conf_wrong/count_tot)+","+
                                        str(count_adv/count_tot*100)+","+
                                        str.format("{0:.3f}",conf_adv/count_tot)+","+
                                        str(count_abs_wrong/count_tot*100)+","+
                                        str.format("{0:.3f}",conf_abs/count_tot)+","+str(1)+
                                        "\n")
                        plotfile.close()
                        DEV_MAG=DEV_MAG+0.05
                        start_time=time.time()
                        count_tot=0.0
                        count_wrong=0.0
                        conf_wrong=0.0
                        count_abs_wrong=0.0
                        conf_abs=0.0
                        count_adv=0.0
                        conf_adv=0.0
                elif 10<=dev_count<20:
                    adv_x_l=list()
                    cleanline=line.rstrip('\n')
                    clean_list=cleanline.split(' ')
                    [adv_x_l.append(float(clean_list[i])) for i in range(784)]
                    adv_x=np.array(adv_x_l).reshape((1,784))
                    adv_x_dr=pca.transform(adv_x).reshape((1,1,rd))
                    curr_x_index=y_labels[int(count_tot)+dev_count*10000,2]
                    input_curr=X_train_dr[curr_x_index].reshape((1,1,rd))
                    y_curr=y_labels[int(count_tot)+dev_count*10000,0].reshape((1,))
                    prediction_curr=predict_fn(adv_x_dr)
                    if prediction_curr!=predict_fn(input_curr):
                        count_wrong=count_wrong+1
                        conf_wrong=conf_wrong+confidence(adv_x_dr)[0][prediction_curr[0]]
                    if prediction_curr!=y_curr and y_curr==predict_fn(input_curr):
                        count_adv=count_adv+1
                        conf_adv=conf_adv+confidence(adv_x_dr)[0][prediction_curr[0]]
                    if prediction_curr!=y_curr:
                        count_abs_wrong=count_abs_wrong+1
                        conf_abs=conf_abs+confidence(adv_x_dr)[0][prediction_curr[0]]
                    count_tot=count_tot+1
                    if int(count_tot)==10000:
                        dev_count=dev_count+1
                        plotfile=open(abs_path_o+'FSG_MNIST_cnn_9_layers_papernot_unaware_PCA_'
                                        +str(rd)+'.txt','a')
                        print("Deviation {} took {:.3f}s".format(
                            DEV_MAG, time.time() - start_time))
                        plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                                        str(count_wrong/count_tot*100)+","+
                                        str.format("{0:.3f}",conf_wrong/count_tot)+","+
                                        str(count_adv/count_tot*100)+","+
                                        str.format("{0:.3f}",conf_adv/count_tot)+","+
                                        str(count_abs_wrong/count_tot*100)+","+
                                        str.format("{0:.3f}",conf_abs/count_tot)+","+str(0)+
                                        "\n")
                        plotfile.close()
                        DEV_MAG=DEV_MAG+0.05
                        start_time=time.time()
                        count_tot=0.0
                        count_wrong=0.0
                        conf_wrong=0.0
                        count_abs_wrong=0.0
                        conf_abs=0.0
                        count_adv=0.0
                        conf_adv=0.0



# DEPTH=2
# WIDTH=800
# DROP_IN=0.2
# DROP_HID=0.5
script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

#rd_list=[100,50,30]
rd=100
num_epochs=50

# kwargs = {}
# kwargs['model'] = ('custom_mlp:'+str(DEPTH)+","+str(WIDTH)+","+str(DROP_IN)
#                             +","+str(DROP_HID))
# kwargs['num_epochs'] = num_epochs
# kwargs['rd'] = rd
#
adv_example_flag=1
 pca_main(rd)
#
# pool=multiprocessing.Pool(processes=5)
# pool.map(pca_main,rd_list)
# pool.close()
# pool.join()
