from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import multiprocessing

#from lasagne.regularization import l2

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
        if not os.path.exists(filename):
            download(filename)
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
        if not os.path.exists(filename):
            download(filename)
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

def build_custom_mlp(input_var=None, DEPTH=2, WIDTH=800, DROP_INput=.2,
                     DROP_HIDden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if DROP_INput:
        network = lasagne.layers.dropout(network, p=DROP_INput)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.sigmoid
    for _ in range(DEPTH):
        network = lasagne.layers.DenseLayer(
                network, WIDTH, nonlinearity=nonlin)
        if DROP_HIDden:
            network = lasagne.layers.dropout(network, p=DROP_HIDden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network

def build_hidden_fc(input_var=None, WIDTH=100):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    in_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    #if DROP_INput:
    #    network = lasagne.layers.dropout(network, p=DROP_INput)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.sigmoid
    #for _ in range(DEPTH):
    layer_1 = lasagne.layers.DenseLayer(in_layer, WIDTH, nonlinearity=nonlin)
    layer_2 = lasagne.layers.DenseLayer(layer_1, WIDTH, nonlinearity=nonlin)
    #    if DROP_HIDden:
    #        network = lasagne.layers.dropout(network, p=DROP_HIDden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(layer_2, 10, nonlinearity=softmax)
    return network
    #, layer_1, layer_2
def build_hidden_fc_rd(input_var, WIDTH,rd):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    in_layer = lasagne.layers.InputLayer(shape=(None, 1, rd),
                                        input_var=input_var)
    #if DROP_INput:
    #    network = lasagne.layers.dropout(network, p=DROP_INput)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.sigmoid
    #for _ in range(DEPTH):
    layer_1 = lasagne.layers.DenseLayer(in_layer, WIDTH, nonlinearity=nonlin)
    layer_2 = lasagne.layers.DenseLayer(layer_1, WIDTH, nonlinearity=nonlin)
    #    if DROP_HIDden:
    #        network = lasagne.layers.dropout(network, p=DROP_HIDden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(layer_2, 10, nonlinearity=softmax)
    return network
    #, layer_1, layer_2

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # 2 Convolutional layers with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 64 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(network, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, num_units=200,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(network, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

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
def no_pca_attack():
    rel_path_o="Output_data/"
    abs_path_o=os.path.join(script_dir,rel_path_o)
    # plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_papernot.txt','a')
    plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
                    +str(WIDTH)+'_'+'.txt','a')
    plotfile.write('rd,Dev.,Wrong,Conf.,Adv.,Conf.,Either,Conf.,Train \n')
    plotfile.close()
    start_time=time.time()
    mag_count=0
    for DEV_MAG in np.linspace(0.01,0.1,10):
        count_tot=0.0
        count_wrong=0.0
        conf_wrong=0.0
        count_abs_wrong=0.0
        conf_abs=0.0
        count_adv=0.0
        conf_adv=0.0
        for i in range(train_size):
            input_curr=X_train[i].reshape((1,1,28,28))
            y_curr=y_train[i].reshape((1,))
            # Gradient w.r.t to input and current class
            delta_x=grad_function(input_curr,y_curr)
            # Sign of gradient
            delta_x_sign=np.sign(delta_x).reshape((784))
            #delta_x_sign=delta_x_sign/np.linalg.norm((delta_x_sign))
            delta_x_sign=delta_x_sign.reshape((1,1,28,28))
            #Perturbed image
            adv_x=input_curr+DEV_MAG*delta_x_sign
            adv_examples_train[i,:,mag_count]=adv_x.reshape((784))
            # Predicted class for perturbed image
            prediction_curr=predict_fn(adv_x)
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
        plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
                        +str(WIDTH)+'_'+'.txt','a')
        # plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_papernot.txt','a')
        print("Deviation {} took {:.3f}s".format(
            DEV_MAG, time.time() - start_time))
        plotfile.write('no_dr'+","+str(DEV_MAG)+","+
                        str(count_wrong/count_tot*100)+","+
                        str.format("{0:.3f}",conf_wrong/count_wrong)+","+
                        str(count_adv/count_tot*100)+","+
                        str.format("{0:.3f}",conf_adv/count_adv)+","+
                        str(count_abs_wrong/count_tot*100)+","+
                        str.format("{0:.3f}",conf_abs/count_abs_wrong)+","+str(1)+
                        "\n")
        plotfile.close()
        mag_count=mag_count+1

    mag_count=0
    for DEV_MAG in np.linspace(0.01,0.1,10):
        count_tot=0.0
        count_wrong=0.0
        conf_wrong=0.0
        count_abs_wrong=0.0
        conf_abs=0.0
        count_adv=0.0
        conf_adv=0.0
        for i in range(test_size):
            input_curr=X_test[i].reshape((1,1,28,28))
            y_curr=y_test[i].reshape((1,))
            # Gradient w.r.t to input and current class
            delta_x=grad_function(input_curr,y_curr)
            # Sign of gradient
            delta_x_sign=np.sign(delta_x).reshape((784))
            #delta_x_sign=delta_x_sign/np.linalg.norm((delta_x_sign))
            delta_x_sign=delta_x_sign.reshape((1,1,28,28))
            #Perturbed image
            adv_x=input_curr+DEV_MAG*delta_x_sign
            adv_examples_test[i,:,mag_count]=adv_x.reshape((784))
            # Predicted class for perturbed image
            prediction_curr=predict_fn(adv_x)
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
        plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
                        +str(WIDTH)+'_'+'.txt','a')
        # plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_papernot.txt','a')
        print("Deviation {} took {:.3f}s".format(
            DEV_MAG, time.time() - start_time))
        plotfile.write('no_dr'+","+str(DEV_MAG)+","+
                        str(count_wrong/count_tot*100)+","+
                        str.format("{0:.3f}",conf_wrong/count_wrong)+","+
                        str(count_adv/count_tot*100)+","+
                        str.format("{0:.3f}",conf_adv/count_adv)+","+
                        str(count_abs_wrong/count_tot*100)+","+
                        str.format("{0:.3f}",conf_abs/count_abs_wrong)+","+str(1)+
                        "\n")
        plotfile.close()
        mag_count=mag_count+1

def pca_attack(rd):

    #Reduced dimension
    rd=rd

    #Getting the principal axes
    #Reshaping for PCA function
    PCA_in_train=X_train.reshape(50000,784)
    PCA_in_val=X_val.reshape(10000,784)
    PCA_in_test=X_test.reshape(10000,784)

    print("Doing PCA over the training data")
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd)
    pca_train=pca.fit(PCA_in_train)

    X_train_dr=pca.transform(PCA_in_train).reshape((50000,1,rd))
    X_test_dr=pca.transform(PCA_in_test).reshape((10000,1,rd))
    # X_train_rev=pca.inverse_transform(X_train_dr)
    # X_train_rev=X_train_rev.reshape((50000,1,28,28))
    # X_test_rev=pca.inverse_transform(X_test_dr)
    # X_test_rev=X_test_rev.reshape((10000,1,28,28))

    ### Neural network learning with DR examples
    # Prepare Theano variables for inputs and targets
    input_var_rd = T.tensor3('inputs')
    target_var_rd = T.ivector('targets')
    # Create neural network model (depending on if hidden layers exist or not)
    print("Building model and compiling functions...")
    #depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
    # network=build_custom_mlp(input_var, int(DEPTH), int(WIDTH),
    #                            float(DROP_IN), float(DROP_HID),rd)
    #network = build_cnn(input_var,rd)
    network_rd = build_hidden_fc_rd(input_var_rd, WIDTH,rd)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction_rd = lasagne.layers.get_output(network_rd)
    loss_rd = lasagne.objectives.categorical_crossentropy(prediction_rd, target_var_rd)
    loss_rd = loss_rd.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum.
    params_rd = lasagne.layers.get_all_params(network_rd, trainable=True)
    updates_rd = lasagne.updates.nesterov_momentum(
            loss_rd, params_rd, learning_rate=0.1, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction_rd = lasagne.layers.get_output(network_rd, deterministic=True)
    test_loss_rd = lasagne.objectives.categorical_crossentropy(test_prediction_rd,
                                                            target_var_rd)
    test_loss_rd = test_loss_rd.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc_rd = T.mean(T.eq(T.argmax(test_prediction_rd, axis=1), target_var_rd),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn_rd = theano.function([input_var_rd, target_var_rd], loss_rd, updates=updates_rd)
    eval_fn_rd=theano.function([input_var_rd,target_var_rd],loss_rd)

    # Compile a function computing the validation loss and accuracy:
    val_fn_rd = theano.function([input_var_rd, target_var_rd], [test_loss_rd, test_acc_rd])
    # Compile a function giving the network's prediction on any input
    predict_fn_rd=theano.function([input_var_rd],T.argmax(test_prediction_rd, axis=1))
    #Probability vector for each output
    confidence_rd=theano.function([input_var_rd],test_prediction_rd)

    # Loading the trained model
    # abs_path_m=os.path.join(abs_path_m,'model_FC10_'+str(DEPTH)
    #                             +'_'+str(WIDTH)+'_PCA_'+str(rd)+'_drop'+'.npz')
    script_dir=os.path.dirname(__file__)
    rel_path_m="Models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    # with np.load(abs_path_m+'model_cnn_9_layers_papernot'
    #                             +'_PCA_'+str(rd)+'.npz') as f:
    with np.load(abs_path_m+'model_FC10_'+str(DEPTH)+'_'+str(WIDTH)+'_PCA_'+str(rd)+'.npz') as f:
        param_values_rd = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network_rd, param_values_rd)

    # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test_dr, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn_rd(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))
    # rel_path_o="Output_data/"
    # abs_path_o=os.path.join(script_dir,rel_path_o)
    # myfile=open(abs_path_o+'MNIST_test_perform_PCA.txt','a')
    # # myfile.write('Model: FC10_'+str(DEPTH)+'_'+str(WIDTH)+
    # #                 '_'+'\n')
    # # myfile.write('Model: model_cnn_9_layers_papernot'+'\n')
    # myfile.write(str(rd)+", "
    #             +str(test_acc / test_batches * 100)+"\n")
    # myfile.write("#####################################################"+
    #                 "####"+"\n")
    # myfile.close()

    ### To create adversarial exmaples according to fast sign gradient method
    #Theano object to find the gradient with respect to the input
    req_gradient_rd=T.grad(loss_rd,input_var_rd)
    # Theano function to find the nuemrical value of
    #gradient for a particular input-output combination
    grad_function_rd=theano.function([input_var_rd,target_var_rd],req_gradient_rd)

    #After training, we compute and print the test error:
    test_err_pca = 0
    test_acc_pca = 0
    test_batches_pca = 0
    test_conf_pca= 0
    for batch in iterate_minibatches(X_test_dr, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn_rd(inputs, targets)
        test_err_pca += err
        test_acc_pca += acc
        test_batches_pca += 1

    for i in range(10000):
        x_curr=X_test_dr[i].reshape((1,1,rd))
        test_conf_pca+=confidence_rd(x_curr)[0][predict_fn_rd(x_curr)[0]]


    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err_pca / test_batches_pca))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc_pca / test_batches_pca * 100))
    print("  test confidence:\t\t{:.2f} %".format(test_conf_pca / 10000 * 100))

    avg_test_acc_pca=test_acc_pca/test_batches_pca*100

    script_dir=os.path.dirname(__file__)
    rel_path_o="Output_data/"
    abs_path_o=os.path.join(script_dir,rel_path_o)
    # Writing the test results out to a file
    myfile=open(abs_path_o+'MNIST_test_perform_PCA.txt','a')
    # myfile.write('Model: model_cnn_papernot_w_PCA_clean_'+str(rd)+'\n')
    myfile.write(str(rd)+", "
                +str(avg_test_acc_pca)+"\n")
    myfile.write("#####################################################"+
                    "####"+"\n")
    myfile.close()

    # Variables and loop for finding adversarial examples from traning set
    plotfile_pca=open(abs_path_o+'FSG_MNIST_NN_'+str(DEPTH)+'_'
                    +str(WIDTH)+'_retrain_PCA_unaware_'+str(rd)+'.txt','a')
    # plotfile=open(abs_path_o+'FSG_MNIST_data_cnn_papernot'+'_w_PCA_clean'+str(rd)+'.txt','a')
    # plotfile.write('rd,Dev.,Wrong,Conf.,Adv.,Conf.,Either,Conf.,Train \n')
    plotfile_pca.write('rd,Dev.,Correct,Conf.,Adv.,Conf.,Either,Conf.,Train \n')
    plotfile_pca.close()
    #plotfile.close()
    #Loop for training data
    start_time=time.time()
    mag_count=0
    for DEV_MAG in np.linspace(0.01,0.1,10):
        X_adv_dr=pca.transform(adv_examples_train[:,:,mag_count])
        count_tot=0.0
        count_wrong_pca=0.0
        conf_wrong_pca=0.0
        count_abs_wrong_pca=0.0
        conf_abs_pca=0.0
        count_adv_pca=0.0
        conf_adv_pca=0.0
        # count_c_pca=conf_c_pca = count_c_abs_pca = conf_c_abs_pca = count_c_adv_pca = conf_c_adv_pca=0.0
        for i in range(train_size):
            input_curr=X_train_dr[i].reshape((1,1,rd))
            y_curr=y_train[i].reshape((1,))
            #Reducing the dimension of the image
            x_adv=X_adv_dr[i,:].reshape((1,1,rd))
            prediction_pca=predict_fn_rd(x_adv)
            if prediction_pca!=predict_fn_rd(input_curr):
                count_wrong_pca=count_wrong_pca+1
                conf_wrong_pca=conf_wrong_pca+confidence_rd(x_adv)[0][prediction_pca[0]]
            if prediction_pca!=y_curr and y_curr==predict_fn_rd(input_curr):
                count_adv_pca=count_adv_pca+1
                conf_adv_pca=conf_adv_pca+confidence_rd(x_adv)[0][prediction_pca[0]]
            if prediction_pca!=y_curr:
                count_abs_wrong_pca=count_abs_wrong_pca+1
                conf_abs_pca=conf_abs_pca+confidence_rd(x_adv)[0][prediction_pca[0]]
            # if prediction_pca==predict_fn(input_curr):
            #     count_c_pca=count_c_pca+1
            #     conf_c_pca=conf_c_pca+confidence(adv_x_rev)[0][prediction_pca[0]]
            # if prediction_pca==y_curr and y_curr==predict_fn(input_curr):
            #     count_c_adv_pca=count_c_adv_pca+1
            #     conf_c_adv_pca=conf_c_adv_pca+confidence(adv_x_rev)[0][prediction_pca[0]]
            # if prediction_pca==y_curr:
            #     count_c_abs_pca=count_c_abs_pca+1
            #     conf_c_abs_pca=conf_c_abs_pca+confidence(adv_x_rev)[0][prediction_pca[0]]
            count_tot=count_tot+1
        # plotfile_pca=open(abs_path_o+'FSG_MNIST_data_cnn_papernot'+
        #                             '_w_PCA_clean'+str(rd)+'.txt','a')
        plotfile_pca=open(abs_path_o+'FSG_MNIST_NN_'+str(DEPTH)+'_'
                        +str(WIDTH)+'_retrain_PCA_unaware_'+str(rd)+'.txt','a')
        plotfile_pca.write(str(rd)+","+str(DEV_MAG)+","+
                        str(count_wrong_pca/count_tot*100)+","+
                        str.format("{0:.3f}",conf_wrong_pca/count_wrong_pca)+","+
                        str(count_adv_pca/count_tot*100)+","+
                        str.format("{0:.3f}",conf_adv_pca/count_adv_pca)+","+
                        str(count_abs_wrong_pca/count_tot*100)+","+
                        str.format("{0:.3f}",conf_abs_pca/count_abs_wrong_pca)+","+str(1)+
                        "\n")
        # plotfile_pca.write(str(rd)+","+str(DEV_MAG)+","+
        #                 str(count_c_pca/count_tot*100)+","+
        #                 str.format("{0:.3f}",conf_c_pca/count_tot)+","+
        #                 str(count_c_adv_pca/count_tot*100)+","+
        #                 str.format("{0:.3f}",conf_c_adv_pca/count_tot)+","+
        #                 str(count_c_abs_pca/count_tot*100)+","+
        #                 str.format("{0:.3f}",conf_c_abs_pca/count_tot)+","+str(1)+
        #                 "\n")
        plotfile_pca.close()
        mag_count=mag_count+1

    # Loop for test data
    start_time=time.time()
    mag_count=0
    for DEV_MAG in np.linspace(0.01,0.1,10):
        X_adv_dr=pca.transform(adv_examples_test[:,:,mag_count])
        count_tot=0.0
        count_wrong_pca=0.0
        conf_wrong_pca=0.0
        count_abs_wrong_pca=0.0
        conf_abs_pca=0.0
        count_adv_pca=0.0
        conf_adv_pca=0.0
        # count_c_pca=conf_c_pca = count_c_abs_pca = conf_c_abs_pca = count_c_adv_pca = conf_c_adv_pca=0.0
        for i in range(test_size):
            input_curr=X_test_dr[i].reshape((1,1,rd))
            y_curr=y_test[i].reshape((1,))
            #Reducing the dimension of the image
            x_adv=X_adv_dr[i,:].reshape((1,1,rd))
            prediction_pca=predict_fn_rd(x_adv)
            if prediction_pca!=predict_fn_rd(input_curr):
                count_wrong_pca=count_wrong_pca+1
                conf_wrong_pca=conf_wrong_pca+confidence_rd(x_adv)[0][prediction_pca[0]]
            if prediction_pca!=y_curr and y_curr==predict_fn_rd(input_curr):
                count_adv_pca=count_adv_pca+1
                conf_adv_pca=conf_adv_pca+confidence_rd(x_adv)[0][prediction_pca[0]]
            if prediction_pca!=y_curr:
                count_abs_wrong_pca=count_abs_wrong_pca+1
                conf_abs_pca=conf_abs_pca+confidence_rd(x_adv)[0][prediction_pca[0]]
            # if prediction_pca==predict_fn(input_curr):
            #     count_c_pca=count_c_pca+1
            #     conf_c_pca=conf_c_pca+confidence(adv_x_rev)[0][prediction_pca[0]]
            # if prediction_pca==y_curr and y_curr==predict_fn(input_curr):
            #     count_c_adv_pca=count_c_adv_pca+1
            #     conf_c_adv_pca=conf_c_adv_pca+confidence(adv_x_rev)[0][prediction_pca[0]]
            # if prediction_pca==y_curr:
            #     count_c_abs_pca=count_c_abs_pca+1
            #     conf_c_abs_pca=conf_c_abs_pca+confidence(adv_x_rev)[0][prediction_pca[0]]
            count_tot=count_tot+1
        # plotfile_pca=open(abs_path_o+'FSG_MNIST_data_cnn_papernot'+
        #                             '_w_PCA_clean'+str(rd)+'.txt','a')
        plotfile_pca=open(abs_path_o+'FSG_MNIST_NN_'+str(DEPTH)+'_'
                        +str(WIDTH)+'_retrain_PCA_unaware_'+str(rd)+'.txt','a')
        plotfile_pca.write(str(rd)+","+str(DEV_MAG)+","+
                        str(count_wrong_pca/count_tot*100)+","+
                        str.format("{0:.3f}",conf_wrong_pca/count_wrong_pca)+","+
                        str(count_adv_pca/count_tot*100)+","+
                        str.format("{0:.3f}",conf_adv_pca/count_adv_pca)+","+
                        str(count_abs_wrong_pca/count_tot*100)+","+
                        str.format("{0:.3f}",conf_abs_pca/count_abs_wrong_pca)+","+str(0)+
                        "\n")
        # plotfile_pca.write(str(rd)+","+str(DEV_MAG)+","+
        #                 str(count_c_pca/count_tot*100)+","+
        #                 str.format("{0:.3f}",conf_c_pca/count_tot)+","+
        #                 str(count_c_adv_pca/count_tot*100)+","+
        #                 str.format("{0:.3f}",conf_c_adv_pca/count_tot)+","+
        #                 str(count_c_abs_pca/count_tot*100)+","+
        #                 str.format("{0:.3f}",conf_c_abs_pca/count_tot)+","+str(1)+
        #                 "\n")
        plotfile_pca.close()
        mag_count=mag_count+1

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")

NUM_EPOCHS=500

#if hidden_layers==0:
DEPTH=2
WIDTH=100
# DROP_IN=0.2
# DROP_HID=0.5
# network = build_custom_mlp(input_var, int(DEPTH), int(WIDTH),
#                             float(DROP_IN), float(DROP_HID))
network=build_hidden_fc(input_var, WIDTH=WIDTH)
    #lambda_network=1e-4
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #l2_penalty=lasagne.regularization.regularize_layer_params(
                                                    #network,l2)*lambda_network
    #loss=loss+l2_penalty/10
    # loss = loss.mean()
    #myfile=open('FSG_results.txt','a')
    #myfile.write('Model: FC10('+str(lambda_network)+')'+'\n')
# elif hidden_layers!=0:
#     WIDTH=100
#     network=build_hidden_fc(input_var,int(WIDTH))
    #, layer_1, layer_2
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #layers={layer_1:1e-7,layer_2:1e-7,network:1e-7}
    #l2_penalty=lasagne.regularization.regularize_layer_params_weighted(layers,
                                                                            #l2)
    #loss=loss
    #+l2_penalty
    # loss = loss.mean()
    # myfile=open('FSG_results.txt','a')
    # myfile.write('Model: FC10_'+str(hidden_layers)+'_'+str(WIDTH)+'\n')

# network=build_cnn(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss=loss.mean()
# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

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

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

predict_fn=theano.function([input_var],T.argmax(test_prediction, axis=1))
confidence=theano.function([input_var],test_prediction)
correct_confidence=theano.function([input_var],T.max(test_prediction,axis=1))

model_exist_flag=1
adv_example_flag=0

script_dir=os.path.dirname(__file__)

if model_exist_flag==1:
    rel_path_m="Models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    # And load them again later on like this:
    with np.load(abs_path_m+'model_FC10_'+str(DEPTH)+'_'+str(WIDTH)+'_.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # with np.load(abs_path_m+'model_cnn_9_layers_papernot.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
# elif model_exist_flag==1 and hidden_layers!=0:
#     with np.load('model_FC10_'+str(hidden_layers)+'_'+str(WIDTH)+'.npz') as f:
#         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#     lasagne.layers.set_all_param_values(network, param_values)
elif model_exist_flag==0:
        # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 1, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        #print("{}".format(inputs))
        #print("{}".format(targets+1))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, NUM_EPOCHS, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    rel_path_m="Models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    # np.savez(abs_path_m+'model_cnn_9_layers_papernot.npz',
    #         *lasagne.layers.get_all_param_values(network))
    np.savez(abs_path_m+'model_FC10_'+str(DEPTH)+'_'+str(WIDTH)+'_.npz',
            *lasagne.layers.get_all_param_values(network))
# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
test_conf= 0
for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
for i in range(10000):
    x_curr=X_test[i].reshape((1,1,28,28))
    test_conf+=confidence(x_curr)[0][predict_fn(x_curr)[0]]

print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
print("  test confidence:\t\t{:.2f} %".format(test_conf / 10000 * 100))

avg_test_acc=test_acc/test_batches*100

# Writing the test results out to a file
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)
myfile=open(abs_path_o+'MNIST_test_perform.txt','a')
myfile.write('Model: FC10_'+str(DEPTH)+'_'+str(WIDTH)+
                '_'+'\n')
# myfile.write('Model: model_cnn_9_layers_papernot'+'\n')
myfile.write("reduced_dim: "+"N.A."+"\n"+"Epochs: "
            +str(NUM_EPOCHS)+"\n"+"Test accuracy: "
            +str(avg_test_acc)+"\n")
myfile.write("#####################################################"+
                "####"+"\n")
myfile.close()

#Computing adversarial examples using fast sign gradient method
req_gradient=T.grad(loss,input_var)
grad_function=theano.function([input_var,target_var],req_gradient)
#Computing and storing adv. examples
no_of_mags=10
train_size=50000
test_size=10000
adv_examples_train=np.zeros((train_size,784,no_of_mags))
adv_examples_test=np.zeros((test_size,784,no_of_mags))

script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

no_pca_attack()

rd_list=[784,331,100,50,40,30,20,10]
#d_list=[784]
# pca_attack(30)

pool=multiprocessing.Pool(processes=8)
pool.map(pca_attack,rd_list)
pool.close()
pool.join()
