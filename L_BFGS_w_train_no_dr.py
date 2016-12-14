###############################################################################
#Author: Arjun Nitin Bhagoji
#Description: Code to train a fully connected neural network with no hidden
#layers with different values of lambda_network
###############################################################################
#from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import multiprocessing

from lasagne.regularization import l2
from sklearn.decomposition import PCA

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
    X_train =np.float32( load_mnist_images('train-images-idx3-ubyte.gz'))
    y_train = np.int32( load_mnist_labels('train-labels-idx1-ubyte.gz'))
    X_test = np.float32( load_mnist_images('t10k-images-idx3-ubyte.gz'))
    y_test =  np.int32 (load_mnist_labels('t10k-labels-idx1-ubyte.gz'))

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

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
    return network, layer_1, layer_2

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
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
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
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

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")

num_epochs=500

DEPTH=2
WIDTH=100
# drop_in=0
# drop_hid=0
network,layer_1, layer_2=build_hidden_fc(input_var, WIDTH=WIDTH)
#network = build_custom_mlp(input_var, int(depth), int(width), float(drop_in), float(drop_hid))
#Set lambda
lambda_network=1

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
# layers={layer_1:1e-7,layer_2:1e-7,network:1e-7}
# l2_penalty=lasagne.regularization.regularize_layer_params_weighted(layers,l2)
# loss=loss+l2_penalty
# l2_penalty=lasagne.regularization.regularize_layer_params(network,l2)*lambda_network
# loss=loss+l2_penalty/10
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

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

script_dir=os.path.dirname(__file__)
model_exist_flag=1

if model_exist_flag==0:
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
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
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
         inputs, targets = batch
         err, acc = val_fn(inputs, targets)
         test_err += err
         test_acc += acc
         test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model'+str(lambda_network)+'.npz', *lasagne.layers.get_all_param_values(network))
elif model_exist_flag==1:
# And load them again later on like this:
    rel_path_m="Models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    # And load them again later on like this:
    with np.load(abs_path_m+'model_FC10_'+str(DEPTH)+'_'+str(WIDTH)+'_.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
         inputs, targets = batch
         err, acc = val_fn(inputs, targets)
         test_err += err
         test_acc += acc
         test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
         test_acc / test_batches * 100))

###############################################################################
#Create adversarial examples for the current neural network
###############################################################################
import scipy
from matplotlib import pyplot as plt

def adv_exp():
    count_wrong=0.0
    count_tot=0.0
    count_correct=0.0
    deviation=0.0
    magnitude=0.0
    C=0.7
    adv_list=[]
    advfile=open(abs_path_a+'L_BFGS_test_adv.txt','ab')
    labelfile=open(abs_path_a+'L_BFGS_test_adv_labels.txt','ab')
    for old_class in range(outer):
        def f(x):
            # return x[0]*np.sum(np.absolute(x[1:]))+eval_fn(X_curr+
            #                                 x[1:].reshape((1,1,28,28)),y_curr)
            loss_adv,acc_adv=val_fn(X_curr+x.reshape((1,1,28,28)),y_curr)
            return C*np.linalg.norm(x)+loss_adv
        X_curr=np.copy(X_test[old_class].reshape((1,1,28,28)))
        X_curr_flat=X_curr.flatten()
        y_old=np.copy(y_test[old_class].reshape((1,)))
        ini_class=predict_fn(X_curr)
        #print ("Actual class is {}".format(y_old))
        upper_limit=np.ones(784)-X_curr_flat
        lower_limit=np.zeros(784)-X_curr_flat
        bound=zip(lower_limit,upper_limit)
        #bound.insert(0,(0,None))
        #inner_loop=np.random.randint(0,10000,10)
        #plt.imshow((X_curr[0][0])*255, cmap='gray', interpolation='nearest',
                    #vmin=0, vmax=255)
        #plt.savefig('trial'+str(old_class)+'_orig'+'_'+str(y_old)+'.png')
        inner_count=0
        for i in np.random.randint(0,10000,inner):
            print i
            y_curr=np.copy(y_test[i].reshape((1,)))
            if y_curr==y_old:
                continue
            adv_list.append(old_class)
            #print ("Target class is {}".format(y_curr))
            x_0=np.zeros(784)
            r,fval,info=scipy.optimize.fmin_l_bfgs_b(f,x_0,approx_grad=1,
                                                bounds=bound)
            # r_mat=r.reshape((28,28))
            #loss_actual_ini=eval_fn(X_curr,y_old)
            #loss_induced_ini=eval_fn(X_curr,y_curr)
            #loss_induced_final=eval_fn(X_curr+r[1:].reshape((1,1,28,28)),
                                                                        #y_curr)
            #loss_actual_ini=eval_fn(X_curr+r[1:].reshape((1,1,28,28)),y_old)
            adv_x=X_curr+r.reshape((1,1,28,28))
            prediction_curr=predict_fn(adv_x)
            #Counting successful adversarial examples
            if ini_class[0]==y_test[old_class]:
                count_correct=count_correct+1
                magnitude=magnitude+np.sqrt(np.sum(X_curr_flat**2)/784)
            if prediction_curr!=predict_fn(X_curr) and ini_class[0]==y_test[old_class]:
                count_wrong=count_wrong+1
                deviation=deviation+np.sqrt(np.sum(r**2)/784)
                old_class=np.array(old_class)
                i=np.array(i)
                label_array=np.concatenate((y_old,y_curr,prediction_curr,
                                        old_class.reshape((1,)),i.reshape((1,))))
                np.savetxt(labelfile,label_array.reshape(5,1).T,fmt='%.1f')
                np.savetxt(advfile,adv_x.reshape(1,784))
            #Saving modified images and their labels
            # modified_image=(adv_x.reshape((28,28)))
            # adv_examples_train[old_class*inner+inner_count,:]=adv_x.reshape((784))
            #fig=plt.imshow(modified_image*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            #plt.savefig('trial_no_PCA_'+str(old_class)+'_'+str(i)+'_'+str(y_old)+'.png')
            inner_count=inner_count+1
        count_tot=count_tot+1
        # plt.imshow(r.reshape((28,28))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # plt.savefig('perturb_no_PCA'+str(prediction_curr)+'_'+str(y_old)+'.png')
        # plt.imshow(adv_x.reshape((28,28))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # plt.savefig('trial_no_PCA'+str(prediction_curr)+'_'+str(y_old)+'.png')
    if count_wrong!=0:
        avg_deviation=deviation/count_wrong
    else:
        avg_deviation=deviation/count_correct
    avg_magnitude=magnitude/count_correct
    print ("% of misclassified examples is {}".
                                    format(count_wrong/count_correct*100))
    print ("Deviation is {}".format(avg_deviation))
    print ("Magnitude is {}".format(avg_magnitude))
    myfile=open(abs_path_o+'MNIST_Optimal_attack.txt','ab')
    myfile.write("#FC10"+"_"+str(DEPTH)+'_'+str(WIDTH)+
        str(num_epochs)+" "+str(test_acc / test_batches * 100)+" "+
        str(C)+" "+str(count_wrong/count_correct*100)+"% of "+
        str(count_correct)+" Avg_dev: "+str(avg_deviation)+
        " Avg_mag: "+str(avg_magnitude)+"\n")
    myfile.write("#########################################################"
                +"\n")
    myfile.close()
    advfile.close()
    labelfile.close()
    return adv_list

def adv_exp_pca(rd):
    #Reduced dimension
    C=0.7
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

    X_train_dr=pca.transform(PCA_in_train)
    X_test_dr=pca.transform(PCA_in_test)
    X_train_rev=pca.inverse_transform(X_train_dr)
    X_train_rev=X_train_rev.reshape((50000,1,28,28))
    X_test_rev=pca.inverse_transform(X_test_dr)
    X_test_rev=X_test_rev.reshape((10000,1,28,28))

    X_adv=np.genfromtxt(abs_path_a+'L_BFGS_test_adv.txt')
    X_adv_dr=pca.transform(X_adv)
    rev_adv=pca.inverse_transform(X_adv_dr)
    labels=np.genfromtxt(abs_path_a+'L_BFGS_test_adv_labels.txt')
    label_list=labels[:,3]

    # X_adv_dr=pca.transform(adv_examples_train)
    # rev_adv=pca.inverse_transform(X_adv_dr)

    count_wrong=0.0
    count_tot=0.0
    deviation=0.0
    magnitude=0.0
    count_correct=0.0
    for i in label_list:
        input_curr=X_test_rev[i].reshape((1,1,28,28))
        X_curr_flat=input_curr.reshape((784))
        ini_class=predict_fn(input_curr)
        #Perturbed image
        adv_x=rev_adv[count_tot,:]
        r=adv_x-X_curr_flat
        # Predicted class for perturbed image
        # cw,ca = val_fn(adv_x,y_curr)
        # count_wrong=count_wrong+ca
        prediction_curr=predict_fn(adv_x.reshape((1,1,28,28)))
        if ini_class[0]==y_test[i]:
            count_correct=count_correct+1
            magnitude=magnitude+np.linalg.norm(X_curr_flat)/np.sqrt(784)
        if prediction_curr!=predict_fn(input_curr) and ini_class[0]==y_test[i]:
            count_wrong=count_wrong+1
            deviation=deviation+np.linalg.norm(r)/np.sqrt(784)
        count_tot=count_tot+1
        # plt.imshow(input_curr.reshape((28,28))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # plt.savefig('trial_PCA'+'_'+str(current_predict)+'.png')
        # plt.imshow(adv_x.reshape((28,28))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # plt.savefig('trial_PCA'+str(adv_predict)+'_'+str(current_predict)+'.png')
        # plt.imshow(r.reshape((28,28))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        # plt.savefig('perturb_PCA'+str(adv_predict)+'_'+str(current_predict)+'.png')
    if count_wrong!=0:
        avg_deviation=deviation/count_wrong
    else:
        avg_deviation=deviation/count_correct
    avg_magnitude=magnitude/count_correct
    print ("% of misclassified examples is {}".
                                            format(count_wrong/count_correct*100))
    print ("Deviation is {}".format(deviation))
    print ("Magnitude is {}".format(avg_magnitude))
    myfile=open(abs_path_o+'MNIST_Optimal_attack.txt','ab')
    myfile.write("#FC10"+"_"+str(DEPTH)+'_'+str(WIDTH)+'_PCA_'+str(rd)+'_clean '+
        str(C)+" "+str(count_wrong/count_correct*100)+'% of ' +
        str(count_correct)+" Avg_dev: "+str(avg_deviation)+
        " Avg_mag: "+str(avg_magnitude)+"\n")
    myfile.write("#########################################################"
                +"\n")
    myfile.close()
    myfile=open(abs_path_o+'L_BFGS_recons.txt','ab')
    myfile.write(str(rd)+", "+str(count_wrong/count_correct*100)+"\n")
    myfile.write("#########################################################"
                +"\n")
    myfile.close()


# adv_exp(600)
script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)
rel_path_a="Adv_examples/"
abs_path_a=os.path.join(script_dir,rel_path_a)
# C_list=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
# C_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
#     #outer_loop=np.random.randint(0,10000,10)
# pool=multiprocessing.Pool(processes=8)
# pool.map(adv_exp,C_list)
# pool.close()
# pool.join()
outer=10000
inner=1

# adv_examples_train=np.zeros((inner*outer,784))

# adv_list=adv_exp()

rd_list=[784,331,100,50,40,30,20,10]
#
# adv_exp_pca(100)
#
pool=multiprocessing.Pool(processes=8)
pool.map(adv_exp_pca,rd_list)
pool.close()
pool.join()

#for item in bfgs_list:
#    adv_exp(item)
