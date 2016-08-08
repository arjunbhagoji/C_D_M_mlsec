###############################################################################
#Author: Arjun Nitin Bhagoji
#Description: Code to train a fully connected neural network with no hidden
#layers with different values of lambda_network
###############################################################################
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import multiprocessing

from lasagne.regularization import l2

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

num_epochs=5

depth=0
width=0
drop_in=0
drop_hid=0
network = build_custom_mlp(input_var, int(depth), int(width), float(drop_in), float(drop_hid))
#Set lambda
lambda_network=1

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
l2_penalty=lasagne.regularization.regularize_layer_params(network,l2)*lambda_network
loss=loss+l2_penalty/10
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
#
# And load them again later on like this:
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)

###############################################################################
#Create adversarial examples for the current neural network
###############################################################################
import scipy
from matplotlib import pyplot as plt

def adv_exp(bfgs_iter):
    count_wrong=0.0
    count_tot=0.0
    deviation=0.0
    magnitude=0.0
    for old_class in range(1000):
        def f(x):
            return x[0]*np.sum(np.absolute(x[1:]))+eval_fn(X_curr+
                                            x[1:].reshape((1,1,28,28)),y_curr)
        X_curr=np.copy(X_train[old_class].reshape((1,1,28,28)))
        X_curr_flat=X_curr.flatten()
        y_old=np.copy(y_train[old_class].reshape((1,)))
        #print ("Actual class is {}".format(y_old))
        upper_limit=np.ones(784)-X_curr_flat
        lower_limit=np.zeros(784)-X_curr_flat
        bound=zip(lower_limit,upper_limit)
        bound.insert(0,(0,None))
        #inner_loop=np.random.randint(0,10000,10)
        #plt.imshow((X_curr[0][0])*255, cmap='gray', interpolation='nearest',
                    #vmin=0, vmax=255)
        #plt.savefig('trial'+str(old_class)+'_orig'+'_'+str(y_old)+'.png')
        for i in range(1000):
            y_curr=np.copy(y_train[i].reshape((1,)))
            if y_curr==y_old:
                continue
            #print ("Target class is {}".format(y_curr))
            x_0=np.zeros(785)
            r,fval,info=scipy.optimize.fmin_l_bfgs_b(f,x_0,approx_grad=1,
                                                bounds=bound,maxfun=bfgs_iter)
            r_mat=r[1:].reshape((28,28))
            #loss_actual_ini=eval_fn(X_curr,y_old)
            #loss_induced_ini=eval_fn(X_curr,y_curr)
            #loss_induced_final=eval_fn(X_curr+r[1:].reshape((1,1,28,28)),
                                                                        #y_curr)
            #loss_actual_ini=eval_fn(X_curr+r[1:].reshape((1,1,28,28)),y_old)
            prediction_curr=predict_fn(X_curr+r[1:].reshape((1,1,28,28)))
            #Counting successful adversarial examples
            if prediction_curr!=predict_fn(X_curr):
                count_wrong=count_wrong+1
                deviation=deviation+np.sqrt(np.sum(r_mat**2)/784)
            magnitude=magnitude+np.sqrt(np.sum(X_curr_flat**2)/784)
            #Saving modified images and their labels
            modified_image=(X_curr+r[1:].reshape((1,1,28,28))).reshape((28,28))
            #label_array=np.concatenate((y_old,y_curr,prediction_curr,
                                    #old_class.reshape((1,)),i.reshape((1,))))
            #np.savetxt(myfile_2,label_array.reshape(5,1).T,fmt='%.1f')
            #np.savetxt(myfile,modified_image.reshape(1,784))
            #plt.imshow((r_mat+X_curr[0][0])*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            #plt.savefig('trial'+str(old_class)+'_'+str(i)+'_'+str(y_old)+'.png')
            count_tot=count_tot+1
    if count_wrong!=0:
        avg_deviation=deviation/count_wrong
    else:
        avg_deviation=deviation/count_tot
    avg_magnitude=magnitude/count_tot
    print ("% of misclassified examples is {}".
                                            format(count_wrong/count_tot*100))
    print ("Deviation is {}".format(avg_deviation))
    print ("Magnitude is {}".format(avg_magnitude))
    myfile=open('output_'+str(lambda_network)+'_noPCA.txt','ab')
    (myfile.write("#FC10"+"_"+str(lambda_network)+
        str(num_epochs)+" "+str(test_acc / test_batches * 100)+" "+
        str(bfgs_iter)+" "+str(count_wrong/count_tot*100)+"% of "+
        str(count_tot)+" Avg_dev: "+str(avg_deviation)+
        " Avg_mag: "+str(avg_magnitude)+"\n"))
    (myfile.write("#########################################################"
                +"\n"))
    myfile.close()


bfgs_list=[600,700,800,1000,15000]
    #outer_loop=np.random.randint(0,10000,10)
pool=multiprocessing.Pool(processes=5)
pool.map(adv_exp,bfgs_list)
pool.close()
pool.join()

#for item in bfgs_list:
#    adv_exp(item)
