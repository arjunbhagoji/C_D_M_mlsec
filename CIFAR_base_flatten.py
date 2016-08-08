#!/usr/bin/env python

"""
Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)
Check the accompanying files for pretrained models. The 32-layer network (n=5), achieves a validation error of 7.42%,
while the 56-layer network (n=9) achieves error of 6.75%, which is roughly equivalent to the examples in the paper.
"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

# for the larger networks (n>=9), we need to adjust pythons recursion limit
#sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle(abs_path_i+'cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle(abs_path_i+'cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 1024, 3)).transpose(0,2,1)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:]
    Y_train = y[0:50000]
    # X_train_flip = X_train[:,:,:,::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train,X_train_flip),axis=0)
    # Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

# ##################### Build the neural network model #######################

# from lasagne.layers import Conv2DLayer
# #from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers import ElemwiseSumLayer
# from lasagne.layers import InputLayer
# from lasagne.layers import DenseLayer
# from lasagne.layers import GlobalPoolLayer
# from lasagne.layers import PadLayer
# #from lasagne.layers import ExpressionLayer
# from lasagne.layers import NonlinearityLayer
# from lasagne.nonlinearities import softmax, rectify
# from lasagne.layers import batch_norm

# def build_cnn(input_var=None, n=5):
#
#     # create a residual learning building block with two stacked 3x3 convlayers as in paper
#     def residual_block(l, increase_dim=False, projection=False):
#         input_num_filters = l.output_shape[1]
#         if increase_dim:
#             first_stride = (2,2)
#             out_num_filters = input_num_filters*2
#         else:
#             first_stride = (1,1)
#             out_num_filters = input_num_filters
#
#         stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
#         stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
#
#         # add shortcut connections
#         if increase_dim:
#             if projection:
#                 # projection shortcut, as option B in paper
#                 projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
#                 block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
#             else:
#                 # identity shortcut, as option A in paper
#                 identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
#                 padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
#                 block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
#         else:
#             block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
#
#         return block
#
#     # Building the network
#     l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
#
#     # first layer, output is 16 x 32 x 32
#     l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
#
#     # first stack of residual blocks, output is 16 x 32 x 32
#     for _ in range(n):
#         l = residual_block(l)
#
#     # second stack of residual blocks, output is 32 x 16 x 16
#     l = residual_block(l, increase_dim=True)
#     for _ in range(1,n):
#         l = residual_block(l)
#
#     # third stack of residual blocks, output is 64 x 8 x 8
#     l = residual_block(l, increase_dim=True)
#     for _ in range(1,n):
#         l = residual_block(l)
#
#     # average pooling
#     l = GlobalPoolLayer(l)
#
#     # fully connected layer
#     network = DenseLayer(
#             l, num_units=10,
#             W=lasagne.init.HeNormal(),
#             nonlinearity=softmax)
#
#     return network

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 1024),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # 2 Convolutional layers with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=64, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=64, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    # Another convolution with 64 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=128, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=128, filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),
                                        num_units=256,
                                    nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),
                                        num_units=256,
                                    nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),
                                        num_units=10,
                                    nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

# ############################## Main program ################################

def main(NUM_EPOCHS=50):
    # Check if cifar data exists
    # if not os.path.exists("./cifar-10-batches-py"):
    #     print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
    #     return

    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # add weight decay
    # all_layers = lasagne.layers.get_all_layers(network)
    # l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    # loss = loss + l2_penalty

    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.01
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.momentum(
            loss, params, learning_rate=sh_lr, momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    predict_fn=theano.function([input_var],T.argmax(test_prediction, axis=1))

    if model_exist_flag==0:
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(NUM_EPOCHS):
            # shuffle training data
            train_indices = np.arange(50000)
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, Y_train, 128, shuffle=True, augment=False):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, NUM_EPOCHS, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        np.savez(abs_path_m+'cifar10_cnn_model_no_reg_flat.npz',
                    *lasagne.layers.get_all_param_values(network))
    elif model_exist_flag==1:
        # load network weights from model file
        with np.load(abs_path_m+'cifar10_cnn_model_no_reg_flat.npz') as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # Calculate validation error of model:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    avg_test_acc=test_acc/test_batches*100
    testfile=open(abs_path_o+'CIFAR10_test_perform.txt','a')
    testfile.write('Model: model_cnn_9_layers_papernot_no_reg'+'\n')
    testfile.write("reduced_dim: "+"N.A."+"\n"+"Epochs: "
                +str(NUM_EPOCHS)+"\n"+"Test accuracy: "
                +str(avg_test_acc)+"\n")
    testfile.write("#####################################################"+
                    "####"+"\n")
    testfile.close()

    # #Computing adversarial examples using fast sign gradient method
    # req_gradient=T.grad(loss,input_var)
    # grad_function=theano.function([input_var,target_var],req_gradient)
    # confidence=theano.function([input_var],test_prediction)
    # # Variables and loop for finding adversarial examples from traning set
    # # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
    # #                 +str(WIDTH)+'_'+'.txt','a')
    # plotfile=open(abs_path_o+'FSG_CIFAR_data_cnn_9_layers_papernot.txt','a')
    # plotfile.write('rd,Dev.,Wrong,Conf.,Adv.,Conf.,Either,Conf.,Train \n')
    # plotfile.close()
    # #Loop for training data
    #
    # # advfile=open(abs_path_o+'adv_examples_FC10_'+str(DEPTH)+'_'+
    # #                 str(WIDTH)+'_'+'.txt','a')
    # # labelfile=open(abs_path_o+'adv_examples_labels_FC10_'+str(DEPTH)+'_'+
    # #                 str(WIDTH)+'_'+'.txt','a')
    # advfile=open(abs_path_o+'adv_CIFAR_cnn_9_layers_papernot.txt','a')
    # labelfile=open(abs_path_o+'adv_CIFAR_labels_cnn_9_layers_papernot.txt','a')
    # for DEV_MAG in np.linspace(0.05,0.5,10):
    #     start_time=time.time()
    #     adv_set_size=500
    #     count_tot=0.0
    #     count_wrong=0.0
    #     conf_wrong=0.0
    #     count_abs_wrong=0.0
    #     conf_abs=0.0
    #     count_adv=0.0
    #     conf_adv=0.0
    #     for old_class in range(adv_set_size):
    #         print ("Iteration is {}".format(old_class))
    #         input_curr=X_train[old_class].reshape((1,3,32,32))
    #         y_curr=Y_train[old_class].reshape((1,))
    #         # Gradient w.r.t to input and current class
    #         delta_x=grad_function(input_curr,y_curr)
    #         # Sign of gradient
    #         delta_x_sign=np.sign(delta_x)
    #         #Perturbed image
    #         adv_x=input_curr+DEV_MAG*delta_x_sign
    #         # Predicted class for perturbed image
    #         prediction_curr=predict_fn(adv_x)
    #         #Saving the perturbed sample for use with other classifiers
    #         np.savetxt(advfile,adv_x.reshape(1,3072))
    #         #label_array=np.concatenate((y_curr,prediction_curr,
    #         #                            old_class.reshape((1,)),
    #         #                            DEV_MAG.reshape((1,))))
    #         #np.savetxt(labelfile,label_array.reshape(4,1).T)
    #         labelfile.write(str(float(y_curr[0]))+" "+
    #                         str(float(prediction_curr[0]))+" "+
    #                         str(old_class)+" "+str(DEV_MAG)+"\n")
    #         if prediction_curr!=predict_fn(input_curr):
    #             count_wrong=count_wrong+1
    #             conf_wrong=conf_wrong+confidence(adv_x)[0][prediction_curr[0]]
    #         if prediction_curr!=y_curr and y_curr==predict_fn(input_curr):
    #             count_adv=count_adv+1
    #             conf_adv=conf_adv+confidence(adv_x)[0][prediction_curr[0]]
    #         if prediction_curr!=y_curr:
    #             count_abs_wrong=count_abs_wrong+1
    #             conf_abs=conf_abs+confidence(adv_x)[0][prediction_curr[0]]
    #         count_tot=count_tot+1
    #     # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
    #     #                 +str(WIDTH)+'_'+'.txt','a')
    #     plotfile=open(abs_path_o+'FSG_CIFAR_data_cnn_9_layers_papernot.txt','a')
    #     print("Deviation {} took {:.3f}s".format(
    #         DEV_MAG, time.time() - start_time))
    #     plotfile.write('no_dr'+","+str(DEV_MAG)+","+
    #                     str(count_wrong/count_tot*100)+","+
    #                     str.format("{0:.3f}",conf_wrong/count_tot)+","+
    #                     str(count_adv/count_tot*100)+","+
    #                     str.format("{0:.3f}",conf_adv/count_tot)+","+
    #                     str(count_abs_wrong/count_tot*100)+","+
    #                     str.format("{0:.3f}",conf_abs/count_tot)+","+str(1)+
    #                     "\n")
    #     plotfile.close()
    # # Loop for test data
    # for DEV_MAG in np.linspace(0.05,0.5,10):
    #     start_time = time.time()
    #     adv_set_size=100
    #     count_tot=0.0
    #     count_wrong=0.0
    #     conf_wrong=0.0
    #     count_abs_wrong=0.0
    #     conf_abs=0.0
    #     count_adv=0.0
    #     conf_adv=0.0
    #     for old_class in range(adv_set_size):
    #         #print ("Actual class is {}".format(y_old))
    #         input_curr=X_test[old_class].reshape((1,3,32,32))
    #         y_curr=Y_test[old_class].reshape((1,))
    #         # Gradient w.r.t to input and current class
    #         delta_x=grad_function(input_curr,y_curr)
    #         # Sign of gradient
    #         delta_x_sign=np.sign(delta_x)
    #         #Perturbed image
    #         adv_x=input_curr+DEV_MAG*delta_x_sign
    #         # Predicted class for perturbed image
    #         prediction_curr=predict_fn(adv_x)
    #         #Saving adversarial examples
    #         np.savetxt(advfile,adv_x.reshape(1,3072))
    #         labelfile.write(str(float(y_curr[0]))+" "+
    #                         str(float(prediction_curr[0]))+" "+
    #                         str(old_class)+" "+str(DEV_MAG)+"\n")
    #         #label_array=np.concatenate((y_curr,prediction_curr,
    #                                     #old_class.reshape((1,)),
    #                                     #DEV_MAG.reshape((1,))))
    #         #np.savetxt(labelfile,label_array.reshape(4,1).T)
    #         if prediction_curr!=predict_fn(input_curr):
    #             count_wrong=count_wrong+1
    #             conf_wrong=conf_wrong+confidence(adv_x)[0][prediction_curr[0]]
    #         if prediction_curr!=y_curr and y_curr==predict_fn(input_curr):
    #             count_adv=count_adv+1
    #             conf_adv=conf_adv+confidence(adv_x)[0][prediction_curr[0]]
    #         if prediction_curr!=y_curr:
    #             count_abs_wrong=count_abs_wrong+1
    #             conf_abs=conf_abs+confidence(adv_x)[0][prediction_curr[0]]
    #         count_tot=count_tot+1
    #     # plotfile=open(abs_path_o+'FSG_MNIST_data_hidden_'+str(DEPTH)+'_'
    #     #                 +str(WIDTH)+'_'+'.txt','a')
    #     plotfile=open(abs_path_o+'FSG_CIFAR_data_cnn_9_layers_papernot.txt','a')
    #     print("Deviation {} took {:.3f}s".format(
    #         DEV_MAG, time.time() - start_time))
    #     plotfile.write('no_rd'+","+str(DEV_MAG)+","+
    #                     str(count_wrong/count_tot*100)+","+
    #                     str.format("{0:.3f}",conf_wrong/count_tot)+","+
    #                     str(count_adv/count_tot*100)+","+
    #                     str.format("{0:.3f}",conf_adv/count_tot)+","+
    #                     str(count_abs_wrong/count_tot*100)+","+
    #                     str.format("{0:.3f}",conf_abs/count_tot)+","+str(0)+
    #                     "\n")
    #     plotfile.close()
    # advfile.close()
    # labelfile.close()

script_dir=os.path.dirname(__file__)
rel_path_i="Input_data/"
abs_path_i=os.path.join(script_dir,rel_path_i)
rel_path_m="Models/"
abs_path_m=os.path.join(script_dir,rel_path_m)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)


model_exist_flag=0

kwargs = {}
if len(sys.argv) > 1:
    kwargs['n'] = int(sys.argv[1])
if len(sys.argv) > 2:
    kwargs['model'] = sys.argv[2]
main(**kwargs)
