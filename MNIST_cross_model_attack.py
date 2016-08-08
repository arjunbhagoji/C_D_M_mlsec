###############################################################################
#Author: Arjun Nitin Bhagoji
#Description: Code to find cross model errors for models with 2 hidden layers
###############################################################################
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5,rd):
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
    #if drop_input:
    #    network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.sigmoid
    #for _ in range(depth):
    layer_1 = lasagne.layers.DenseLayer(in_layer, width, nonlinearity=nonlin)
    layer_2 = lasagne.layers.DenseLayer(layer_1, width, nonlinearity=nonlin)
    #    if drop_hidden:
    #        network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(layer_2, 10, nonlinearity=softmax)
    return network, layer_1, layer_2

# Prepare Theano variables for inputs and targets
input_var = T.tensor3('inputs')
target_var = T.ivector('targets')

def attack_main(rd):
    network= build_custom_mlp(input_var, int(DEPTH), int(WIDTH),
                                float(DROP_IN), float(DROP_HID),rd)
    # Loading the trained model
    script_dir=os.path.dirname(__file__)
    rel_path_m="Models/"
    abs_path_m=os.path.join(script_dir,rel_path_m)
    abs_path_m=os.path.join(abs_path_m,'model_FC10_'+str(DEPTH)
                                +'_'+str(WIDTH)+'_PCA_'+str(rd)+'.npz')
    #Load pre-computed weights for current network
    with np.load(abs_path_m) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    #Functions to predict class for a given input
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn=theano.function([input_var],T.argmax(test_prediction, axis=1))
    confidence=theano.function([input_var],test_prediction)
    model_names=['FC10'+str(1e-2),'hidden1']

    #Find class labels for adversarial examples for other networks
    #and count wrongly classified ones
    for string in model_names:
        count_tot=0.0
        count_wrong=0.0
        conf_wrong=0.0
        count_abs_wrong=0.0
        conf_abs=0.0
        count_adv=0.0
        conf_adv=0.0
        cross_examples=np.loadtxt('adv_examples_'+string+'.txt')
        cross_labels=np.loadtxt('adv_examples_labels_'+string+'.txt')
        for i in range(np.shape(cross_labels)[0]):
            curr_example=cross_examples[i,:]
            curr_example=curr_example.reshape((1,1,rd))
            curr_label=cross_labels[i,0]
            curr_label=curr_label.reshape((1,))
            prediction_curr=predict_fn(curr_example)
            if prediction_curr!=predict_fn(input_curr):
                count_wrong=count_wrong+1
                conf_wrong=conf_wrong+confidence(curr_example)[0][prediction_curr[0]]
            if prediction_curr!=y_curr and curr_label==predict_fn(input_curr):
                count_adv=count_adv+1
                conf_adv=conf_adv+confidence(curr_example)[0][prediction_curr[0]]
            if prediction_curr!=y_curr:
                count_abs_wrong=count_abs_wrong+1
                conf_abs=conf_abs+confidence(curr_example)[0][prediction_curr[0]]
        print ("% of misclassified examples on model "+string+" is {}".format(count_cross/np.shape(cross_labels)[0]*100))

num_epochs=500

DEPTH=2
WIDTH=100
DROP_IN=0
DROP_HID=0
