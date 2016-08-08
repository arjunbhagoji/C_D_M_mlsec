import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

# from matplotlib import pyplot as plt
import glob as glob
import os
from matplotlib.pyplot import cm
from cycler import cycler

cm=plt.get_cmap('gist_rainbow')
NUM_COLORS=9

script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

# plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':', '-.'])))

fig, ax = plt.subplots(1, 1,figsize=(12,9))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_prop_cycle(cycler('color',[cm(1.*i/9) for i in range(9)])*cycler('linestyle',['-', '--', ':']))

handle_list=[]

for item in glob.glob(abs_path_o+'Gradient_attack_SVM_PCA_*.txt'):
    curr_array=np.genfromtxt(item,skip_header=1,delimiter=',')
    handle_list.append(plt.plot(curr_array[:,1],curr_array[:,2],label=int(curr_array[0,0])))

curr_array=np.genfromtxt(abs_path_o+'Gradient_attack_SVM_no_PCA_.txt',
skip_header=1,delimiter=',')
plt.plot(curr_array[:,1],curr_array[:,2],label='noPCA')

plt.xlabel(r'Adversarial perturbation $\epsilon$')
plt.ylabel('Misclassification percentage')
plt.title('Adversarial example success vs. perturbation magnitude \n'
            ' for various reduced dimensions using the '
            'gradient attack \n Model: Linear_SVM')

# handles, labels = ax.get_legend_handles_labels()
# # sort both labels and handles by labels
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# ax.legend(handles, labels)

plt.legend(loc=4)
#plt.show()
plt.savefig('adv_success_linear_svm.png', bbox_inches='tight')
