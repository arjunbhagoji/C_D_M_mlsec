import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# from matplotlib import pyplot as plt
import glob as glob
import os
from matplotlib.pyplot import cm
from cycler import cycler

font ={'size': 17}

matplotlib.rc('font', **font)

cm=plt.get_cmap('gist_rainbow')
NUM_COLORS=9

script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

# plt.rc('axes', prop_cycle=(cycler('linestyle', ['-', '--', ':', '-.'])))

fig, ax = plt.subplots(1, 1,figsize=(12,9))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
# ax.set_prop_cycle(cycler('color',[cm(1.*i/3) for i in range(3)])*cycler('linestyle',['-', '--', ':','-.']))
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
markers = []

for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass


handle_list=[]

path_curr=abs_path_o+'strategic_heuristic_results.txt'
curr_array=np.genfromtxt(path_curr,delimiter=' ',skip_header=0)*100

x_axis=np.arange(5)+1.0
# list_dim=[10,20,30,40,50,100,155]
list_dim=[331,100,50,40,30,20,10]
#for item in glob.glob(abs_path_o+'Gradient_attack_SVM_PCA_clean_*.txt'):
count=0
for item in list_dim:
    print item
    count=count+1
    color = colors[count % len(colors)]
    style = markers[count % len(markers)]
    handle_list.append(plt.plot(x_axis,curr_array[(count-1)*5:count*5,2],linestyle='-', marker=style, color=color, markersize=10,label=item))

curr_array=np.genfromtxt(abs_path_o+'no_dr_strategic.txt',skip_header=0,delimiter=' ')*100
plt.plot(x_axis,curr_array[:,2],label='No defense',marker='o',markersize=10)

# theo_limit=np.array((1.0-0.915)*100)
# y=np.tile(theo_limit,len(curr_array))
# plt.plot(curr_array[:,1],y,color='black',marker='o',label='SVM limit')

plt.xlabel(r'Deviation multiplier $\alpha$ for strategic dimensions')
plt.ylabel('Adversarial success')
plt.title('Re-training defense for MNIST data against strategic attack targeting PCA \n Model: Linear SVM')
# plt.ylim(0.0,100.0)
# plt.xlim(0.01,0.1)
plt.xticks([1.0,2.0,3.0,4.0,5.0])

# handles, labels = ax.get_legend_handles_labels()
# # sort both labels and handles by labels
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# ax.legend(handles, labels)

script_dir=os.path.dirname(__file__)
rel_path_p="Plots/"
abs_path_p=os.path.join(script_dir,rel_path_p)

plt.legend(loc='upper left',fontsize=14)
#plt.show()
plt.savefig(abs_path_p+'MNIST_SVM_PCA_retrain_strategic.png', bbox_inches='tight')
