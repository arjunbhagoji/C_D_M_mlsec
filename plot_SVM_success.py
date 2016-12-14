import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import os


font ={'size': 17}

matplotlib.rc('font', **font)
#Plot second figure

fig, ax = plt.subplots(1,1,figsize=(12,9))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

x=np.arange(9)
#labels=['10','20','30','40','50','100','155','561','no_PCA']
labels=['10','20','30','40','50','100','331','784']

script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

curr_array=np.loadtxt(abs_path_o+'MNIST_SVM_results_recons.txt',delimiter=',')
recons,=plt.plot(x,curr_array[:,2]*100,label='Reconstructed',marker='o')

curr_array=np.genfromtxt(abs_path_o+'MNIST_SVM_results.txt',skip_header=5,delimiter=',')
retrained,=plt.plot(x[0:8],curr_array[:,2]*100,label='Retrained',marker='x')

theo_limit=np.array(91.5)
y=np.tile(theo_limit,len(labels))
orig,=plt.plot(x[0:8],y,color='black',label='No defense')


xticks, xticklabels=plt.xticks()
plt.xlabel('Reduced dimension after PCA')
plt.ylabel('Classification success on test set')
plt.title('Classification success vs. PCA dimensions \n Model: Linear SVM')

plt.legend(handles=[recons,retrained,orig],loc=7)

# xmin = (3*xticks[0] - xticks[1])/2.
# # shaft half a step to the right
# xmax = (3*xticks[-1] - xticks[-2])/2.
# plt.xlim(xmin, xmax)
plt.xticks(xticks,labels)

script_dir=os.path.dirname(__file__)
rel_path_p="Plots/"
abs_path_p=os.path.join(script_dir,rel_path_p)

plt.savefig(abs_path_p+'MNIST_Class_success_PCA_SVM.png', bbox_inches='tight')
