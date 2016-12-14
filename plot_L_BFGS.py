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

x=np.arange(7)
#labels=['10','20','30','40','50','100','155','561','no_PCA']
labels=['10','20','30','40','50','100','331']

script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

curr_array=np.loadtxt(abs_path_o+'L_BFGS_recons.txt',delimiter=',')
recons,=plt.plot(x,curr_array[:7,1],label='Reconstructed',marker='o',markersize=10)

curr_array=np.genfromtxt(abs_path_o+'L_BFGS_retrain_unaware.txt',skip_header=0,delimiter=',')
retrained,=plt.plot(x,curr_array[:7,1],label='Re-trained',marker='x',markersize=10)

theo_limit=np.array(99.98)
y=np.tile(theo_limit,len(labels))
orig,=plt.plot(x,y,color='black',label='No defense',marker='o')


xticks, xticklabels=plt.xticks()
plt.xlabel('Reduced dimension after PCA')
plt.ylabel('Adversarial success')
plt.title('Adversarial success vs. PCA dimensions \n Model: FC100-100-10, Attack: Optimization-based')

plt.legend(handles=[recons,retrained,orig],loc=2)

# xmin = (3*xticks[0] - xticks[1])/2.
# # shaft half a step to the right
# xmax = (3*xticks[-1] - xticks[-2])/2.
# plt.xlim(xmin, xmax)
# plt.ylim(0,100)
plt.xticks(xticks,labels)

script_dir=os.path.dirname(__file__)
rel_path_p="Plots/"
abs_path_p=os.path.join(script_dir,rel_path_p)

plt.savefig(abs_path_p+'MNIST_L_BFGS_PCA_NN_2.png', bbox_inches='tight')
