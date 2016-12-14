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

x=np.arange(8)
#labels=['10','20','30','40','50','100','155','561','no_PCA']
labels=['10','20','30','40','50','100','331','784']

script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

curr_array=np.loadtxt(abs_path_o+'L_BFGS_retrain_unaware.txt',delimiter=',')
retrained,=plt.plot(x,curr_array[:,1],label='Re-training',marker='o')

curr_array=np.loadtxt(abs_path_o+'L_BFGS_recons.txt',delimiter=',')
recons,=plt.plot(x,curr_array[:,1],label='Reconstruction',marker='o')

theo_limit=np.array(99.977)
y=np.tile(theo_limit,len(labels))
orig,=plt.plot(x,y,color='black',label='No defense', marker='o')


xticks, xticklabels=plt.xticks()
plt.xlabel('Reduced dimension after PCA')
plt.ylabel('Adversarial success')
plt.title('Adversarial success for optimization based attack vs. PCA dimensions (MNIST) \n Model: FC100-100-10')

plt.legend(handles=[recons,retrained,orig],loc=6)

# xmin = (3*xticks[0] - xticks[1])/2.
# # shaft half a step to the right
# xmax = (3*xticks[-1] - xticks[-2])/2.
# plt.xlim(xmin, xmax)
# plt.ylim(0,100)
plt.xticks(xticks,labels)

script_dir=os.path.dirname(__file__)
rel_path_p="Plots/"
abs_path_p=os.path.join(script_dir,rel_path_p)

plt.savefig(abs_path_p+'MNIST_adv_BFGS_success_PCA_NN_2.png', bbox_inches='tight')
