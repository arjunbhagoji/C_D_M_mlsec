import numpy as np
from matplotlib import pyplot as plt
import glob as glob

fig, ax = plt.subplots(1, 1,figsize=(12,9))

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

for item in glob.glob('FSG_plots_data_hidden_*.txt'):
    curr_array=np.loadtxt(item)
    plt.plot(curr_array[:,1],curr_array[:,2],label=str(int(curr_array[0,0])))

curr_array=np.loadtxt('FSG_plots_data_hidden.txt')
plt.plot(curr_array[:,0],curr_array[:,1],label='noPCA')

plt.xlabel(r'Adversarial perturbation $\epsilon$')
plt.ylabel('Misclassification percentage')
plt.title('Adversarial example success vs. perturbation magnitude \n'
            ' for various reduced dimensions using the '
            'fast sign gradient method \n Model: FC_2_100')
plt.legend()
#plt.show()
plt.savefig('adv_success_hidden.png', bbox_inches='tight')
