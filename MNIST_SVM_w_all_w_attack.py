import numpy as np
import sys
import os
from sklearn import svm
from sklearn.decomposition import PCA
import multiprocessing
import time

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
        #if not os.path.exists(filename):
        #    download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 784)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        #if not os.path.exists(filename):
        #    download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    script_dir=os.getcwd()
    rel_path="Input_data/"
    abs_path=os.path.join(script_dir,rel_path)
    X_train = load_mnist_images(abs_path+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(abs_path+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(abs_path+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(abs_path+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ### Success rate on MNIST dataset without PCA
def svm_no_pca():
    print('Running with no PCA')
    X_train, y_train, X_val, y_val, X_test, y_test=load_dataset()
    X_train=X_train.reshape((50000,784))
    clf=svm.LinearSVC(dual=False)
    #clf=svm.SVC(kernel=KERNEL,decision_function_shape=METHOD)
    clf.fit(X_train,y_train)

    X_val=X_val.reshape((10000,784))
    val_out=out=clf.predict(X_val)
    validation_success=(10000-np.count_nonzero(val_out-y_val))/10000.0

    X_test=X_test.reshape((10000,784))
    test_out=clf.predict(X_test)
    test_success=(10000-np.count_nonzero(test_out-y_test))/10000.0

    resultfile=open(abs_path_o+'SVM_results.txt','a')
    resultfile.write('##################################################'+'\n')
    resultfile.write('Solver: LinearSVC, Kernel: '+str(KERNEL)
                        +', Method: '+str(METHOD)+'\n')
    resultfile.write('##################################################'+'\n')
    resultfile.write('rd: val_success, test_success'+'\n')
    resultfile.write('no_pca: '+str.format("{0:.3f}",validation_success)+
                        ','+str.format("{0:.3f}",test_success)+'\n')
    resultfile.close()

    #Computing and storing adv. examples
    no_of_mags=10
    adv_examples=np.zeros((50000,784,no_of_mags))

    plotfile=open(abs_path_o+'Gradient_attack_SVM_no_PCA_'+'.txt','a')
    plotfile.write('rd,Dev,Adv,Wrong,pure_adv,Train \n')
    plotfile.close()
    mag_count=0

    for DEV_MAG in np.linspace(0.1,1.0,10):
        #print_flag=0
        count_pure_adv=0.0
        count_adv=0.0
        count_wrong=0.0
        start_time=time.time()
        for i in range(50000):
            x_ini=(X_train[i,:]).reshape((1,784))
            ini_class=clf.predict(x_ini)
            x_adv=(x_ini-DEV_MAG*(clf.coef_[ini_class[0],:]/
                (np.linalg.norm(clf.coef_[ini_class[0],:])))).reshape((1,784))
            adv_examples[i,:,mag_count]=x_adv
            final_class=clf.predict(x_adv)
            if ini_class[0]!=final_class[0]:
                count_adv=count_adv+1
            if y_train[i]!=final_class[0]:
                count_wrong=count_wrong+1
            if y_train[i]!=final_class[0] and ini_class[0]==y_train[i]:
                count_pure_adv=count_pure_adv+1
            # if y_train[i]!=final_class[0] and ini_class[0]==y_train[i] and print_flag==0:
            #     #plt.imshow((x_ini.reshape((28,28)))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            #     plt.imshow(((x_adv-x_ini).reshape((28,28)))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
            #     print_flag=print_flag+1
        plotfile=open(abs_path_o+'Gradient_attack_SVM_no_PCA_'+'.txt','a')
        print("Deviation {} took {:.3f}s".format(
            DEV_MAG, time.time() - start_time))
        plotfile.write("no_dr"+","+str(DEV_MAG)+","+
                        str(count_wrong/50000.0*100.0)+","+
                        #str.format("{0:.3f}",conf_wrong/count_tot)+","+
                        str(count_adv/50000.0*100.0)+","+
                        #str.format("{0:.3f}",conf_adv/count_tot)+","+
                        str(count_pure_adv/50000.0*100.0)+","+
                        #str.format("{0:.3f}",conf_abs/count_tot)+","+str(1)+
                        "\n")
        plotfile.close()
        mag_count=mag_count+1

### Success rate on MNIST dataset with PCA
def svm_pca(rd):
    print('Running with PCA with {}'.format(rd))
    X_train, y_train, X_val, y_val, X_test, y_test=load_dataset()
    rd=rd
    PCA_in_train=X_train.reshape(50000,784)
    PCA_in_val=X_val.reshape(10000,784)
    PCA_in_test=X_test.reshape(10000,784)

    ### Doing PCA over the training data
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd)
    pca_train=pca.fit(PCA_in_train)
    #Transforming the training, validation and test data
    X_train_dr=pca.transform(PCA_in_train).reshape((50000,rd))
    X_test_dr=pca.transform(PCA_in_test).reshape((10000,rd))
    X_val_dr=pca.transform(PCA_in_val).reshape((10000,rd))

    clf_pca=svm.LinearSVC(dual=False)
    #clf_pca=svm.SVC(kernel='linear',decision_function_shape=METHOD)
    clf_pca.fit(X_train_dr,y_train)
    val_out_pca=clf_pca.predict(X_val_dr)
    validation_success_pca=(10000-np.count_nonzero(val_out_pca-y_val))/10000.0

    test_out_pca=clf_pca.predict(X_test_dr)
    test_success_pca=(10000-np.count_nonzero(test_out_pca-y_test))/10000.0

    resultfile=open(abs_path_o+'SVM_results.txt','a')
    resultfile.write(str(rd)+': '+str.format("{0:.3f}",validation_success_pca)
                        +','+str.format("{0:.3f}",test_success_pca)+'\n')
    # Creating adversarial examples
    no_of_mags_pca=50
    adv_examples=np.zeros((50000,784,no_of_mags_pca))

    plotfile=open(abs_path_o+'Gradient_attack_SVM_PCA_'+str(rd)+'.txt','a')
    plotfile.write('rd,Dev,Adv,Wrong,pure_adv,Train \n')
    plotfile.close()
    mag_count=0

    for DEV_MAG in np.linspace(0.1,10,no_of_mags_pca):
        count_pure_adv_pca=0.0
        count_adv_pca=0.0
        count_wrong_pca=0.0
        for i in range(50000):
            x_ini=(X_train_dr[i,:]).reshape((1,rd))
            ini_class=clf_pca.predict(x_ini)
            x_adv=(x_ini-DEV_MAG*(clf_pca.coef_[ini_class[0],:]/(np.linalg.norm(clf_pca.coef_[0,:])))).reshape((1,rd))
            #x_adv=X_adv_dr[i,:].reshape((1,rd))
            final_class=clf_pca.predict(x_adv)
            if ini_class[0]!=final_class[0]:
                count_adv_pca=count_adv_pca+1
            if y_train[i]!=final_class[0]:
                count_wrong_pca=count_wrong_pca+1
            if y_train[i]!=final_class[0] and ini_class[0]==y_train[i]:
                count_pure_adv_pca=count_pure_adv_pca+1
            #if y_train[i]!=final_class[0] and ini_class[0]==y_train[i] and print_flag==0:
                #plt.imshow((x_ini.reshape((28,28)))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                #plt.imshow(((x_adv-x_ini).reshape((28,28)))*255, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                #print_flag=print_flag+1
        plotfile=open(abs_path_o+'Gradient_attack_SVM_PCA_'+str(rd)+'.txt','a')
        # print("Deviation {} took {:.3f}s".format(
        #     DEV_MAG, time.time() - start_time))
        plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                        str(count_wrong_pca/50000.0*100.0)+","+
                        #str.format("{0:.3f}",conf_wrong/count_tot)+","+
                        str(count_adv_pca/50000.0*100.0)+","+
                        #str.format("{0:.3f}",conf_adv/count_tot)+","+
                        str(count_pure_adv_pca/50000.0*100.0)+","+
                        #str.format("{0:.3f}",conf_abs/count_tot)+","+str(1)+
                        "\n")
        plotfile.close()
        mag_count=mag_count+1


script_dir=os.path.dirname(__file__)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir,rel_path_o)

KERNEL='linear'
METHOD='ovr'

svm_no_pca()

rd_list=[784,331,100,50,40,30,20,10]

pool=multiprocessing.Pool(processes=8)
pool.map(svm_pca,rd_list)
pool.close()
pool.join()
resultfile=open(abs_path_o+'SVM_results.txt','a')
resultfile.close()