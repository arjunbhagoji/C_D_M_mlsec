import os
import numpy as np
import sys
import time
from sklearn import svm
from sklearn.decomposition import PCA
import multiprocessing

def svm_pca(rd):
    print('Running with PCA with {}'.format(rd))
    PCA_in_train=X_train
    PCA_in_test=X_test

    ### Doing PCA over the training data
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd)
    pca.fit(PCA_in_train)
    #Transforming the training, validation and test data
    X_train_dr=pca.transform(PCA_in_train)
    X_test_dr=pca.transform(PCA_in_test)
    X_test_recons=pca.inverse_transform(X_test_dr)

    clf_pca=svm.LinearSVC(dual=False)
    clf_pca.fit(X_train_dr,y_train)

    test_out_pca=clf_pca.predict(X_test_dr)
    test_success_pca=(2947.0-np.count_nonzero(test_out_pca-y_test))/2947.0

    test_out_recons=clf.predict(X_test_recons)
    test_success_recons=(2947.0-np.count_nonzero(test_out_recons-y_test))/2947.0

    resultfile=open(abs_path_o+'HAR_SVM_results.txt','a')
    resultfile.write(str(rd)+': '+str.format("{0:.3f}",test_success_pca)+'\n')
    resultfile.close()

    resultfile_2=open(abs_path_o+'HAR_SVM_results_recons.txt','a')
    resultfile_2.write(str(rd)+': '+str.format("{0:.3f}",test_success_recons)+'\n')
    resultfile_2.close()


    mag_count=0
    for DEV_MAG in np.linspace(0.1,1.0,no_of_mags):
        X_adv_dr=pca.transform(adv_examples_test[:,:,mag_count])
        rev_adv=pca.inverse_transform(X_adv_dr)
        count_pure_adv_2=0.0
        count_adv_2=0.0
        count_wrong_2=0.0
        count_correct_2=0.0
        for i in range(2947):
            x_ini_2=(X_test[i,:]).reshape((1,561))
            # x_ini_2=(X_test_dr[i,:]).reshape((1,rd))
            ini_class_2=clf.predict(x_ini_2)
            x_adv_2=rev_adv[i,:].reshape((1,561))
            # x_adv_2=X_adv_dr[i,:].reshape((1,rd))
            final_class_2=clf.predict(x_adv_2)
            if ini_class_2[0]==y_test[i]:
                count_correct_2=count_correct_2+1
            if ini_class_2[0]!=final_class_2[0]:
                count_adv_2=count_adv_2+1
            if y_test[i]!=final_class_2[0]:
                count_wrong_2=count_wrong_2+1
            if y_test[i]!=final_class_2[0] and ini_class_2[0]==y_test[i]:
                count_pure_adv_2=count_pure_adv_2+1
#print count_wrong_2/2947.0, count_adv_2/2947.0, count_pure_adv_2/2947.0
        plotfile=open(abs_path_o+'HAR_Gradient_attack_SVM_PCA_recons'+str(rd)+'.txt','a')
        # print("Deviation {} took {:.3f}s".format(
        #     DEV_MAG, time.time() - start_time))
        plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                        str(count_adv_2/2947.0*100.0)+","+
                        #str.format("{0:.3f}",conf_wrong/count_tot)+","+
                        str(count_wrong_2/2947.0*100.0)+","+
                        #str.format("{0:.3f}",conf_adv/count_tot)+","+
                        str(count_pure_adv_2/count_correct_2*100.0)+","+
                        #str.format("{0:.3f}",conf_abs/count_tot)+","+
                        str(0)+"\n")
        plotfile.close()
        mag_count=mag_count+1
    mag_count=0
    for DEV_MAG in np.linspace(0.1,1.0,10):
        X_adv_dr=pca.transform(adv_examples_train[:,:,mag_count])
        rev_adv=pca.inverse_transform(X_adv_dr)
        count_pure_adv=0.0
        count_adv=0.0
        count_wrong=0.0
        count_correct=0.0
        for i in range(7352):
            x_ini=(X_train[i,:]).reshape((1,561))
            # x_ini=(X_train_dr[i,:]).reshape((1,rd))
            ini_class=clf.predict(x_ini)
            # ini_class=clf_pca.predict(x_ini)
            x_adv=rev_adv[i,:].reshape((1,561))
            # x_adv=X_adv_dr[i,:].reshape((1,rd))
            final_class=clf.predict(x_adv)
            if ini_class[0]==y_train[i]:
                count_correct=count_correct+1
            if ini_class[0]!=final_class[0]:
                count_adv=count_adv+1
            if y_train[i]!=final_class[0]:
                count_wrong=count_wrong+1
            if y_train[i]!=final_class[0] and ini_class[0]==y_train[i]:
                count_pure_adv=count_pure_adv+1
        #print count_wrong/7352.0, count_adv/7352.0, count_pure_adv/7352.0
        plotfile=open(abs_path_o+'HAR_Gradient_attack_SVM_PCA_recons'+str(rd)+'.txt','a')
        print("Deviation {} took {:.3f}s".format(
            DEV_MAG, time.time() - start_time))
        plotfile.write(str(rd)+","+str(DEV_MAG)+","+
                        str(count_adv/7352.0*100.0)+","+
                        #str.format("{0:.3f}",conf_wrong/count_tot)+","+
                        str(count_wrong/7352.0*100.0)+","+
                        #str.format("{0:.3f}",conf_adv/count_tot)+","+
                        str(count_pure_adv/count_correct*100.0)+","+
                        #str.format("{0:.3f}",conf_abs/count_tot)+","+
                        str(1)+"\n")
        plotfile.close()
        mag_count=mag_count+1
    # mag_count=0
    # for DEV_MAG in np.linspace(0.1,1.0,no_of_mags):
    #     count_pure_adv_2=0.0
    #     count_adv_2=0.0
    #     count_wrong_2=0.0
    #     for i in range(2947):
    #         x_ini_2=(X_test_dr[i,:]).reshape((1,rd))
    #         ini_class_2=clf_pca.predict(x_ini_2)
    #         x_adv_2=(x_ini_2-DEV_MAG*(clf_pca.coef_[ini_class[0]-1,:]/(np.linalg.norm(clf_pca.coef_[ini_class[0]-1,:])))).reshape((1,rd))
    #         final_class_2=clf_pca.predict(x_adv_2)
    #         if ini_class_2[0]!=final_class_2[0]:
    #             count_adv_2=count_adv_2+1
    #         if y_test[i]!=final_class_2[0]:
    #             count_wrong_2=count_wrong_2+1
    #         if y_test[i]!=final_class_2[0] and ini_class_2[0]==y_test[i]:
    #             count_pure_adv_2=count_pure_adv_2+1
    #     plotfile=open(abs_path_o+'HAR_Gradient_attack_SVM_PCA_retrain_'+str(rd)+'.txt','a')
    #     plotfile.write(str(rd)+","+str(DEV_MAG)+","+
    #                     str(count_adv_2/2947.0*100.0)+","+
    #                     #str.format("{0:.3f}",conf_wrong/count_tot)+","+
    #                     str(count_wrong_2/2947.0*100.0)+","+
    #                     #str.format("{0:.3f}",conf_adv/count_tot)+","+
    #                     str(count_pure_adv_2/2947.0*100.0)+","+
    #                     #str.format("{0:.3f}",conf_abs/count_tot)+","+
    #                     str(0)+"\n")
    #     plotfile.close()
    #     mag_count=mag_count+1

script_dir=os.getcwd()
rel_path="Input_data/UCI_HAR/"
abs_path=os.path.join(script_dir,rel_path)
rel_path_o="Output_data/"
abs_path_o=os.path.join(script_dir, rel_path_o)

KERNEL='linear'
METHOD='ovr'


X_train=np.loadtxt(abs_path+'train/X_train.txt')
y_train=np.loadtxt(abs_path+'train/y_train.txt')
X_test=np.loadtxt(abs_path+'test/X_test.txt')
y_test=np.loadtxt(abs_path+'test/y_test.txt')

clf=svm.LinearSVC(dual=False)
clf.fit(X_train,y_train)
test_out=clf.predict(X_test)
test_success=(2947.0-np.count_nonzero(test_out-y_test))/2947.0

resultfile=open(abs_path_o+'HAR_SVM_results.txt','a')
resultfile.write('##################################################'+'\n')
resultfile.write('Solver: LinearSVC, Kernel: '+str(KERNEL)
                    +', Method: '+str(METHOD)+'\n')
resultfile.write('##################################################'+'\n')
resultfile.write('rd: test_success'+'\n')
resultfile.write('no_pca: '+str.format("{0:.3f}",test_success)+'\n')
resultfile.close()

#Computing and storing adv. examples
no_of_mags=10

plotfile=open(abs_path_o+'HAR_Gradient_attack_SVM_no_PCA_'+'.txt','a')
plotfile.write('rd,Dev,Adv,Wrong,pure_adv,Train \n')
plotfile.close()

adv_examples_test=np.zeros((2947,561,no_of_mags))
adv_examples_train=np.zeros((7352,561,no_of_mags))

mag_count=0
for DEV_MAG in np.linspace(0.1,1.0,10):
    count_pure_adv=0.0
    count_adv=0.0
    count_wrong=0.0
    count_correct=0.0
    start_time=time.time()
    for i in range(2947):
        x_ini=(X_test[i,:]).reshape((1,561))
        ini_class=clf.predict(x_ini)
        x_adv=(x_ini-DEV_MAG*(clf.coef_[ini_class[0]-1,:]/(np.linalg.norm(clf.coef_[ini_class[0]-1,:])))).reshape((1,561))
        adv_examples_test[i,:,mag_count]=x_adv
        final_class=clf.predict(x_adv)
        if ini_class[0]==y_test[i]:
            count_correct=count_correct+1
        if ini_class[0]!=final_class[0]:
            count_adv=count_adv+1
        if y_test[i]!=final_class[0]:
            count_wrong=count_wrong+1
        if y_test[i]!=final_class[0] and ini_class[0]==y_test[i]:
            count_pure_adv=count_pure_adv+1
    #print count_wrong/2947.0, count_adv/2947.0, count_pure_adv/2947.0
    plotfile=open(abs_path_o+'HAR_Gradient_attack_SVM_no_PCA_'+'.txt','a')
    print("Deviation {} took {:.3f}s".format(
        DEV_MAG, time.time() - start_time))
    plotfile.write("no_dr"+","+str(DEV_MAG)+","+
                    str(count_adv/2947.0*100.0)+","+
                    #str.format("{0:.3f}",conf_wrong/count_tot)+","+
                    str(count_wrong/2947.0*100.0)+","+
                    #str.format("{0:.3f}",conf_adv/count_tot)+","+
                    str(count_pure_adv/count_correct*100.0)+","+
                    #str.format("{0:.3f}",conf_abs/count_tot)+","+
                    str(0)+"\n")
    plotfile.close()
    mag_count=mag_count+1
mag_count=0
for DEV_MAG in np.linspace(0.1,1.0,10):
    count_pure_adv=0.0
    count_adv=0.0
    count_wrong=0.0
    count_correct=0.0
    for i in range(7352):
        x_ini=(X_train[i,:]).reshape((1,561))
        ini_class=clf.predict(x_ini)
        x_adv=(x_ini-DEV_MAG*(clf.coef_[ini_class[0]-1,:]/(np.linalg.norm(clf.coef_[ini_class[0]-1,:])))).reshape((1,561))
        adv_examples_train[i,:,mag_count]=x_adv
        final_class=clf.predict(x_adv)
        if ini_class[0]==y_train[i]:
            count_correct=count_correct+1
        if ini_class[0]!=final_class[0]:
            count_adv=count_adv+1
        if y_train[i]!=final_class[0]:
            count_wrong=count_wrong+1
        if y_train[i]!=final_class[0] and ini_class[0]==y_train[i]:
            count_pure_adv=count_pure_adv+1
    #print count_wrong/7352.0, count_adv/7352.0, count_pure_adv/7352.0
    plotfile=open(abs_path_o+'HAR_Gradient_attack_SVM_no_PCA_'+'.txt','a')
    print("Deviation {} took {:.3f}s".format(
        DEV_MAG, time.time() - start_time))
    plotfile.write("no_dr"+","+str(DEV_MAG)+","+
                    str(count_adv/7352.0*100.0)+","+
                    #str.format("{0:.3f}",conf_wrong/count_tot)+","+
                    str(count_wrong/7352.0*100.0)+","+
                    #str.format("{0:.3f}",conf_adv/count_tot)+","+
                    str(count_pure_adv/7352.0*100.0)+","+
                    #str.format("{0:.3f}",conf_abs/count_tot)+","+
                    str(1)+"\n")
    plotfile.close()
    mag_count=mag_count+1

rd_list=[561,155,100,50,40,30,20,10]
#rd_list=[784]
#svm_pca(784)


pool=multiprocessing.Pool(processes=8)
pool.map(svm_pca,rd_list)
pool.close()
pool.join()
resultfile=open(abs_path_o+'HAR_SVM_results.txt','a')
resultfile.close()
