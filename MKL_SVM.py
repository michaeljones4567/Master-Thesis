# Title - MKL applied to Tuberculousis transcription data with a subset of features
# Author - Michael Jones, UCL

import numpy as np
import code
import matplotlib.pyplot as plt

from MKLpy.lists import HPK_generator
from MKLpy.lists import SFK_generator

from MKLpy.arrange import summation

import scipy
from scipy.spatial.distance import pdist, squareform

from MKLpy.regularization import rescale_01, normalization
from MKLpy.algorithms import EasyMKL
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from MKLpy.metrics.pairwise import HPK_kernel as HPK
from MKLpy.metrics import radius,margin

from sklearn.metrics import roc_auc_score

from itertools import combinations

#===============================================================================#
# Parameter choosing

Metric = 'Total_score'

Experiments = np.array(['Cured'])
ML_Methods = np.array(['OSCAR'])

Number_of_repeat_runs = 30

Size_of_combinations = 3

Training_proportion = 0.7

# MKL Parameters

Max_Width = 6.1
Min_Width = 0.1
Number_of_widths = 20

#===============================================================================#

def my_kernel(X,Y,sigma):
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            K[i,j] = np.exp(-(1/(sigma ** 2))*np.linalg.norm(x-y)**2)
    return K

def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

for aacount in range(0,1):

    for bbcount in range(0,1):

        Experiment = Experiments[aacount]
        ML_Method = ML_Methods[bbcount]

        if Experiment == 'Cured':
        # Load training data
            Data_uncleaned = np.genfromtxt('JRTB0&2y.txt',dtype=None) # 13720, 117

            # Split into infected patient data and uninfected patient data
            Number_of_data_points_TB = 46
            Number_of_data_points_Clear = 31
            Total_data_points = Number_of_data_points_TB + Number_of_data_points_Clear

            Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
            Data_cleaned = np.transpose(Data_cleaned) # 116, 13718
            Gene_names = Data_uncleaned[2:,0]

        if Experiment == 'Fever':

            Data_uncleaned = np.genfromtxt('JRTB0&Fever.txt',dtype=None) # 13720, 117

            # Split into infected patient data and uninfected patient data
            Number_of_data_points_TB = 46
            Number_of_data_points_Clear = 70
            Total_data_points = Number_of_data_points_TB + Number_of_data_points_Clear

            Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
            Data_cleaned = np.transpose(Data_cleaned) # 116, 13718
            Gene_names = Data_uncleaned[2:,0]

        if Experiment == 'Full':

            Data_uncleaned = np.genfromtxt('JRTB0&2y.txt',dtype=None) # 13720, 117
            Data_uncleaned_2 = np.genfromtxt('JRTB0&Fever.txt',dtype=None) # 13720, 117

            Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
            Data_cleaned = np.transpose(Data_cleaned) # 116, 13718

            Data_cleaned_2 = Data_uncleaned_2[2:,1:] # 13718, 116
            Data_cleaned_2 = np.transpose(Data_cleaned_2) # 116, 13718


            Gene_names = Data_uncleaned[2:,0]

            # Split into infected patient data and uninfected patient data
            Number_of_data_points_TB = 46
            Number_of_data_points_Clear = 101
            Total_data_points = Number_of_data_points_TB + Number_of_data_points_Clear

            X_TB = Data_cleaned[0:Number_of_data_points_TB,:]
            X_Clear = Data_cleaned[Number_of_data_points_TB:,:]
            X_Fever = Data_cleaned_2[Number_of_data_points_TB:,:]

            Data_cleaned = np.concatenate([X_TB, X_Clear, X_Fever], axis=0)

        if Experiment == 'Cured':

            if ML_Method == 'SCoRS':

                Number_of_genes_to_include = 13

                if Metric == 'Top_10':

                    Number_of_genes_to_include = 16

                    args = np.load(Experiment + '_Top_survival_scores_By_Stability.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]

                if Metric == 'Total_score':

                    #Number_of_genes_to_include = 21
                    Number_of_genes_to_include = 10

                    args = np.load(Experiment + '_scores_By_Stability.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]


            if ML_Method == 'OSCAR':

                Number_of_genes_to_include = 10

                if Metric == 'Top_10':

                    Number_of_genes_to_include = 20

                    args = np.load(Experiment + '_Top_survival_scores_By_Oscar.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]

                if Metric == 'Total_score':

                    #Number_of_genes_to_include = 12
                    Number_of_genes_to_include = 10

                    args = np.load(Experiment + '_scores_By_Oscar.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]

            if ML_Method == 'OWL':

                if Metric == 'Top_10':

                    Number_of_genes_to_include = 15

                    args = np.load(Experiment + '_Top_10_survival_scores_By_OWL.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]

                if Metric == 'Total_score':

                    #Number_of_genes_to_include = 11
                    Number_of_genes_to_include = 10

                    args = np.load(Experiment + '_scores_By_OWL.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]

            if ML_Method == 'L1L2':

                Number_of_genes_to_include = 10
                args = np.load('L1L2norm_score_results_Cured.npy')
                args = np.abs(args)
                args = np.argsort(args)
                args = args[-Number_of_genes_to_include:]

        if Experiment == 'Fever':

                if ML_Method == 'SCoRS':

                    Number_of_genes_to_include = 10

                    if Metric == 'Top_10':

                        Number_of_genes_to_include = 11

                        args = np.load(Experiment + '_Top_survival_scores_By_Stability.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                    if Metric == 'Total_score':

                        #Number_of_genes_to_include = 19
                        Number_of_genes_to_include = 10

                        args = np.load(Experiment + '_scores_By_Stability.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                if ML_Method == 'OSCAR':

                    Number_of_genes_to_include = 16

                    if Metric == 'Top_10':

                        Number_of_genes_to_include = 16

                        args = np.load(Experiment + '_Top_survival_scores_By_Stability.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                    if Metric == 'Total_score':

                        #Number_of_genes_to_include = 13
                        Number_of_genes_to_include = 10

                        args = np.load(Experiment + '_scores_By_Stability.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                if ML_Method == 'OWL':

                    Number_of_genes_to_include = 10

                    #Number_of_genes_to_include = 14

                    if Metric == 'Top_10':

                        #Number_of_genes_to_include = 11
                        Number_of_genes_to_include = 10

                        args = np.load(Experiment + '_Top_10_survival_scores_By_OWL.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                    if Metric == 'Total_score':

                        #Number_of_genes_to_include = 11
                        Number_of_genes_to_include = 10

                        args = np.load(Experiment + '_scores_By_OWL.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                if ML_Method == 'L1L2':

                    Number_of_genes_to_include = 10
                    args = np.load('L1L2norm_score_results_Fever.npy')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]

        if Experiment == 'Full':

                if ML_Method == 'SCoRS':

                    Number_of_genes_to_include = 11

                    if Metric == 'Top_10':

                        #Number_of_genes_to_include = 11

                        args = np.load(Experiment + '_Top_10_By_Stability.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                    if Metric == 'Total_score':

                        #Number_of_genes_to_include = 19
                        Number_of_genes_to_include = 10

                        args = np.load(Experiment + '_scores_By_Stability.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                if ML_Method == 'OSCAR':

                    Number_of_genes_to_include = 10

                    if Metric == 'Top_10':

                        umber_of_genes_to_include = 18
                        args = np.load(Experiment + '_Top_10_By_Oscar.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                    if Metric == 'Total_score':

                        Number_of_genes_to_include = 10
                        #Number_of_genes_to_include = 16

                        args = np.load(Experiment + '_scores_By_Oscar.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]


                if ML_Method == 'OWL':

                    #args = np.genfromtxt('OWL_results_Fever_Total_Weights_1')
                    #args = np.argsort(args)

                    #Number_of_genes_to_include = 14

                    if Metric == 'Top_10':

                        #Number_of_genes_to_include = 19

                        args = np.load(Experiment + '_Top_10_survival_scores_By_OWL.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                    if Metric == 'Total_score':

                        #Number_of_genes_to_include = 14
                        Number_of_genes_to_include = 10

                        args = np.load(Experiment + '_scores_By_OWL.npy')
                        args = np.argsort(args)
                        args = args[-Number_of_genes_to_include:]

                if ML_Method == 'L1L2':

                    args = np.genfromtxt('L1L2norm_score_results_Fever')
                    args = np.argsort(args)
                    args = args[-Number_of_genes_to_include:]


        #===============================================================================#
        # Combinations chooser

        indexes_list = np.arange(0,Number_of_genes_to_include)

        Combinations = combinations(indexes_list,Size_of_combinations)
        Combinations = list(Combinations)
        Combinations = np.array(Combinations).astype(np.int)

        total_loops = len(Combinations)

        #===============================================================================#

        Y1 = np.ones([Number_of_data_points_TB,1])
        Y2 = np.zeros([Number_of_data_points_Clear,1])
        Y = np.concatenate((Y1, Y2), axis=0)


        Max_training_TB = np.round(Number_of_data_points_TB*Training_proportion).astype(int)
        Max_training_2y = np.round(Number_of_data_points_Clear*Training_proportion).astype(int)

        Training_size = Max_training_TB + Max_training_2y
        Test_size = (Number_of_data_points_TB - Max_training_TB) + (Number_of_data_points_Clear - Max_training_2y)

        overall_predictions_storer = np.zeros([Number_of_repeat_runs,total_loops])

        for acount in range(0,Number_of_repeat_runs):

            print(acount)

            random_numbers_TB = np.random.choice(range(0,Number_of_data_points_TB),Number_of_data_points_TB,replace=False)
            random_numbers_2y = np.random.choice(range(46,Total_data_points),Number_of_data_points_Clear,replace=False)

            random_numbers_TB_tr = random_numbers_TB[0:Max_training_TB]
            random_numbers_TB_te = random_numbers_TB[Max_training_TB:Number_of_data_points_TB]

            random_numbers_2y_tr = random_numbers_2y[0:Max_training_2y]
            random_numbers_2y_te = random_numbers_2y[Max_training_2y:Number_of_data_points_Clear]

            storer_counter = 0

            predictions_storer = np.zeros([total_loops,Test_size])

            for icount in range(0,total_loops):
                print(acount, icount, total_loops)
                args_temp = args[Combinations[icount]].astype(np.int)
                ##

                Top_TB = Data_cleaned[:,args_temp]
                Top_TB = Top_TB.astype(np.float)
                Top_TB = np.squeeze(Top_TB)

                X = Top_TB
                X = np.reshape(X,[Total_data_points,-1])

                X1_tr = X[random_numbers_TB_tr,:]
                X1_te = X[random_numbers_TB_te,:]

                X2_tr = X[random_numbers_2y_tr,:]
                X2_te = X[random_numbers_2y_te,:]

                Xtr = np.concatenate((X1_tr, X2_tr), axis=0)
                Xte = np.concatenate((X1_te, X2_te), axis=0)

                # Label infected patient data and uninfected patient data
                Y1 = np.ones([Number_of_data_points_TB,1])
                Y2 = np.zeros([Number_of_data_points_Clear,1])
                Y = np.concatenate((Y1, Y2), axis=0)

                Y1_tr = Y[random_numbers_TB_tr,:]
                Y1_te = Y[random_numbers_TB_te,:]

                Y2_tr = Y[random_numbers_2y_tr,:]
                Y2_te = Y[random_numbers_2y_te,:]

                Ytr = np.concatenate((Y1_tr, Y2_tr), axis=0)
                Yte = np.concatenate((Y1_te, Y2_te), axis=0)

                ##

                #print(Training_size)

                K_list_tr = np.zeros([Number_of_widths,Training_size,Training_size])
                counter = 0

                for jcount in np.arange(Min_Width,Max_Width,(Max_Width-Min_Width)/Number_of_widths):

                    K_list_tr[counter,:,:] = my_kernel(Xtr,Xtr,jcount)
                    counter += 1

                K_list_tr_te = np.zeros([Number_of_widths,Test_size,Training_size])
                counter = 0

                for jcount in np.arange(Min_Width,Max_Width,(Max_Width-Min_Width)/Number_of_widths):

                    K_list_tr_te[counter,:,:] = my_kernel(Xte,Xtr,jcount)
                    counter += 1

                ax = EasyMKL(lam=0.1, kernel='precomputed')
                ker_matrix_tr = ax.arrange_kernel(K_list_tr,Ytr)

                kernel_weights = ax.weights
                kernel_weights = np.reshape(kernel_weights,[-1,1,1])
                K_tr =  np.multiply(kernel_weights,K_list_tr_te)
                K_tr = np.sum(K_tr,axis=0)

                clf = SVC(C=2, kernel='precomputed').fit(ker_matrix_tr,Ytr)
                predictions = clf.predict(K_tr)
                predictions_storer[icount,:] = predictions
                #print(icount)
                ##
                v = predictions == Yte.T
                v.astype(np.float)
                c = np.sum(v, axis=1)
                #print(c)
                #print(predictions)

            v = predictions_storer == Yte.T
            v.astype(np.float)
            c = np.sum(v, axis=1)

            overall_predictions_storer[acount,:] = c

            ##

        mean_of_all_predictions = np.mean(overall_predictions_storer,axis=0)
        #plt.plot(mean_of_all_predictions/Test_size.astype(np.float))

        #Graph_colour = "orchid"

        #plt.bar(np.arange(len(mean_of_all_predictions)),mean_of_all_predictions/Test_size,align="center",color = Graph_colour, edgecolor = "none")

        #plt.xlabel('Number of Features Included in Kernel')
        #plt.ylabel('Number of Correct Predictions')
        #plt.title('Accuracy vs Number of Features')

        #plt.show()

        ##

        best_predicators = np.argsort(mean_of_all_predictions)
        predicator_accuracies = np.sort(mean_of_all_predictions)
        best_predicators_args = args[Combinations[best_predicators]].astype(np.int)
        best_predicators_Gene_names = Gene_names[best_predicators_args]
        print(best_predicators_Gene_names, predicator_accuracies)

        np.save(Experiment + '_' + ML_Method +'_Best_Predicating_Gene_names',best_predicators_Gene_names)
        np.save(Experiment + '_' + ML_Method +'_Best_Predicating_Accuracies',predicator_accuracies)


#
