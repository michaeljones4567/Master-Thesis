
# 56x45x25cm - Baggage allowance size on easy jet

# Title - Survival Count on Random Subsmaples
# Author - Michael Jones, UCL

# Machine Learning Technique Used
#----------------------------------
#       SCoRS (survival count on random subsamples), c.f. Rondina et al. 2014 - 'SCoRS - A Method Based on Stability for Feature Selection and Apping in Neuroimaging'
#
#       Uses sub-sampling of data points as well as sub-sampling of features combined with Lasso to select features. This is applied many times and stable features are counted.

# Hyperparameter selection
#----------------------------------
#       Rondina et al. suggest using 10% of the data points and 0.5% of the features in each sub-sample loop.

# Output:
#----------------------------------
#       A set of stable features for the data set provided.



import numpy as np
import code
import sklearn

from sklearn.linear_model import Lasso
from sklearn.linear_model import RandomizedLasso

from sklearn.linear_model import LogisticRegression

#====================================================================================================================================================================================#
# Parameters

# Parameters for SCoRS
Number_of_loops = 100000

#   Subset of features Parameters
Start_fraction_of_features = 0.0025
Final_fraction_of_features = 0.011
Subset_fraction_of_features = 0.0025

#   Subset of data point of Parameters
Start_size_Subset_example_size = 5
Final_size_Subset_example_size = 31
Subset_example_size_intervals = 5

# Data Parameters
Number_of_data_points_TB = 46
Number_of_data_points_Clear = 31
Number_of_data_points_Fever = 70

Number_of_data_points_Clear = Number_of_data_points_Clear + Number_of_data_points_Fever

# LASSO regulariser range to test
Start_lambda = 0.1
Final_lambda = 1.40
Lambda_Intervals = 0.25


#====================================================================================================================================================================================#
# Load training data

Data_cleaned = np.load('data.npy')#,dtype=None) # 13720, 117
Data_cleaned_Fever = np.load('data_1.npy')
Data_cleaned_Fever = np.transpose(Data_cleaned_Fever)



X = np.transpose(Data_cleaned) # 116, 13718

# Extract data dimensions
Number_of_data_points = np.shape(X)[0]
Number_of_features = np.shape(X)[1]

# Split into infected patient data and uninfected patient data

X_TB = X[0:Number_of_data_points_TB,:]
X_Clear = X[Number_of_data_points_TB:,:]
X_Fever = Data_cleaned_Fever[Number_of_data_points_TB:,:]

X = np.concatenate([X_TB, X_Clear, X_Fever], axis=0)

X_Clear = np.concatenate([X_Clear, X_Fever], axis=0)
X = X.astype(np.float)

X_Clear = X_Clear.astype(np.float)

# Label infected patient data and uninfected patient data


# Loop through data each time applying Lasso to a sub-sample of data and features


for zcount in np.arange(Start_fraction_of_features,Final_fraction_of_features,Subset_fraction_of_features):

    Subset_features_size = np.round(Number_of_features*zcount).astype(int)

    for mcount in np.arange(Start_size_Subset_example_size,Final_size_Subset_example_size,Subset_example_size_intervals):

        Y1 = np.ones([mcount,1])
        Y2 = np.zeros([mcount,1])
        Data_Sub_Sample_Y = np.concatenate((Y1, Y2), axis=0)

        for jcount in np.arange(Start_lambda,Final_lambda,Lambda_Intervals):

            # Set up counter for number of times features are selected

            Features_chosen_counter = np.zeros([1,Number_of_features])


            Features_chosen_counter_selected = np.zeros([1,Number_of_features])
            Features_chosen_counter_selected = np.reshape(Features_chosen_counter_selected,[-1,1])

            filename = 'FULL_Stability_Results_Fraction_2_' + str(zcount) + '_Subset_size_' + str(mcount) + '_Lambda_' + str(jcount)

            Storer_to_use_counter = 1

            for icount in range(0,Number_of_loops):

                print(icount)
                # Ramdomly select sub-sample of data points
                Chosen_example_samples_TB = np.random.random_integers(0,Number_of_data_points_TB-1,[mcount])
                Chosen_example_samples_Clear = np.random.random_integers(0,Number_of_data_points_Clear-1,[mcount])

                # Ramdomly select sub-sample of features

                Chosen_example_features = np.random.random_integers(0,np.round(Number_of_features)-1,[Subset_features_size])

                Features_chosen_counter_selected[Chosen_example_features] = Features_chosen_counter_selected[Chosen_example_features] + 1

                # Extract sub-samples
                Data_Sub_Sample_X_TB = X_TB[Chosen_example_samples_TB]
                Data_Sub_Sample_X_TB = Data_Sub_Sample_X_TB[:,Chosen_example_features]

                Data_Sub_Sample_X_Clear = X_Clear[Chosen_example_samples_Clear]
                Data_Sub_Sample_X_Clear = Data_Sub_Sample_X_Clear[:,Chosen_example_features]

                Data_Sub_Sample_X = np.concatenate((Data_Sub_Sample_X_TB, Data_Sub_Sample_X_Clear), axis=0)

                # Apply Lasso to sub-sample

                clf_l1_LR = LogisticRegression(C=jcount, penalty='l1', tol=0.01)

                clf_l1_LR.fit(Data_Sub_Sample_X,Data_Sub_Sample_Y)

                # Count non-features
                coefs = np.reshape(clf_l1_LR.coef_,[-1])
                Chosen_features_indexes = np.where(np.abs(coefs)>0)

                if np.sum(Chosen_features_indexes) != 0:

                    Chosen_features = Chosen_example_features[Chosen_features_indexes]
                    Features_chosen_counter[:,Chosen_features] = Features_chosen_counter[:,Chosen_features] + 1

            Features_chosen_counter_selected = np.transpose(Features_chosen_counter_selected)
            Features_chosen_counter  = Features_chosen_counter/Features_chosen_counter_selected
            Features_chosen_counter = np.transpose(Features_chosen_counter)

            np.save(filename,Features_chosen_counter)

# Output most significant feature and other significant features

Features_chosen_counter_selected = np.transpose(Features_chosen_counter_selected)

Features_chosen_counter  = Features_chosen_counter/Features_chosen_counter_selected

print(np.argmax(Features_chosen_counter))
print(np.where(Features_chosen_counter>(0.84)))

# Output names of most significant genes
Significant_features_indexes = np.where(Features_chosen_counter>(0.84))[1]

code.interact(local=dict(globals(), **locals()))
