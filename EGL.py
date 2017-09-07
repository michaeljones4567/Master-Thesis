import numpy as np
import code
import sklearn
import math
import scipy
#import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.linear_model import RandomizedLasso

from sklearn.linear_model import LogisticRegression

#====================================================================================================================================================================================#
# Load Data

Data_cleaned = np.load('data.npy')

X = np.transpose(Data_cleaned) # 116, 13718
X = X.astype(np.float)

#====================================================================================================================================================================================#
# Parameters

# Data Parameters
Number_of_data_points_TB = 46
Number_of_data_points_Clear = 70

Total_data_points = Number_of_data_points_TB + Number_of_data_points_Clear

Number_of_features = np.shape(Data_cleaned)[0]

#   Subset of data point of Parameters
Batch_size = 30

L1_Regulariser = 0.05
L2_regulariser = 0.1

#====================================================================================================================================================================================#
# Functions

def L1(X,Y,reg):

    clf_l1_LR = LogisticRegression(C=reg, penalty='l1', tol=0.01)
    clf_l1_LR.fit(X,Y)
    coefs = np.reshape(clf_l1_LR.coef_,[-1])

    return coefs

def CostFunc(intial_weights,y_hat_matrix,Y,reg):

    intial_weights = np.reshape(intial_weights,[-1,1])
    intial_weights = np.transpose(intial_weights)
    y_hat_matrix = np.transpose(y_hat_matrix)

    error = np.transpose(Y) - np.sign(np.sum(np.multiply(intial_weights,y_hat_matrix),axis=1))

    Cost = np.sum(np.square(error)) + np.multiply(reg,np.sqrt(np.sum(np.square(intial_weights))))

    print(Cost)
    ##

    return Cost

def F_builder(weights,group_indicator,group_indicator_2,remainder):

    remainder = int(remainder)



    c = np.reshape(group_indicator[0:(len(weights)-remainder)],[-1,Batch_size])
    d = weights[c]
    a = np.sum(np.abs(d),1)
    group_divide = np.divide(np.reshape(a,[-1,1]),np.abs(d))


    q = group_indicator[-remainder:]
    y = weights[q]
    z = np.sum(np.abs(y))
    group_divide_remainder = np.divide(np.reshape(z,[-1,1]),np.abs(y))



    group_divide_flattened = np.reshape(group_divide,[-1])
    group_divide_remainder_flattened = np.reshape(group_divide_remainder,[-1])
    group_divides_together = np.concatenate([group_divide_flattened,group_divide_remainder_flattened],0)

    r = np.concatenate([np.reshape(c,[-1]),np.reshape(q,[-1])],0)
    w = np.argsort(r)

    group_divides_together = group_divides_together[w]

    F_1 = np.diag(group_divides_together)

    return F_1

def Sigmoid_1(weights,X):
    ##
    weights = np.reshape(weights,[-1])
    z = np.reshape(np.matmul(X,weights),[-1,1])

    return 1/(1 + np.exp(-z))

def Sigmoid_2(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
    return G_of_Z

#====================================================================================================================================================================================#
# Load training data




# Extract data dimensions
Number_of_data_points = np.shape(X)[0]
Number_of_features = np.shape(X)[1]

Y1 = np.ones([Number_of_data_points_TB,1])
Y2 = np.zeros([Number_of_data_points_Clear,1])
Y = np.concatenate((Y1, Y2), axis=0)



#   Subset of features Parameters
L2_regulariser_start = 0.5
L2_regulariser_finish = 0.51
L2_regulariser_interval = 0.5

#

# Number of runs

Number_of_runs = 5

Best_cost = 100000000

number_of_loops_to_check_to_save = 1000

for icount in np.arange(L2_regulariser_start,L2_regulariser_finish,L2_regulariser_interval):

    for jcount in np.arange(0,Number_of_runs):

        End = 0

        theta = np.load('decentL1L2_theta_L2_Fever_0.5run_0.npy')

        m = len(Y)

        group_indicator = np.random.choice(Number_of_features, Number_of_features, replace=False)
        group_indicator_2 = np.reshape(group_indicator[0:Number_of_features-np.remainder(Number_of_features,Batch_size)],[-1,Batch_size])
        Difference = Batch_size - np.remainder(Number_of_features,float(Batch_size))
        Difference_zeros = np.ones(int(Difference))*(Number_of_features)

        if Difference != Batch_size:

            coefs = np.concatenate([group_indicator[-np.remainder(Number_of_features,Batch_size):],Difference_zeros],0)


            group_indicator_2 = np.concatenate([group_indicator_2,np.reshape(coefs,[1,-1])],0)

        counter = 0

        learning_rate = 0.000001

        number_of_loops = 70000

        while End == 0:

            F = F_builder(theta,group_indicator,group_indicator_2,np.remainder(Number_of_features,float(Batch_size)))

            a = Sigmoid_1(theta,X)
            b = a - Y
            temp_derivatives = np.sum(np.multiply(b,X),0) #*(alpha/m)

            first_term = temp_derivatives + np.multiply(icount,np.matmul(F,theta))

            theta = theta - np.multiply(learning_rate,first_term)

            counter += 1

            if counter % number_of_loops == 0:

                Best_Cost = Cost
                Best_w = theta
                number_of_loops = int(input('Enter number of loops to next step: '))
                learning_rate = float(input('Learning rate (suggested = 0.0001): '))
                End = int(input('Enter 1 to end or 0 to continue: '))

                Save = input('Save, yes or no? : ')

                if Save == 'yes':
                    print('saving')
                    np.save('decentL1L2_theta_L2_Fever_'+str(icount)+'run_'+str(jcount),theta)
                    print('saved')

                #

            error = Y - Sigmoid_1(theta,X)
            Cost = np.sum(np.square(error)) + np.multiply(L2_regulariser,np.sqrt(np.sum(np.square(theta))))
            print(counter)
            print(Cost)

            if Best_cost < Cost and counter % number_of_loops_to_check_to_save:

                Best_cost = Cost
                np.save('decentL1L2_theta_L2_Fever_'+str(icount)+'run_'+str(jcount),theta)


print('FINAL')
#
