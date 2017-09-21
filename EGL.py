import numpy as np
import code

# Title - Exclusive Group LASSO
# Author - Michael Jones, UCL

# Machine Learning Technique Used
#----------------------------------
#       Exclusive Group LASSO, based on c.f. Kong et al 2014 - 'Exclusive Feature Learning on Arbitrary Structures via `1,2-norm.' Groups are made randomly instead of chosen.
#
#       Splits features into random groups, then applies LASSO regression across the groups and ridge regression over the entire feature set. See Kong et al 2014 for more details.
#
# Output:
#----------------------------------
#       A weighting for each feature in the group. The larger the weighting, the more important for classification the feature is deemed.
#
#----------------------------------
# List of important variables:
#       theta - feature weights
#       F - matrix that encodes group information. Please see Kong et al. 2014 for more detail.

#====================================================================================================================================================================================#
# Load Data

# Rows of data needs to be data points and columns features

Data_cleaned = np.load('TB_Cured_Data.npy')

# Shape and change data into appropiate type
X = np.transpose(Data_cleaned)
X = X.astype(np.float)

#====================================================================================================================================================================================#
# Parameters

# Set number of runs to repeat the experiment. As the groups are formed randomly, repeat runs are advised.
Number_of_runs = 5

# Set learning for gradient descent optimisation
learning_rate = 0.000001

# Set number of loops of optimisation after which to check results
Number_of_loops_to_check_to_save = 1000

# Data Classification parameters. Tell algorithm how many points in the data are TB patients and how many are not.
Number_of_data_points_TB = 46
Number_of_data_points_non_TB = 31

# Select group sizes over which to run exclusive group LASSO
Batch_size = 30

L2_regulariser = 0.1

#====================================================================================================================================================================================#
# Functions

def F_builder(weights,group_indicator,group_indicator_reshaped,remainder):

    remainder = int(remainder)

    reshaped_group_indicator = np.reshape(group_indicator[0:(len(weights)-remainder)],[-1,Batch_size])
    weights_shaped = weights[reshaped_group_indicator]
    weights_shaped = np.squeeze(weights_shaped)

    a = np.sum(np.abs(weights_shaped),1)
    code.interact(local=dict(globals(), **locals()))
    group_divide = np.divide(np.reshape(a,[-1,1]),np.abs(weights_shaped))

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

#====================================================================================================================================================================================#
# Extract data dimensions
Total_data_points = np.shape(X)[0]
Number_of_features = np.shape(X)[1]

#====================================================================================================================================================================================#
# Set up 1 and 0 labelling for classification of data

# Label TB infected patients as 1
Y1 = np.ones([Number_of_data_points_TB,1])

# Label non TB infected patients as 0
Y2 = np.zeros([Number_of_data_points_non_TB,1])

# Concatentate labels into overall data label vector
Y = np.concatenate((Y1, Y2), axis=0)

# Initialise best cost storer
Best_cost = np.inf


for icount in np.arange(0,Number_of_runs):

    # Set up marker that allows the user to end the program if wished at a later stage.
    End = 0

    # Randomly initialise weights for each feature
    theta = np.random.randn(Number_of_features,1)

    #====================================================================================================================================================================================#
    ## Indicators for what groups features are in
    # Firstly the features are placed into random groups
    group_indicator = np.random.choice(Number_of_features, Number_of_features, replace=False)

    # Secondly the indicator is re-shaped into a matrix where each column is a group
    group_indicator_reshaped = np.reshape(group_indicator[0:Number_of_features-np.remainder(Number_of_features,Batch_size)],[-1,Batch_size])

    #====================================================================================================================================================================================#
    ## Incase there are not a perfect amount features to fit into each group, this part of the code deals with this by forming the last group with the remainder of features.
    # Calculate difference between feature and the number of features
    Difference = Batch_size - np.remainder(Number_of_features,float(Batch_size))
    Difference_zeros = np.ones(int(Difference))*(Number_of_features)

    # Check if whether there is a remainder or not
    if Difference != Batch_size:

        # Generate weights for the remainder of features
        remainder_weights = np.concatenate([group_indicator[-np.remainder(Number_of_features,Batch_size):],Difference_zeros],0)

        # Concatenate remainder weights onto reshaped group indicator matrix
        group_indicator_reshaped = np.concatenate([group_indicator_reshaped,np.reshape(remainder_weights,[1,-1])],0)

    # Initialise counter for optimisation loops
    counter = 0

    # Optimisation while loop
    while End == 0:

        # Build F matrix which encodes the group information. See Kong et al 2014 for more information on F.
        F = F_builder(theta,group_indicator,group_indicator_reshaped,np.remainder(Number_of_features,float(Batch_size)))

        # Calculate sigmoid function for logistic regression classification
        Sigmoid_predictions = Sigmoid_1(theta,X)
        Difference_predictions_truth = Sigmoid_predictions - Y

        # Calculate derivatives of error part of cost function
        error_derivatives = np.sum(np.multiply(Difference_predictions_truth,X),0)

        # Note, this is not a pure cost function derivative. See Kong et al 2014 for more detail.
        total_optimisation_derivative = error_derivatives + np.multiply(L2_regulariser,np.matmul(F,theta))

        # Update feature weights
        theta = theta - np.multiply(learning_rate,total_optimisation_derivative)

        # Update counter
        counter += 1

        # Calculate error
        error = Y - Sigmoid_1(theta,X)

        # Calculate Cost
        Cost = np.sum(np.square(error)) + np.multiply(L2_regulariser,np.sqrt(np.sum(np.square(theta))))
        print(counter)
        print(Cost)

        # Save feature weights
        if Best_cost < Cost and counter % Number_of_loops_to_check_to_save:

            Best_cost = Cost
            np.save('decentL1L2_theta_L2_Fever_'+str(L2_regulariser)+'run_'+str(icount),theta)

        # Pause optimisation to allow user to check and analyse current results
        if counter % number_of_loops == 0:

            # Check if cost has improved
            if Cost < Best_Cost:
                # Update best cost storer
                Best_Cost = Cost

                # Update best set of feature weights
                Best_theta = theta

            # Allow user to interact with program
            print('Interactive shell. Press ctrl+D to exit shell.')
            code.interact(local=dict(globals(), **locals()))

            # Enter number of loops until next pause in program
            number_of_loops = int(input('Enter number of loops to next step: '))

            # Update learning rate
            learning_rate = float(input('Learning rate (suggested = 0.0001): '))

            # User decide whether to end program or continue
            End = int(input('Enter 1 to end or 0 to continue: '))

            # Save feature weights
            if Save == 'yes':
                print('saving')
                np.save('decentL1L2_theta_L2_Fever_'+str(L2_regulariser)+'run_'+str(icount),theta)
                print('saved')
