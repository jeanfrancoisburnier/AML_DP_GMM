<<<<<<< HEAD
﻿** GMM_reg **: A machine learning algorithm that use the Gaussian Mixture Model to do regression. The GMM-reg uses files of the machine learning toolbox ML_toolbox that you can find here: https://github.com/epfl-lasa/ML_toolbox/

To use the algorithm, chose a dataset included in your matlab's path. You can then use the training ratio, the hyperparameters are determined using the EM-step. By using the metrics provided, you can select the number of cluster before running the regression.

Author: Aurélie Balsa, Jean-François Burnier, Nicolas Winteler.


** GMR_comparison **: A machine learning algorithm that use the Gaussian Mixture Model to do regression. The GMR_comparison scripts uses files of the machine learning toolbox ML_toolbox (https://github.com/epfl-lasa/ML_toolbox/)and Frank Wood's DP-GMM toolbox (...)

The sections of the code operate the following operations:

1. 	Set the hyperparameters and variables to record statistics
2. 	Load a recorded dataset
3. 	Generate a dataset using the load_regression_dataset
4. 	Start a gridsearch on two parameters	
5. 	Repeat loop to get statistical values (mean, var, quantiles)
6.	Define the training and testing set
7. 	Perform DP-GMM, keeps the result with highest likelihood
8. 	Create the Gaussians using the classes from DP-GMM
9. 	Perform the Regression and calculate the metrics (MSE, NMSE, R squared)
10. Plot the result of the gridsearch
=======
** GMM_reg ** : A machine learning algorithm that uses the Gaussian Mixture Model to do regression. The GMM-reg uses files of the machine learning toolbox ML_toolbox that you can find here : https://github.com/epfl-lasa/ML_toolbox/

To use the algorithm, first chose a dataset that needs
To see how the algorithm is working, chose a dataset that needs to be in the path. Then you can use the training ratio and the hyperparameters are determined using the EM-step. From the graph you can select the number of cluster before running the regression.

Author: Aurélie Balsa, Jean-François Burnier, Nicolas Winteler.
>>>>>>> refs/remotes/origin/master
