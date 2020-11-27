# Neural-Network

For this project, I used Bank Churn data. After reading the data and investigating, I found this data has no null values and shape is 14 variables with 10000 instances. Except
Gender, Geography and surnames all variables are numeric. Although variables are numeric but further investigation and understanding explains there are many categorical 
variables. Before going further, I categorized credit score as below and made a new variable 'Credit_ScoreGroup'. For understanding the multicollinearity between variables, 
I used heatmap for numeric variables and found there is no multicollinearity.

I divided the independent and dependent variables in X and y. After this using columntransfer I only used  OneHotEncoder for Geography variable. As other
categorical variables are either just have 0 and 1 values or the variables are ordinal. Here I used neural networking for the classification purpose. I used below libraries. 
Sequential groups a linear stack of layers into a model. In the first model, I used two hidden layers and ReLU activation function whereas for output layer I used sigmoid as I 
am doing binomial classification. For optimizer, I used adam as adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order 
and second-order moments. The cross-entropy computes the cross-entropy loss between true labels and predicted labels. The batch size is a number of samples processed before the
model is updated. The number of epochs is the number of complete passes through the training dataset. The size of a batch must be more than or equal to one and less than or 
equal to the number of samples in the training dataset. Although model converged very quickly still I started with 100 epochs and in different models reduced it further. 
After changing hyperparameters(including different activation function and optimization algorithm), different models were made. Although accuracy is almost same in all the 
models. From the model 2, I compared the train and test accuracy on different epochs. Although train and test has almost 0.86 accuracy still at epoch 10, test showing a
peak in accuracy. Moreover I plotted loss and compared for train and test; test has more loss than train but it is very very small. Confusion Matrix is also
given and specificity, Precision, Recall(Sensitivity), accuracy these values were also calculated.
Lift is a measure of the effectiveness of a predictive model calculated as the ratio between the results obtained with and without the predictive model. The greater the area
between the lift curve and the baseline, the better the model. The lift chart shows how much more likely we are to receive respondents than if we contact a random sample
of customers. By contacting only 10% of customers based on the predictive model we will reach 5 times as many respondents as if we use no model.
I compared the neural network model with the other classification models. For comparison of the neural network model, I used the Logistic regression, Naive Bayes and KNN. 
