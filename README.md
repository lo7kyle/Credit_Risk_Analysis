# Credit_Risk_Analysis
Supervised Machine Learning with Python and Scikit Learn

## Overview:
In this challenge we are given the task to use Scikit learn and Supervised Machine learning to determine which Machine Learning model gives the most accurate classification of High or Low credit card risk depending on certain features. 

## Purpose:
The purpose of this challenge is to introduce supervised machine learning and the different models for classification. There are two types of supervised ML such as regression to predict continuous data and classification in which what we are asked to perform. We are to take the different classification models and compare them with each other to determine which model produces the best accuracy. The data set given is inherently unbalanced due to the nature of credit card risk having more low risk than higher risk customers. Another way to look at this is asking, out of 1,000,000 patients in a hospital, how many of that 1,000,000 will have diabetes. The number should be low since not everyone has diabetes.            

## Resources
* Data Source: (LoanStats_2019Q1.csv)
* Software: 
\ Jupyter Notebook 6.3
\ Python 3.7


## Analysis:
### Overview of Analysis:
When running the 6 different models the results were as expected with a few surprising observations I made during the tests. This observation came from using scaled data vs unscaled data.

### Results:

 Starting with the Random Oversampling, the results showed that there is about a 20% difference in accuracy between scaled and unscaled data. This is mostly due to the weight of certain features having over others. When looking at the recall we see that we have a decent sensitivity level (scaled data) however it is still far from perfect. The precision is extremely low at detecting high credit risk resulting in a low f1 score; however, the precision for the low_risk is perfect. One reason for this is probably due to the imbalance of the data. A high recall in this case is a good thing because since credit risk impacts a bank financially, you wouldn't want to just classify just anything as high_risk. 

*SPLIT TRAIN SCALE*
![Split train scale](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/split_train_scale.PNG) 

![Naive Oversampling](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/Naive_Oversampling.PNG) 

![Naive Oversampling ACC](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/Naive_Oversampling_accuracy.PNG) 

![Naive Oversampling Class](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/Naive_Oversampling_classification.PNG) 

Looking at the SMOTE model we see that the accuracy hasn't changed much. The unscaled data increased by 2% but it is very marginal. The f1 score is also slightly higher but is very nominal. 

![SMOTE Oversampling](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTE_Oversampling.PNG) 

![SMOTE Oversampling ACC](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTE_Oversampling_acc.PNG) 

![SMOTE Oversampling Class](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTE_Oversampling_classification.PNG) 

Undersampling is the worst model out of the 6 used. Undersampling is only useful if you have an abundance of data, but in our case we probably didn't have enough. There might be a possibility that if we had at least 1000+ low_risk data, the undersampling might have resulted in a more accurate prediction. 

![Undersampling](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/Undersampling.PNG) 

![Undersampling ACC](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_acc.PNG) 

![Undersampling Class](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/Undersampling_classification.PNG) 

The last resampling method used was the SMOTEENN model. This model yields the highest accuracy for the scaled data in terms of resampling, but not so much for the unscaled data. This is probably due to the high variance and potential outliers. When I mention best, I am basing it off what I deemed best for this application. The f1 score isn't the best in comparison to all of the resampling models; however, the difference is nominal in my analysis. We are looking for the highest accuracy for recall because we want our model to be as sensitive as possible for our potential clients and for the financials of the company.   

![SMOTEENN Oversampling](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN.PNG) 

![SMOTEENN Oversampling ACC](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTE_Oversampling_acc.PNG) 

![SMOTEENN Oversampling Class](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN_classification.PNG) 

The next two model analysis are Ensemble models that use weak learners in combination to create an aggregated output. The first model we used was the BalancedRandomForest classifier. In this model we see that regardless of scaled or unscaled data, the accuracy is the same. When it comes to logistic models, scaling matters more and will benefit from scaling. In a random forest, each decision tree is operating in parallel with each other and are not affect by each other's features. Although the Random Forest is robust to outliers, the accuracy is still low. 

![Balanced RBF](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/brf.PNG) 

![Balanced RBF ACC](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/brf_acc.PNG) 

![Balanced RBF Class](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/brf_classification.PNG) 

The last model used is called Adaboost. This method is the most accurate model of all 6 models and yields the highest accuracy in terms of f1. Adaboosting is evaluating the error of each model and gives extra weight of the error of the previous model. From the classification report we see that the recall for both low/high risk are above 90.  

![Adaboost](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/adaboost.PNG) 

![Adaboost ACC](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/adaboost_acc.PNG) 

![Adaboost Class](https://github.com/lo7kyle/Credit_Risk_Analysis/blob/main/Resources/adaboost_classification.PNG) 

### Summary:
When not scaling the data, the accuracy is extremely low. It was around 65% for the random oversampling and the smote oversampling. The undersampling was around 54%. The SMOTEENN was around 62%. When you normalize the data, the accuracy shoots up to around 83% for oversampling and 81% of undersampling. The best results ended up with the combination of Over and Under Sampling however when you normalize the data. The SMOTEEN model gave the best results when considering all of the resampling methods. Referring back to the normalized data, the big discrepancy of accuracy is probably due to the Loan amount being a really big number. Although we were not asked to scale the data, if we don't scale the data, the Loan Amount feature which has a std of 10,277 has too much of a variation for our model to really predict accurately. When looking at the ensemble models we see that Adaboost gives the best results out of all 6 models.
