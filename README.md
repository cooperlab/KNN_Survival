# KNN as a means of survival outcome prediction and imputation in high-dimensional genomic settings. 

## 0- Is it novel?
   tl;dr In this context, YES!
   a- KNN has been used in genomics for a long time, but not in right-censored settings.
   b- This is corroborated by the relatively recent paper from Gad Getz that only used linear models and random survival forests.
   c- KNN has been used in relatively high-dimensional settings (300+ features) in kidney tumor cases using clinical data, but not genomic data. In that case, K-M curves were constructed using closes K-patients (cumulative survival probability) and no clear way of ranking features. 

## 1- Does it work:
   a- in different cancer types
   b- in different feature types
   
## 2- How does it compare to existing methods:
   a- Regularized cox regression
   b- Random survival forests
   c- Deep survival models

## 3- Can it rank features?
   a- Univariate - use trained model to predict survival of censored cases then use simple correlation between each feature and survival. Essentially, the model is used to "impute" censored cases to enable cimple correlations to be made.
   b- Multivariate - a random forest-like technique. This is what I used in the project to rank features. Essentially, random subsets of features are used to calculate the distance between patient pairs, such that each feature appears randomly in a number of different models, each with its own calculated accuracy. Features are then ranked by the median accuracy of models in which they were used. In other words, how much does feature Xi contribute to improving the prediction accuracy (c-index) of models in which it appears.

## 4- Cumulative vs. non-cumulative survival probability.
   - Which one is better? (confir in different cancer types)
   - What factor affect this? Possible candidate: number of censored cases.

## 5- Handling of missing values - 
    need to do comparison with imputation

## 6- Calculating survival based on non-cumulative (as opposed to cumulative) probability:
    need to try on different datasets and to investigate impact of censorship on this.
