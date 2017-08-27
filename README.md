# KNN for survival analysis in high-dimensional genomic settings. 


## @TODO:
	
	- NCA Survival (Neighborhood Component Analysis):
		- NCA for survival analysis -> cast as classification problem
		- NCA for feature selection and clustering by survival

	--------------------------------------------------------------
	- Port to python.
	- Bagging -> combined prediction and feature ranking.
	- Experiments with cumulative vs vs non-cumulative probability.

## 0- Is it novel?
   tl;dr In this context, YES!
   - KNN has been used in genomics for a long time, but not in right-censored settings.
   - This is corroborated by the relatively recent paper from Gad Getz that only used linear models and random survival forests.

   - Most important publications:
	-- https://rdrr.io/cran/bnnSurvival/ 
        -- Lowsky 2013: "A K-nearest neighbors survival probability prediction method"
   - Differences:
     - Lowski's method relies on creating KM curves, i.e. it relies on cumulative survival probability. -- our method uses non-cumulative survival probability which (at least in the GBMLGG dataset) results in a higher c-index.
     - Lowski's method was developed/tested using moderate-dimensional clinical data (300+ features), but has not been tested using genomic data or very high dimensional settings.
     - While the R package of Lowski's method uses bagged KNN to improve prediction accuracy and to de-correlate features (using different sets of features to find the nearest neighbors and averaging predictions), it still fails to use this as a method of feature ranking -- our method does not use bagging for the main model (but can be easily extended to do so), but uses it as a means to rank features. 
     - The impact of missing data on the method has not been investigated before. 
     - The effect of feature correlations (and the benefit gained from the moderate correlation seen in gene expression data) has not been investigated before.

2- Lowski's method was developed/tested using moderate-dimensional clinical data (300+ features), but has not been tested using genomic data or very high dimensional settings.

3- While the R package of Lowski's method uses bagged KNN to improve prediction accuracy and to de-correlate features (using different sets of features to find the nearest neighbors and averaging predictions), it still fails to use this as a method of feature ranking -- our method does not use bagging for the main model (but can be easily extended to do so), but uses it as a means to rank features.

4- The impact of missing data on the method has not been investigated before.
5- The effect of feature correlations (and the benefit gained from the moderate correlation seen in gene expression data) has not been investigated before.

## 1- Does it work:
   a- in different cancer types
   b- in different feature types
   
## 2- How does it compare to existing methods:
   a- Regularized cox regression
   b- Random survival forests
   c- Deep survival models

## 3- Can it rank features?
   - Univariate -> use trained model to predict survival of censored cases then use simple correlation between each feature and survival. Essentially, the model is used to "impute" censored cases to enable cimple correlations to be made.
   - Multivariate -> a random forest-like technique. This is what I used in the project to rank features. Essentially, random subsets of features are used to calculate the distance between patient pairs, such that each feature appears randomly in a number of different models, each with its own calculated accuracy. Features are then ranked by the median accuracy of models in which they were used. In other words, how much does feature Xi contribute to improving the prediction accuracy (c-index) of models in which it appears.

## 4- Cumulative vs. non-cumulative survival probability.
   - Which one is better? (confirm in different cancer types)
   - What factors affect this? Possible candidate: number of censored cases.

## 5- Handling of missing values: 
   - Need to do comparison with imputation

## 6- Calculating survival based on non-cumulative (vs cumulative) probability:
   - Need to try on different datasets and to investigate impact of censorship on this.
