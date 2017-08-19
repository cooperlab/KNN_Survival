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
   Since we are dealing with right-censored data, simply correlating features with survival does not work due to missing data. Once we got an accurate model, "impute" the survival of right-censored cases using your method. Now we've converted right-censored data into complete prediction data! Now we can simply use the Pearson or Spearman correlation coefficients to rank the features.

## 4- Cumulative vs. non-cumulative survival probability.
   - Which one is better? (confir in different cancer types)
   - What factor affect this? Possible candidate: number of censored cases.

## 5- Imputation using KNN vs relying directly on KNN survival prediction - handling missing values.
