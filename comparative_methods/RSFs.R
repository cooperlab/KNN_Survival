#!/usr/bin/Rscript

library(randomForestSRC, quietly=TRUE)
library(prodlim, quietly=TRUE)
#library(pec, quietly=TRUE)
library(survival, quietly=TRUE)
library(R.matlab, quietly=TRUE)

# paths
#basePath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
basePath = "/home/mtageld/Desktop/KNN_Survival/"

# data description
sites = c("GBMLGG", "BRCA", "KIPAN")
dtypes = c("Integ", "Gene")

# Hyperparameters to try
treeArray = c(50, 100, 500, 1000)
nodeSizeArray = c(1, 3, 5, 7, 9)
nSplitArray = c(0, 1)

# Go through different datasets
for (dtype in dtypes) {
  for (site in sites) {
    
    cat("\n============================================\n")
    cat(site, dtype, "\n")
    cat("============================================\n")
    
    # Reading the matfiles
    dpath = paste(basePath, 'Data/SingleCancerDatasets/', site, '/', 
                  site, '_', dtype, '_Preprocessed', sep="")
    Data = readMat(paste(dpath, '.mat', sep=""))
    splitIdxs = readMat(paste(dpath, '_splitIdxs.mat', sep=""))
    
    # Getting the data in needed format
    if (dtype == "Integ"){
      allInfo = data.frame(allData = Data$Integ.X, time = Data$Survival[1,], 
                           status = 1-Data$Censored[1,])
    } else {
      #options("expression" = 500000)
      allInfo = data.frame(allData = Data$Gene.X, time = Data$Survival[1,], 
                           status = 1-Data$Censored[1,])
    }
    
    # initialize accuracy
    # Note that we can't initialize to a fized length
    # bacuse some folds are larger than others, so we'll
    # just init to max possible no of patients
    preds_val = matrix(nrow=length(Data$Survival), ncol=30)
    preds_test = matrix(nrow=length(Data$Survival), ncol=30)
    
    # Go through folds
    for (fold in 1:30){
    
      # Getting split indices
      # Note the +1 because R is one-indexed
      if (site == "KIPAN" && dtype == "Integ"){
        trainIdxs = splitIdxs$train[fold,] + 1
        validIdxs = splitIdxs$valid[fold,]  + 1
        testIdxs = splitIdxs$test[fold,]  + 1
      } else {
        trainIdxs = splitIdxs$train[fold][[1]][[1]][1,]  + 1
        validIdxs = splitIdxs$valid[fold,]  + 1
        testIdxs = splitIdxs$test[[fold]][[1]][1,]  + 1
      }
      
      # Go through hyperparams and get Ci
      
      cat("\nfold | trees | nodes | Split |\n")
      cat("--------------------------------------------\n")
      
      for (i in 1:length(treeArray)){
        for (j in 1:length(nodeSizeArray)){
          for (k in 1:length(nSplitArray)){
            
            cat(fold, " | ", treeArray[i], " | ", nodeSizeArray[j], " | ", 
                nSplitArray[k], "|\n") 
            
            # training the model
            train.obj <- rfsrc(Surv(time, status) ~ ., allInfo[trainIdxs,], 
                               nsplit = nSplitArray[k], nodesize = nodeSizeArray[j], 
                               ntree = treeArray[i])
            
            # predict validation set
            preds_val[1:length(validIdxs),fold] = 
              predict.rfsrc(train.obj, allInfo[validIdxs,])$predicted
            
            # test
            preds_test[1:length(testIdxs),fold] = 
              predict.rfsrc(train.obj, allInfo[testIdxs,])$predicted
            
            # clear space      
            rm(train.obj)
      
          }
        }
      }
    }
    
    # save result
    resultFile_val = paste(basePath, 'Results/9_8Oct2017/', 
                       site, '_', dtype, '_preds_val.txt', sep="")
    resultFile_test = paste(basePath, 'Results/9_8Oct2017/', 
                           site, '_', dtype, '_preds_test.txt', sep="")
    
    preds_val = data.frame(preds_val)
    preds_test = data.frame(preds_test)
    
    write.table(preds_val, file=resultFile_val)
    write.table(preds_test, file=resultFile_test)
    
    cat("\nPreds is stored to: \n", 
        resultFile_val, "\n", resultFile_test, "\n")

  }
}

