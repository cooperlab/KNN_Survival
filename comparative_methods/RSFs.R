#!/usr/bin/Rscript

library(randomForestSRC, quietly=TRUE)
library(prodlim, quietly=TRUE)
#library(pec, quietly=TRUE)
library(survival, quietly=TRUE)
library(R.matlab, quietly=TRUE)

# paths
basePath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#basePath = "/home/mtageld/Desktop/KNN_Survival/"

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
    
    # Create output dirs
    result_dir = paste(basePath, 'Results/9_8Oct2017/', site, '_', dtype, '/', sep="")
    dir.create(result_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
    
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
      
      # initialize predictions
      n_hyperparam_sets = length(treeArray) * length(nodeSizeArray) * length(nSplitArray)
      preds_val = matrix(nrow=length(validIdxs), ncol=n_hyperparam_sets)
      preds_test = matrix(nrow=length(testIdxs), ncol=n_hyperparam_sets)
      
      # Go through hyperparams and get Ci
      
      cat("\nfold | trees | nodes | Split |\n")
      cat("--------------------------------------------\n")
      
      hyperp_idx = 1
      
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
            preds_val[1:length(validIdxs),hyperp_idx] = 
              predict.rfsrc(train.obj, allInfo[validIdxs,])$predicted
            
            # test
            preds_test[1:length(testIdxs),hyperp_idx] = 
              predict.rfsrc(train.obj, allInfo[testIdxs,])$predicted
            
            # clear space and increment      
            rm(train.obj)
            hyperp_idx = hyperp_idx + 1
      
          }
        }
      }
      
      # save result
      preds_val = data.frame(preds_val)
      preds_test = data.frame(preds_test)
      resultFile_val = paste(result_dir, 'fold_', fold, '_preds_val.txt', sep="")
      resultFile_test = paste(result_dir, 'fold_', fold, '_preds_test.txt', sep="")
      write.table(preds_val, file=resultFile_val)
      write.table(preds_test, file=resultFile_test)
      
      # clear space
      rm(preds_val)
      rm(preds_test)
      
    }
  }
}

