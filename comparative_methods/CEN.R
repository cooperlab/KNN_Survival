#!/usr/bin/Rscript

library(R.matlab, quietly=TRUE)
library(Coxnet, quietly=TRUE)

# paths
basePath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
#basePath = "/home/mtageld/Desktop/KNN_Survival/"

# data description
sites = c("GBMLGG", "KIPAN")
dtypes = c("Integ", "Gene")

# Hyperparameters to try
alphasArray = c(1, 0.7, 0.5, 0.3, 0.1, 0)
n_lambda  = 20

# Go through different datasets
for (dtype in dtypes) {
  for (site in sites) {

    cat("\n============================================\n")
    cat(site, dtype, "\n")
    cat("============================================\n")
    
    # Create output dirs
    result_dir = paste(basePath, 'Results/13_22Oct2017/', site, '_', dtype, '/', sep="")
    dir.create(result_dir, showWarnings = TRUE, recursive = FALSE, mode = "0777")
    
    # Reading the matfiles
    dpath = paste(basePath, 'Data/SingleCancerDatasets/', site, '/', 
                  site, '_', dtype, '_Preprocessed', sep="")
    Data = readMat(paste(dpath, '.mat', sep=""))
    splitIdxs = readMat(paste(dpath, '_splitIdxs.mat', sep=""))
    
    #
    # Getting the data in needed format
    #
    
    if (dtype == "Integ"){
      X = data.matrix(Data$Integ.X)
    } else {
      #options("expression" = 500000)
      X = data.matrix(Data$Gene.X)
    }
    
    Y = cbind(time=Data$Survival[1,], status=1-Data$Censored[1,])
    
    #
    # Go through folds
    #
    
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
      n_hyperparam_sets = length(alphasArray) * n_lambda
      preds_val = matrix(nrow=length(validIdxs), ncol=n_hyperparam_sets)
      preds_test = matrix(nrow=length(testIdxs), ncol=n_hyperparam_sets)
      
      # Go through hyperparams and get predictions
      cat("\nfold | alpha | lambda\n")
      cat("--------------------------------------------\n")
      
      hyperp_idx = 1
      
      #
      # Go through various alphas
      #
      
      for (alphaidx in 1:length(alphasArray)){
        
        # train model
        trained_model = Coxnet(X[trainIdxs,], Y[trainIdxs,], penalty= c("Lasso", "Enet", "Net"), 
                               alpha = alphasArray[alphaidx], nlambda = n_lambda)
        
        #
        # Predict val/test sets for various lambdas
        #
        
        for (lambdaidx in 1:length(trained_model$Beta[1, ])) {
          
          cat(fold, " | ", alphasArray[alphaidx], " | ", trained_model$fit$lambda[lambdaidx], "|\n") 
          
          beta = trained_model$Beta[, lambdaidx]
          beta[is.nan(beta)] = 0
          
          # predict validation
          preds_val[1:length(validIdxs), hyperp_idx] = X[validIdxs,] %*% beta
          
          # predict testing
          preds_test[1:length(validIdxs), hyperp_idx] = X[testIdxs,] %*% beta
          
          # increment
          hyperp_idx = hyperp_idx + 1
        }
        
        # clear space and increment      
        rm(trained_model)
        
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
    rm(Data)
    rm(X)
    rm(Y)
  }
}