library(randomForestSRC, quietly=TRUE)
library(prodlim, quietly=TRUE)
library(pec, quietly=TRUE)
library(survival, quietly=TRUE)
library(R.matlab, quietly=TRUE)

# paths
#basePath = "/home/mohamed/Desktop/CooperLab_Research/KNN_Survival/"
basePath = "/home/mtageld/Desktop/KNN_Survival/"

# data description
sites = c("GBMLGG", "BRCA", "KIPAN", "MM")
dtypes = c("Integ", "Genes")

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
    allInfo = data.frame(allData = Data$Integ.X, time = Data$Survival[1,], 
                         status = 1-Data$Censored[1,])
    
    # define result file
    resultFile = paste(basePath, 'Results/9_8Oct2017/RSF_', 
                       site, '_', dtype, '.txt', sep="")
    
    # Go through folds
    for (fold in 1:30){
    
      # Getting split indices
      trainIdxs = splitIdxs$train[fold][[1]][[1]][1,]
      validIdxs = splitIdxs$valid[fold,]
      testIdxs = splitIdxs$test[[fold]][[1]][1,]
      
      # initialize accuracy
      val_c_indices = c()
      test_c_indices = c()
      
      # Go through hyperparams and get Ci
      
      cat("\nfold | trees nodes Split | Ci_val Ci_test\n")
      cat("--------------------------------------------\n")
      
      for (i in 1:length(treeArray)){
        for (j in 1:length(nodeSizeArray)){
          for (k in 1:length(nSplitArray)){
            
            # training the model
            train.obj <- rfsrc(Surv(time, status) ~ ., allInfo[trainIdxs,], 
                               nsplit = nSplitArray[k], nodesize = nodeSizeArray[j], 
                               ntree = treeArray[i])
            
            # validate
            val_c_index <- cindex(train.obj, formula = Surv(time, status) ~ ., 
                                  data=allInfo[validIdxs,])
            val_c_indices <- c(val_c_indices, val_c_index$AppCindex$rfsrc)
            
            # test
            test_c_index  <- cindex(train.obj, formula = Surv(time, status) ~ ., 
                                    data=allInfo[testIdxs,])
            test_c_indices <- c(test_c_indices, test_c_index$AppCindex$rfsrc)
            
            cat(fold, "|", treeArray[i], nodeSizeArray[j], nSplitArray[k], "|",
                val_c_index$AppCindex$rfsrc, test_c_index$AppCindex$rfsrc, "\n") 
            
            # clear space      
            rm(train.obj)
            rm(val_c_index)
            rm(test_c_index)
      
          }
        }
      }
    }
    
    # Find testing Ci corresponding to maximal validation Ci
    resultData = data.frame(maxTest = test_c_indices[
                            which(val_c_indices == max(val_c_indices))])
    # Save result
    write.table(resultData, file=resultFile)
    cat("\nCi_test is stored to: ", resultFile, "\n")

  }
}

