# ~/usr/r/R-3.1.2/bin/Rscript --max-ppsize=400000
#

#shuffle = 1
fileDir = '/home/syouse3/git/survivalnet/data/'
cat(fileDir)

library(randomForestSRC)
library(prodlim)
library(pec)
library(survival)



runRSF = function(fileDir, shuffle) {

  type = "Gene"
  subtype = "OVBRCAUCEC"
  
  dataFile = paste(fileDir, subtype, '_', type, '/X.csv', sep = '')
  survFile = paste(fileDir, subtype, '_', type, '/T.csv', sep = '')
  censorFile = paste(fileDir, subtype, '_', type, '/O.csv', sep = '')
  
  resultValueFile = paste(fileDir, subtype, '_', type, '/shuffle', shuffle, '.max.txt', sep="")
#  resultAllFile = paste(fileDir, subtype, '_', type, '.shuffle', shuffle, '.all.txt', sep="")
  
  allData <- read.csv(dataFile, header = FALSE)
  survData = read.table(survFile, header = FALSE)
  survData = survData - min(survData) + 1
  censorData = read.table(censorFile, header = FALSE)
  
  n = length(survData$V1)
  
  val_c_indices = c()
  test_c_indices = c()
  
  allInfo = data.frame(allData = allData, y = survData, c = censorData)
  allInfo <- allInfo[sample(nrow(allInfo)),]
  
  test_i = floor(n * 0.2)
  validate_i = 2 * test_i
  
#  testData <- allInfo[(0:test_i),]
  
#  validateData <- allInfo[((test_i + 1):validate_i),]
  
#  trainData <- allInfo[((validate_i + 1):n),]

  testData <- allInfo[((n - test_i + 1):n),]
   
  cat(n-test_i+1, n-validate_i+1, n,"\n") 
  validateData <- allInfo[((n - validate_i + 1):(n - test_i)),]
  
  trainData <- allInfo[(0:(n - validate_i)),]
  
  treeArray = c(50, 100, 500,1000 )
  nodeSizeArray = c(1,3,5,7,9)
  nSplitArray = c(0,1)
  
  for (i in 1:length(treeArray)){
    for (j in 1:length(nodeSizeArray)){
      for (k in 1:length(nSplitArray)){
        
        cat(i, j, k,"\n") 
        # training
        train.obj <- rfsrc(Surv(V1, V1.1) ~ . , trainData, nsplit = nSplitArray[k], nodesize = nodeSizeArray[j], ntree = treeArray[i])
        # validate
        val_c_index <- cindex(train.obj, formula = Surv(V1,V1.1) ~ ., data=validateData)
        val_c_indices <- c(val_c_indices, val_c_index$AppCindex$rfsrc)
        
        #cat("validate c index: ", val_c_index$AppCindex$rfsrc)
        
        # test
        test_c_index  <- cindex(train.obj, formula = Surv(V1,V1.1) ~ ., data=testData)
        test_c_indices <- c(test_c_indices, test_c_index$AppCindex$rfsrc)
        
        #cat(", test c index: ", test_c_index$AppCindex$rfsrc, "\n")
        
        rm(train.obj)
        rm(val_c_index)
        rm(test_c_index)
      }
    }
  }
  
  resultData = data.frame(maxTest = test_c_indices[which(val_c_indices == max(val_c_indices))])
  
  resultAllData = data.frame(val_c_indices =  val_c_indices, test_c_indices = test_c_indices)
  
  write.table(resultData, file=resultValueFile)
  #write.table(resultAllData, file=resultAllFile)    
  
  cat("\nmax is stored to: ", resultValueFile, "\n")
  #cat("\nall data is stored to: ", resultAllFile, "\n")
}

for (i in 1:20){
  runRSF(fileDir = fileDir, shuffle = i)
}
