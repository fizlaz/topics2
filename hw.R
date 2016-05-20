setwd("~/bgse/15D013_Topics_2/Prof. A. Arratia")

library(quantmod); library(xts)
library(fBasics)

#getSymbols("AAPL",from="1997-01-01")
sym <- "GOOGL"

# Q1

acplot <- function(sym){
  
  dat <- getSymbols(sym, auto.assign = FALSE)
  df <- as.data.frame(dat)
  df$rtrn <- 0
  
  column <- paste(c(sym,".Adjusted"),collapse="")
  print(column)
  
  for(i in 2:nrow(df)){
    df[i,"rtrn"] <- abs(df[i,column]/df[(i-1),column]-1)
  }
  
  #teffectPlot(df$rtrn)
  
  df$rtrn2 <- df$rtrn^2
  df$rtrn3 <- df$rtrn^3
  df$rtrn4 <- df$rtrn^4
  df$rtrn5 <- df$rtrn^5
  
  
  ac <- acf(df$rtrn)
  ac2 <- acf(df$rtrn2)
  ac3 <- acf(df$rtrn3)
  ac4 <- acf(df$rtrn4)
  ac5 <- acf(df$rtrn5)
  
  plot(ac$acf,type="l")
  lines(ac2$acf, col="red")
  lines(ac3$acf, col="blue")
  lines(ac4$acf, col="yellow")
  lines(ac5$acf, col="orange")
  
  
}

acplot("GOOGL")

teffectPlot(df$rtrn)

# Q2

getSymbols("UMCSENT",src="FRED")

df <- as.data.frame(UMCSENT)
plot(df$UMCSENT,type="l")
ac <- acf(df$UMCSENT)
plot(ac$acf,type="l")
pac <- pacf(df$UMCSENT)
plot(pac$acf,type="l")

#unit roots test
library(tseries)
adf.test(df$UMCSENT) #non-stationary, cannot fit ARIMA

plot(diff(df$UMCSENT))
df_diff <- diff(df$UMCSENT)
acf(df_diff)
pacf(df_diff)
adf.test(df_diff)

library(forecast)
best_ar <- auto.arima(df_diff)
best_ar

df$rtrn <- 0

for(i in 2:nrow(df)){
  df[i,"rtrn"] <- (df[i,"UMCSENT"]/df[(i-1),"UMCSENT"]-1)
}

df$rtrn2 <- df$rtrn^2

acf(df$rtrn)
acf(df$rtrn2)
pacf(df$rtrn2)
Box.test(df$rtrn, type = "Ljung-Box")
Box.test(df$rtrn2, type = "Ljung-Box")
plot(df$rtrn, type="l")
plot(df$rtrn2, type="l")

library(fGarch)

ar <- arma(df$rtrn2)
ar
ga <- garch(df$rtrn2)
ga

arma_gar <- garchFit(formula = ~arma(1,1)+garch(1,1), data = df$rtrn)

# Q3

log_return <- function(vec){
  
  #v1
  rtrn <- rep(0,length(vec))
  
  for(i in 2:length(vec)){
    rtrn[i] <- vec[i]/vec[(i-1)]-1
  }
  
  #logrtrn <- 100*log(1+rtrn)
  logrtrn <- log(1+rtrn)
  
  #v2
  #   logrtrn <- rep(0,length(vec))
  #   for(i in 2:length(vec)){
  #     logrtrn[i] <- log(vec[i]) - log(vec[i-1])
  #   }
  #   logrtrn <- logrtrn*100
  
  return(logrtrn)
}


sp <- read.csv("data/data/SP500_shiller.csv")

divpr <- sp$Real.Dividend/sp$Real.Price


splogrtrn <- log_return(sp$SP500)

dta <- data.frame(log_return(sp$SP500),log_return(divpr))
dta1 <- data.frame(log_return(sp$SP500),log_return(sp$P.E10))
dta2 <- data.frame(log_return(sp$SP500),log_return(divpr),log_return(sp$P.E10))

dta1 <- dta1[complete.cases(dta1),]
dta2 <- dta2[complete.cases(dta2),]

set <- data.frame(c(1:10),c(11:20))

lagdata <- function(set,lag){
  x <- data.frame(set[,1])
  for(i in 1:lag){
    y <- set[-c(nrow(set):(nrow(set)-i+1)),-1]
    x <- data.frame(x[-1,],y)
  }
  return(x)
}

lagdata(set,3)

training <- lagdata(dta,5)
head(training)
names <- names(training)
names[1] <- "target"
names(training) <- names

training1 <- lagdata(dta1,5)
head(training1)
names <- names(training1)
names[1] <- "target"
names(training1) <- names

training2 <- lagdata(dta2,5)
head(training2)
names <- names(training2)
names[1] <- "target"
names(training2) <- names

########Nonlinear models#############################
####### SVM and Neural networks ############
library(e1071) ##for svm
library(nnet)
library(kernlab)
library(quantmod)
library(caret) ##for some data handling functions
library(Metrics)##Measures of prediction error:mse, mae
library(xts)


library(caret)
set.seed(1234)
trainIndex <- createDataPartition(training$target, p=0.80, list=FALSE)
data_train <- training[ trainIndex,]
data_test <- training[-trainIndex,]

##Train model
##############################################
##OPTION LAZY: one svm, one nnet built w/o tuning  (or tune by hand)
#type="C" ##classification
type="eps-regression" ##regression
#parameters that can be tuned
u= -2 ## -3,-2,-1,0,1,2,3
gam=10^{u}; 
w= 4.5 ##1.5,-1,0.5,2,3,4
cost=10^{w}

gam <- 0.01
cost <- 0.11
##The higher the cost produce less support vectors, increases accuracy
##However we may overfit
svmFit = svm (data_train[,-1], data_train[,1],
              #type=type, 
              kernel= "radial",
              gamma=gam,
              cost=cost
)
summary(svmFit)
##build predictor
predsvm = predict(svmFit, data_test[,-1])

mse(data_test[,1],predsvm)

#########TUNING

trainset <- data.frame(training)
trainset1 <- data.frame(training1)
trainset2 <- data.frame(training2)


# set up the cross-validated hyper-parameter search
svm_grid_1 = expand.grid(
  cost = 10^c(-1,0.5,1.5,2,3,4),
  gamma = 10^c(-3,-2,-1,0,1,2,3)
)

svm_grid_2 = expand.grid(
  cost = 0.11,
  gamma = 0.01
)

# pack the training control parameters
svm_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",               # save losses across all models
  #classProbs = TRUE,                  # set to TRUE for AUC to be computed
  #summaryFunction = twoClassSummary,
  summaryFunction = defaultSummary,
  allowParallel = TRUE
)

svm_train_0 = train(
  x = trainset[,-1],
  y = trainset[,1],
  trControl = svm_trcontrol_1,
  tuneGrid = svm_grid_2,
  method = "svmLinear2"
  #kernel = "radial", #radial is default
  #type="eps-regression"
)

svm_train_1 = train(
  x = trainset1[,-1],
  y = trainset1[,1],
  trControl = svm_trcontrol_1,
  tuneGrid = svm_grid_2,
  method = "svmLinear2"
  #kernel = "radial", #radial is default
  #type="eps-regression"
)

svm_train_2 = train(
  x = trainset2[,-1],
  y = trainset2[,1],
  trControl = svm_trcontrol_1,
  tuneGrid = svm_grid_2,
  method = "svmLinear2"
  #kernel = "radial", #radial is default
  #type="eps-regression"
)


svm_train_0
svm_train_1
svm_train_2

#########

##A nnet with size hidden layers +skip layer. Max iteration 10^4,
size=6
nnetFit = nnet(training[,-1], training[,1],
               size=size,skip=T, maxit=10^4,decay=10^{-2},trace=F,linout=T)
summary(nnetFit) ##gives description w/weights
##build predictor type="raw"
prednet<-predict(nnetFit,testing[,-ncol(testing)],type="raw")

################end of Option Lazy ##############################


cv.nn <- function(train, size=10, rounds=250){

  library(nnet)

  c <- which(colnames(train)=="target")

    library(caret)
    set.seed(1234)
    flds <- createFolds(train$target,k=5)
    nnpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]

      #print("fold")
      #print(j) # tracing progress
      #print(nnpred)

      size=6
      nnetFit = nnet(ktrain[,-1], ktrain[,1],
               size=size,skip=T, maxit=10^4,decay=10^{-2},trace=F,linout=T)

      #nnetFit = nnet(ktrain[,-1], ktrain[,1],
      #         size=size, linout=T)

      prednet<-predict(nnetFit,ktest[,-1],type="raw")


      #print(mse(ktest[,1],prednet))

      nnpred[j] <- mse(ktest[,1],prednet)

    } # end of looping cross validations
    avg <- mean(nnpred)

    print(nnpred)
    print(avg)
    return(list(folds=nnpred,average=avg))

} # end of cv.nn


cvnn <- cv.nn(training)
cv.nn(training1)
cv.nn(training2)

cv.svm <- function(train){

  library(nnet)

  c <- which(colnames(train)=="target")

    library(caret)
    set.seed(1234)
    flds <- createFolds(train$target,k=5)
    nnpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]

      #print("fold")
      #print(j) # tracing progress
      #print(nnpred)

      gam <- 0.01
      cost <- 0.11
      ##The higher the cost produce less support vectors, increases accuracy
      ##However we may overfit
      svmFit = svm (data_train[,-1], data_train[,1],
              #type=type, 
              kernel= "radial",
              gamma=gam,
              cost=cost
      )
      summary(svmFit)
      ##build predictor
      predsvm = predict(svmFit, data_test[,-1])

      mse(data_test[,1],predsvm)
      
      size=6
      nnetFit = nnet(ktrain[,-1], ktrain[,1],
               size=size,skip=T, maxit=10^4,decay=10^{-2},trace=F,linout=T)

      #nnetFit = nnet(ktrain[,-1], ktrain[,1],
      #         size=size, linout=T)

      prednet<-predict(nnetFit,ktest[,-1],type="raw")


      #print(mse(ktest[,1],prednet))

      nnpred[j] <- mse(ktest[,1],prednet)

    } # end of looping cross validations
    avg <- mean(nnpred)

    print(nnpred)
    print(avg)
    return(list(folds=nnpred,average=avg))

} # end of cv.nn

#############
library(caret)
set.seed(1234)
trainIndex <- createDataPartition(training$target, p=0.80, list=FALSE)
data_train <- training[ trainIndex,]
data_test <- training[-trainIndex,]
data_train <- training[ c(1:1400),]
data_test <- training[-c(1:1400),]

library(h2o)
localH2O = h2o.init(nthreads=-1)

data_train_h <- as.h2o(data_train,destination_frame = "h2o_data_train")
data_test_h <- as.h2o(data_test,destination_frame = "h2o_data_test")


y <- "target"
x <- setdiff(names(data_train_h), y)


#grid search
hidden_opt <- list(c(200,200), c(100,300,100), c(500,500,500))
l1_opt <- c(1e-5,1e-7)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)

model_grid <- h2o.grid("deeplearning",
                        hyper_params = hyper_params,
                        x = (2:ncol(data_train_h)),
                        y = 1,
                        #distribution = "multinomial",
                        training_frame = data_train_h,
                        validation_frame = data_test_h)

# print out the Test MSE for all of the models
for (model_id in model_grid@model_ids) {
  model <- h2o.getModel(model_id)
  mse <- h2o.mse(model, valid = TRUE)
  #mse <- h2o.mse(model, valid = FALSE)
  print(sprintf("Test set MSE: %f", mse))
}





h2o.shutdown()

# Q4
