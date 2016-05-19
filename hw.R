setwd("~/bgse/15D013_Topics_2/Prof. A. Arratia")

library(quantmod); library(xts)
library(fBasics)

getSymbols("AAPL",from="1997-01-01")
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
  
  df$rtrn2 <- df$rtrn^2
  df$rtrn3 <- df$rtrn^3
  df$rtrn4 <- df$rtrn^4
  df$rtrn5 <- df$rtrn^5
  
  
  ac <- acf(df$rtrn)
  ac2 <- acf(df$rtrn2)
  ac3 <- acf(df$rtrn3)
  ac4 <- acf(df$rtrn4)
  ac5 <- acf(df$rtrn5)
  
  plot(ac,type="l")
  lines(ac2$acf, col="red")
  lines(ac3$acf, col="blue")
  lines(ac4$acf, col="yellow")
  lines(ac5$acf, col="orange")
  
  
}

acplot("AAPL")

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
Box.test(df$rtrn, type = "Ljung-Box")
Box.test(df$rtrn2, type = "Ljung-Box")
plot(df$rtrn, type="l")
plot(df$rtrn2, type="l")

library(fGarch)

ar <- arma(df$rtrn2)
ar
ga <- garch(df$rtrn2)
ga

arma_gar <- garchFit(formula = ~arma(1,1)+garch(1,1), data = df$rtrn2)

# Q3

log_return <- function(vec){
  
  #v1
  rtrn <- rep(0,length(vec))
  
  for(i in 2:length(vec)){
    rtrn[i] <- vec[i]/vec[(i-1)]-1
  }
  
  logrtrn <- 100*log(1+rtrn)
  
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

lagdata <- function(set,lag){
  x <- data.frame(set[,1])
  for(i in 1:lag){
    y <- set[-c(nrow(set):(nrow(set)-i+1)),]
    x <- data.frame(x[-1,],y)
  }
  return(x)
}

x <- lagdata(dta,2)
head(x)
