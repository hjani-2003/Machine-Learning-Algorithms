library(MASS)
library(olsrr)

mlr <- function()
{
  
  df <- read.csv('dummy.csv')
  
  print(head(df))
  
  full.model <- lm(Y ~ ., data = df)
  
  backward.model <- stepAIC(full.model, direction = "backward", trace = 0)
  bck.sum <- summary(backward.model)
  
  print(bck.sum)
  hist(backward.model$residuals, main="Residual Histogram")
  
 
  newData <- data.frame(
    X=1000,
    X1=2000,
    X2=55000
  )
  backward.pred <- predict(backward.model, newdata = newData)
  print(paste("Predicted Value : ", backward.pred))
}


mlr()
  