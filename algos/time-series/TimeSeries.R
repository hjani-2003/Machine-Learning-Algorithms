
#Installing and importing package and dataset
# install.packages("fpp")
library("fpp")
library("forecast")
library(dplyr)

#Using a dataset for TimeSeries (Additive)
data(ausbeer)

?ausbeer
#Building a Time Series
beer <- ausbeer
head(beer)
beer

# total_sample <- nrow(b)
beer <- ts(ausbeer, start = c(1956,1),  frequency = 4)
beer
plot(beer)



#Decomposing the Time Series
beer.series = decompose(beer)
plot(beer.series)


#Deseasonalizing the Time Series
beer.deseasonalised <- beer - beer.series$seasonal
plot(beer.deseasonalised, col='black')
lines(beer,col='green')


# X <- rnorm(1000, mean=70, sd=10)
# hist(X)

#Forecasting values
yrs = 5
model <- auto.arima(beer) #We can try using deseasonalised series also
f <- forecast(model,level=c(95), h=yrs*4) # model = data, level = level of confidence, h = project intervals (here it is 5 * 12 means 5 years and 60 intervals)


plot(f, col='red', main= paste('Forecast for ',yrs, ' years'))
lines(beer)

#Testing the model with Training and Testing data
b <- matrix(ausbeer)
total_sample <- nrow(b)
total_sample
train_fraction <- 0.8
split_index <- round(total_sample * train_fraction)
split_index
train_data <- b[1:split_index, ]
train_data
train_data <- ts(train_data, start=c(1980,1), frequency = 12)
beer
train_data
test_data <- b[(split_index+1): total_sample]
test_data <- ts(test_data, start = c(1994,2), frequency = 12)
test_data

model = auto.arima(train_data)
f <- forecast(model, h = length(test_data))
plot(f)
lines(test_data, col='red')
lines(beer.series$trend, col='green')
accuracy(f, test_data)


#Using Another dataset for a different model (Multiplicative)
air <- AirPassengers
air
air <- ts(air, start = c(1949,1), frequency = 12)
plot(air)
#lines(beer,col='red')
air.decomp <- decompose(air, "multiplicative")
plot(air.decomp)



yrs = 5
model <- auto.arima(air)
summary(model)
f<- forecast(model, level=c(70), h = yrs * 12)

plot(f, col='red', main = paste('Forecast for ', yrs, ' years'))
lines(air.decomp$trend,col='black')

