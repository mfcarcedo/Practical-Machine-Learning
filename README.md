# Practical-Machine-Learning

# Introduction to the Assignment


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  In this prediction exercise we will data from accelerometers on the belt, forearm, arm and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har). 

The goal of the project is to predict the manner in which they did the exercise, see also the "classe" variable in the training set, using any of the other variables to predict with. This report describes i) how to build a model for the prediction ii) how to use cross validation iii) what we think the expected out of sample error is iv) explanations about the choices made for this prediction project. At the end of this report we use the prediction model to predict 20 different test cases as described in the Data section.

# The Data

Six young health men were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). These type of specifications are collected into the datasets under the "classe" variable. 

For data recording 9 degrees of freedom Razor inertial measurement units (IMU) were used, which provide three-axes acceleration, gyroscope and magnetometer data at a joint sampling rate of 45 Hz. For feature extraction a sliding window approach was used, with different lengths from 0.5 second to 2.5 seconds, and with a 0.5 second overlap. In each step of the sliding window approach features were calculated on the Euler angles (roll, pitch and yaw), as well as the raw accelerometer, gyroscope and magnetometer readings. For the Euler angles of each of the four sensors eight features were calculated: mean, variance, standard deviation, max, min, amplitude, kurtosis and skew- ness, generating in total 96 derived feature sets.

Two separte sets of these data are provided for the prediction exercise. A training set and a so called test set with 20 cases recorded that are used to test the prediction model. 

The training data for this project are available here: 
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
The test data are available here: 
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

More information on the "Qualitative Activity Recognition of Weight Lifting Exercises" that provides the data set can be found [here](http://groupware.les.inf.puc-rio.br/har)

# Approach for building the Prediction Model

###Outcome and prediction process

In our prediction models the outcome (variable to be predicted) is the classe variable in the orginal training set. This is a factor variable with 5 levels, from A to E. All other variables in the original traning set are eligible to be included as predictors in the model, however, a preprocessing first step will take place to clean the data and select the features that will be finally included for fitting the model.

Two models will be fitted, a decision tree and a random forest, applying cross-validation. The two models will be fitted on a training set, and tested on a testing data. The final model will be selected based on highest accuracy in the predicted outcome as returned in the testing set.

A prediction exercise will be perfomed lastly, applying the selected final model to the original test set, renamed as predicionSet after donwload.

### Cross Validation

Cross validation will be performed by subsampling the original training data set (randomly without replacement) into 2 subsamples: a training set comprised of 70% of the data, and a testing set comprised of 30% of the data. 

### Accuracy and Expected Out-of-sample error

Accuracy is the proportion of correct classified observation (predictions in the testing set vs observed outcome in testing set) over the total sample in the testing set. Thus, out-of-sample error is the proportion of missclasified observations in the test set, corresponding to the value 1-accuracy.

Expected accuracy in the expected proportion of correct classified observations in the prediction set, and corresponds to the accuracy in the test set. Subsequently, the expected value of the out-of-sample error will correspond to the expected number of missclassified observations in the prediction set, which is the quantity: 1-accuracy found from the cross-validation data set.

# Performing prediction model

### Data and package downloads

```{r installing packages, echo=TRUE, results='hide', warning=FALSE, message=FALSE}
library (caret); library (dplyr); library (rpart); library (randomForest); library (rattle)
```

Data download

```{r data download, results='asis', warning=FALSE}
Urltrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Urltest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("./Machine_Learning_Final_Assignment")) {
        dir.create ("./Machine_Learning_Final_Assignment")
}

download.file (Urltrain, destfile="./Machine_Learning_Final_Assignment/train.csv", method = "curl")
download.file (Urltest, destfile="./Machine_Learning_Final_Assignment/test.csv", method = "curl")
```

After inspection of the two data sets, it was observed that there are missing values in the tables, labeled as NA, blank cells and #DIV/0!. These three categories of missing data will be identified as such in the next step (reading the data)

Reading data and renaming original test set as predictionSet:

```{r reading data sets, results='asis', warning=FALSE}
trainset <- read.csv("./Machine_Learning_Final_Assignment/train.csv", na.strings=c("NA", " ", "#DIV/0!"))
predictionSet <- read.csv("./Machine_Learning_Final_Assignment/test.csv", na.strings=c("NA", " ", "#DIV/0!"))
dim(trainset); dim(predictionSet)
```

### Cleaning data

To clean the data we will first of all identify features (columns) with all data missing in both sets and delete the columns from the data sets, because we know that high prevalence of missing data can interfere with the performance of the model fitting process.

```{r cleaning NA features, results='asis', warning=FALSE}
cleanTrain <- trainset[,-colSums(is.na(trainset))==0]
cleanPrediction <- predictionSet[,-colSums(is.na(predictionSet))==0]
dim(cleanTrain); dim(cleanPrediction)
```

We want to also eliminate from our potencial predictors features with near zero variability which will not add substantive information to our fitted models and can, otherwise, introduce noise in our model predictions.

```{r near zero variance, results='asis', warning=FALSE}
nearzero <-nearZeroVar(cleanTrain, saveMetrics=TRUE)
subset(nearzero, nzv==TRUE)
```

Only one variable shows little variability. In addition a closer inspection to the training set shows that the first seven columns contain information which is not really relevant to the outcome we want to predict (such as identifiers, names of the subjects, windows, etc), including the variable with near zero variability. We will also delete these 7 columns from the two sets:

```{r deleting non relevant info, warning=FALSE}
cleanTrain <- cleanTrain[,-c(1:7)]
dim(cleanTrain)
cleanPrediction <-cleanPrediction[,-c(1:7)]
dim(cleanPrediction)
```

### Subsampling for cross-validation

We are then going to fit the models using 52 variables as predictors and 1 variable (classe) as outcome. But first we are going to subsample the train set in two sets for cross-validation: a training set and a testing set:

```{r data partition, warning=FALSE}
set.seed(22332)
inTrain <-createDataPartition(cleanTrain$classe, p=0.7, list=FALSE)
training <- cleanTrain[inTrain,]
testing <- cleanTrain[-inTrain,]
```

###Model fitting

Lets first fit a decision tree model in the training set, generate the predictors in the testing set and asses the performance of the model in testing set.

```{r decision tree, warning=FALSE}
modFit1 <- rpart(classe ~ ., data=training, method="class")
pred1 <- predict(modFit1, testing, type="class")
confusionMatrix(pred1, testing$classe)
```

The accuracy rate of the decision tree model is 73.47%.

The final model dendrogram looks like this:

```{r dendrogram, out.width= 1500 , out.height=1500}
fancyRpartPlot (modFit1, main="Decision Tree")
```

Now we will fit the random forest model in the training set, test it in the testing set and asses performance of the model:

```{r random forest, warning=FALSE}
modFit2 <- randomForest(classe~., training, method="class")
pred2 <- predict(modFit2, testing, type="class")
confusionMatrix(pred2, testing$classe)
```

###Conclusion

The accuracy rate of the random forest model is 99.54% by far more performant than accuracy rate of the decision rate model. If we use the random forest model to predict the classe variable we will expect to get an out-of-sample error of 0.06%, near zero error. 

We will thus use this random forest model to predict the 20 cases in the prediction set.

# Prediction

In the Prediction set we apply the model selected for its higher accuracy. The 20 cases predicted are shown below:

```{r prediction, warning=FALSE}
prediction <- predict(modFit2, cleanPrediction, type="class")
prediction
```

