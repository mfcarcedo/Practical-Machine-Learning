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
