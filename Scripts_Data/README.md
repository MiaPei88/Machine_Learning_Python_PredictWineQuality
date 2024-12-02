Files in the zip file:
README.md
run_wine_quality_pred.sh
wine_quality_pred.py
winequality-white.csv
Project_Report_ANSC6100.docx

Usage
To run the wine_quality_pred.py script, execute the following command in Linux/Unix terminals:
./run_wine_quality_pred.sh
Ensure that Python 3 is installed on your system before running the script.
Ensure the file run_wine_quality_pred.sh is executable by running 'chmod +x run_wine_quality_pred.sh'


Comparison of Machine Learning Model Performances on Predicting White Wine Quality in Minho, 
Portugal

Introduction

Welcome to the README file for the manuscript “Comparison of Machine Learning Model Performances 
on Predicting White Wine Quality in Minho, Portugal”. 
This project focuses on predicting the quality of white wine using machine learning techniques. 
The primary goal is to train and compare the performance of various machine learning models in 
predicting wine quality.

Dataset
The dataset utilized for this study was donated to the UC Irvine Machine Learning Depository 
on October 6, 2009 (https://archive.ics.uci.edu/dataset/186/wine+quality). It comprises 4898 
white wine samples, each characterized by 11 physicochemical features and a quality score 
ranging from 0 to 10. Only white wine samples are utilized for this study.
 
Features
Input variables (based on physicochemical tests): 
1. fixed acidity 
2. volatile acidity 
3. citric acid 
4. residual sugar 
5. chlorides 
6. free sulfur dioxide 
7. total sulfur dioxide 
8. density 
9. pH 
10. sulphates 
11. alcohol 
Output variable (based on sensory data): 
12. quality (discrete score between 0 and 10) 
 
Pipeline Summary
1. Exploratory Data Analysis: 
   Understand the structure and distribution of the dataset.
2. Data Cleaning and Splitting: 
   Preprocess the data and split it into training (80%) and testing sets (20%).
3. Model Fitting and Feature Selection: 
   Train machine learning models and select top 5 most features.
4. Model Validation and Preliminary Model Comparison: 
   Evaluate model performance and compare initial results.
5. Hyperparameter Optimization: 
   Tune model hyperparameters using the random search method to improve performance.
6. Final Model Fitting: 
   Train the final model using optimized hyperparameters.
7. Performance Evaluation & Comparison: 
   Evaluate and compare final models’ performances.
