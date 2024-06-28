# Health-Insurance-prediction

The purpose of this project is to determine the contributing factors and predict health insurance cost by performing exploratory data analysis and predictive modeling on the Health Insurance dataset. This project makes use of Numpy, Pandas, Sci-kit learn, and Data Visualization libraries.

Overview:
• Seek insight from the dataset with Exploratory Data Analysis
• Performed Data Processing, Data Engineering and Feature Transformation to prepare data before modeling
• Built a model to predict Insurance Cost based on the features
• Evaluated the model using various Performance Metrics like RMSE, R2, Testing Accuracy, Training Accuracy and MAE

DATA PROCESSING

Check missing value - there are none
Check duplicate value - there are 1 duplicate, will be remove
Feature engineering - make a new column weight_status based on BMI score
Feature transformation:
A) Encoding sex, region, & weight_status attributes
B) Ordinal encoding smoker attribute
Modeling:
A) Separating target & features
B) Splitting train & test data
C) Modeling using Linear Regression, Random Forest, Decision Tree, Ridge, & Lasso algorithm
D) Find the best algorithm

Based on the predictive modeling, Linear Regression algorithm has the best score compared to the others,  R2 Score 0.75.
