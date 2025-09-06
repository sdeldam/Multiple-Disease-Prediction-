# Multiple Disease Prediction  

This repository contains code to **predict multiple diseases** using machine learning models trained on blood sample data.  

## Dataset  
- **blood_sample_dataset.csv** – Contains patient blood sample features along with disease labels.  

## Workflow  

### 1. Data Loading & Preprocessing  
- Load dataset using **pandas**.  
- Encode categorical labels using **LabelEncoder**.  
- Split data into **training** and **testing** sets with **train_test_split**.  

### 2. Model Training  
Implemented multiple classifiers:  
- **Random Forest Classifier**  
- **Decision Tree Classifier**  

### 3. Model Evaluation  
- Evaluate model performance using:  
  - **Confusion Matrix**  
  - Accuracy and classification metrics  

### 4. Visualization  
- **Scatterplots** to visualise data distribution.  
- **Bar charts** to compare model results.  

## Requirements  
- Python 3.x  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
