# Titanic Survival Prediction Project

## Description of the Problem

This project focuses on predicting the survival likelihood of passengers aboard the Titanic using machine learning techniques. The dataset includes information about passengers, such as their age, gender, class, and other features, and the goal is to build a model that can predict whether a passenger survived or not.

The primary tasks involved in the project are:

1. **Exploratory Data Analysis:** Analyze and understand the dataset, handle missing values, and explore relationships between different features.
2. **Data Preprocessing & Feature Engineering:** Prepare the data for training machine learning models by handling missing values, encoding categorical variables, and creating new features.
3. **Model Training:** Train and evaluate machine learning models on the preprocessed data to predict passenger survival.
4. **Prediction Service**: Create a Dockerized /predict endpoint.
5. **Batch Prediction:** Use the trained model to predict survival for a batch of passengers in the test dataset..

## Instructions on How to Run the Project

1. **Data Directory:** The `data` directory contains the raw and processed datasets.
2. **Notebooks Directory:** The `notebooks` directory includes Jupyter notebooks for different stages of the project, such as EDA, data preprocessing, and model training.
3. **Prediction Service Directory:** The `prediction_service` directory contains code for deploying a prediction service using Flask. To run the prediction service:

   ```bash
   cd prediction_service
   python predict.py
   ```
   This will start the Flask app, and you can make predictions by sending POST requests to the `/predict` endpoint.
4. 
5. **Scripts Directory:** The `scripts` directory contains Python scripts for specific tasks, such as preprocessing data and batch prediction. To run the batch prediction script:

   ```bash
   cd scripts
   python batch_predict.py ../data/processed/test.csv
   ```
   This will create a CSV file (`predictions.csv`) with predictions for each passenger in the test dataset.
