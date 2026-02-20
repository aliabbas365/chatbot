Employee Attrition AI Chatbot (Flask + Machine Learning)
Overview

This project is a machine learning-powered chatbot/API that predicts whether an employee is likely to leave the company (Attrition prediction).

The model is trained using Scikit-learn and deployed using a Flask server that allows predictions through API endpoints or a simple web interface.

The system:

Trains a model using DATA.csv

Saves the trained model

Serves predictions via REST API

Provides a UI to interact with the model

Features

Train ML model from dataset

Employee attrition prediction

REST API for predictions

Input preprocessing and feature encoding

Automatic feature alignment

Simple web interface

Test script for quick validation

Project Structure
chatbot-main/
│
├── app.py                # Train the machine learning model
├── server.py             # Flask API server
├── DATA.csv              # Training dataset
├── rf_model.pkl          # Saved trained model
├── scaler.pkl            # Feature scaler
├── feature_columns.pkl   # Feature columns used during training
├── requirements.txt      # Dependencies
├── test_run.py           # API testing script
├── sample_payload.json   # Example request payload
│
└── static/
    └── index.html        # Simple UI
How It Works
Step 1: Train the Model

The training script:

Loads the dataset

Handles missing values

Encodes categorical features

Trains a Random Forest model

Saves the trained model

Run:

python app.py

Output:

rf_model.pkl
scaler.pkl
feature_columns.pkl
Step 2: Start the Server

Run:

python server.py

Server runs on:

http://127.0.0.1:5000
API Endpoints
Health Check
GET /

Response:

Server is running
View Model Info
GET /model-info

Shows:

Model status

Scaler status

Number of features

Get Feature List
GET /features

Returns all model input features.

Preview Processed Input
POST /preview

This shows how input data is processed before prediction.

Predict Attrition
POST /predict

Example request:

{
  "Age": 41,
  "MonthlyIncome": 5993,
  "YearsAtCompany": 6
}

Response:

Yes / No
Run Test Script

You can test the system quickly:

python test_run.py

This will:

Test the server

Send sample data

Show predictions

Installation

Clone the repository:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

Install dependencies:

pip install -r requirements.txt
Technologies Used

Python

Flask

Scikit-learn

Pandas

NumPy

Machine Learning

REST API

Future Improvements

Convert into full chatbot interface

Deploy on cloud (AWS / Render / Railway)

Improve model accuracy

Add authentication

Dashboard for HR teams

Author

Ali Abbas

AI / Machine Learning Project
