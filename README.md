# Diabetes Prediction & Visualization App

This project is a complete pipeline for data preprocessing, model training, and deployment of a web application using **Streamlit** to predict diabetes based on user-input health data. It also includes various **interactive data visualizations** to help users understand key features associated with diabetes.

---

## Features

-  **Visualizations**: Histogram, Line Chart, Scatter Plot, and Heat Map for in-depth data analysis.
-  **ML Model**: Trained using Random Forest Classifier on Pima Indian Diabetes dataset.
-  **Prediction**: Predicts diabetic condition from user input.
-  **Feedback**: Section for users to submit feedback with contact information.

---

## Project Structure

```bash
.
├── strealit_demo.py       # Streamlit app frontend and interaction
├── model_train.py         # Script to train and save the ML model
├── dataframe.py           # Preprocessing and clean dataset preparation
├── diabetes.csv           # CSV dataset (expected path: ./data/diabetes.csv)
├── model.sav              # Trained model pickle file
├── README.md              # Project overview
└── requirements.txt       # Python dependencies

## Dataset
Pima Indians Diabetes Dataset

Ensure the dataset is stored in a folder like ./data/diabetes.csv or update the path accordingly in model_train.py and dataframe.py.

## How to Run
### Install dependencies:

pip install -r requirements.txt

### Train the model:

python model_train.py
This will train and optionally save the model (you may need to add a joblib.dump or pickle.dump manually).

### Run Streamlit App:

streamlit run streamlit_demo.py