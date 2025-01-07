# Maternal-Health-Risk-Prediction-System
# Abstract
    Predicting maternal health risks is a critical task due to the complexity of medical parameters and variations in individual health conditions. This project leverages machine learning to classify maternal health risks into Low Risk, Mid Risk, and High Risk categories. The system aims to assist healthcare providers by identifying high-risk cases, enabling timely interventions and informed decision-making.

    Using a pre-processed dataset of 100,000 records, classification algorithms like Naive Bayes, K-Nearest Neighbors (KNN), Logistic Regression, and Linear Regression analyze patterns in patient data. Key features include age, pregnancy count, smoking status, alcohol use, body temperature, heart rate, and blood pressure. The system delivers early risk predictions to reduce maternal mortality by enabling proactive healthcare decisions.

# Problem Statement
    Develop an accurate predictive model to identify pregnancy complications and optimize maternal health outcomes.

# Objectives
    Enhance Data Quality and Relevance: Ensure robust preprocessing to maintain data integrity.
    Identify Key Predictive Features: Focus on critical attributes influencing maternal health risks.
    Develop Accurate Predictive Models: Build and test models for effective classification.
    Validate Model Performance: Evaluate algorithms using metrics like accuracy and F1 score.
    Optimize Patient Outcomes: Provide healthcare professionals with actionable insights.
# Dataset Description
    Dataset Name: Maternal Health Dataset

    Size: 100,000 records

    Source: Kaggle

    Features (Input Variables):
  
        Patient_ID: Unique identifier for each patient.
        Age: Age of the patient (in years).
        Pregnancy_Count: Total number of pregnancies.
        Smoking_Status: Whether the patient smokes: Yes = 1, No = 0.
        Alcohol_Use: Whether the patient consumes alcohol: Yes = 1, No = 0.
        Body_Temperature: Patient’s body temperature (in °C).
        Heart_Rate: Patient’s heart rate (in beats per minute).
        Systolic_BP: Upper reading of blood pressure (in mmHg).
        Diastolic_BP: Lower reading of blood pressure (in mmHg).
    Target Variable:

      Risk_Level: Predicted maternal health risk level:
        Low Risk = 0
        Mid Risk = 1
        High Risk = 2
# Algorithms Used
      Naive Bayes Classifier
      K-Nearest Neighbors (KNN)
      Logistic Regression
      Linear Regression
# Methodology
    Data Preprocessing:

      Handle missing values.
      Encode categorical variables.
      Scale numerical features.
      Remove outliers to improve model performance.
    Model Training:

      Split the dataset into different training and testing ratios (e.g., 50:50, 60:40, 70:30, 80:20).
      Train models on the dataset using the selected algorithms.
    Model Evaluation:

      Use accuracy and F1 score to compare performance.
      Evaluate models’ ability to generalize across different train-test splits.
    Deployment:

      Create a scalable system for real-world maternal health risk assessment.
# Results and Findings
    Logistic Regression performed the best due to its ability to handle linear relationships and scalability for large datasets.
    Models effectively classified maternal health risks into Low, Mid, and High categories.
    The system proved scalable and efficient for real-world applications.
# Conclusion
    This project demonstrates the successful application of machine learning in maternal health risk prediction.
    Early risk detection can assist healthcare providers in prioritizing care, optimizing resources, and reducing maternal mortality.
    Future improvements could include incorporating lifestyle, medical history, and socio-economic data for better predictions.

## **License**

This project is licensed under the [MIT License] (LICENSE).
