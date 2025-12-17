🩺 Medical Risk Score Predictor

1.Heart Disease 2.Diabetes 3.Stroke

A machine learning–based healthcare application that predicts Heart Disease, Diabetes, and Stroke risk using clinical and lifestyle features, with simple and stable explainability.
Built using XGBoost, Python, and Streamlit.


🚀 Project Overview:-

This project demonstrates an end-to-end applied ML pipeline in a healthcare setting:
1.Multi-disease risk prediction
2.Real-world dataset integration
3.Robust preprocessing & feature engineering
4.Separate disease-specific ML models
5.Explainable predictions
6.Interactive Streamlit web application


🔍 Diseases Covered:-
Disease	Model
❤️ Heart Disease	XGBoost Classifier
🔥 Diabetes      	XGBoost Classifier
🧠 Stroke	        XGBoost Classifier

An Overall Health Risk Score is computed as a weighted combination of the three predictions.

📊 Datasets Used:-

1.Framingham Heart Study Dataset
2.Kaggle Cardiovascular Disease Dataset (70K rows)
3.CDC Diabetes Health Indicators Dataset (250K+ rows)

These datasets were:-
Cleaned individually
Unified into a common schema
Used to train separate disease-specific models


🧠 Machine Learning Approach:-
Algorithm: XGBoost

Why XGBoost?
Handles missing values well
Captures non-linear feature interactions
Performs strongly on tabular healthcare data
Class imbalance handling: scale_pos_weight

Evaluation Metrics:
Precision, Recall, F1-score
ROC–AUC


📈 Model Performance (Approx.):-

Model	         ROC-AUC
Heart Disease	 ~0.80
Diabetes	     ~0.79
Stroke         	 ~0.79


🧩 Explainability (Simple & Stable):-

Instead of complex SHAP plots, this project uses local sensitivity-based explainability:
“How does the prediction change if one feature is slightly modified while others remain fixed?”
For example:
Increasing systolic BP → higher heart risk
Higher BMI → increased risk
Physical activity → reduced risk
This approach is:
Stable in production
Easy to understand
Suitable for non-technical users


🖥️ Web Application (Streamlit):-
Features:
Interactive sliders for patient inputs
Real-time risk prediction
Overall health risk score
Per-patient explainability
Clean dark-themed UI