# Sepsis Prediction Using Fuzzy Logic

## Overview
This project was completed as part of the BBM 407 - Fuzzy Logic course at Hacettepe University's Engineering Faculty, Computer Engineering Department. It explores two machine learning approaches for predicting sepsis:
- Mamdani-style Fuzzy Inference System
- Adaptive Neuro-Fuzzy Inference System (ANFIS)

## Project Structure
- `Midterm.py`: Implementation of Mamdani Fuzzy Inference System
- `Final.py`: Implementation of ANFIS for sepsis prediction
- `MidtermReport.pdf`: Detailed midterm report explaining methodology
- `FinalReport.pdf`: Comprehensive final report with results and analysis

## Methodology

### Data Preparation
- Collected datasets of sepsis and non-sepsis patient records
- Processed and cleaned data by:
  - Handling missing values
  - Feature selection based on correlation and missing data
  - Normalizing input features

### Fuzzy Logic Approach
- Used skfuzzy library for fuzzy inference
- Defined membership functions for various medical features
- Created fuzzy rules based on medical research
- Implemented defuzzification to classify sepsis risk

### ANFIS Approach
- Utilized TensorFlow for adaptive learning
- Employed Gaussian membership functions
- Used Adam optimizer with binary cross-entropy loss
- Implemented 5-fold cross-validation

## Key Features
- Feature selection techniques
- Multiple medical feature analysis
- Advanced machine learning techniques
- Performance evaluation using metrics like:
  - Accuracy
  - F1-score
  - Precision
  - Recall
  - ROC-AUC Score
