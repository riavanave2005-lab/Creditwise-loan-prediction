# 🏦 CreditWise - Loan Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

A comprehensive machine learning system for predicting loan approval status using supervised learning algorithms.

## 📋 Overview

CreditWise is an end-to-end machine learning pipeline that predicts whether a loan application should be approved or rejected. The system implements multiple supervised learning algorithms and provides a user-friendly web interface for making predictions.

## 🚀 Features

- **Data Preprocessing**: Automated cleaning, encoding, and scaling
- **Feature Engineering**: Ratio features, log transformations, interaction features
- **Multiple Models**: KNN, Logistic Regression, and Naive Bayes
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Web Interface**: Flask-based web application with beautiful UI

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **88.5%** | **86.6%** | **93.6%** | **90.0%** | **91.7%** |
| KNN | 84.5% | 79.7% | 96.4% | 87.2% | 91.8% |
| Naive Bayes | 85.0% | 79.9% | 97.3% | 87.7% | 89.4% |

## 🛠️ Tech Stack

- Python 3.8+
- scikit-learn (ML algorithms)
- Flask (Web framework)
- Pandas, NumPy (Data processing)
- Matplotlib, Seaborn (Visualizations)


