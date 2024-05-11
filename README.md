# Thyroid Disease classification with machine learning approaches

This project is about prediction of Hypothyroidism and Hyperthyroidism using machine learning approaches. The data is obtained from the UCI Machine Learning Repository. The data is preprocessed and various machine learning algorithms are applied to predict the disease. The project is divided into 5 parts:

1. Data Preprocessing
2. Exploratory Data Analysis
3. Model Building
4. Model Evaluation
5. Model explanation

The data is preprocessed by removing missing values, encoding categorical variables, and scaling the data. The Exploratory Data Analysis is done to understand the data and the relationship between the features. The model is built using various machine learning algorithms such as Logistic Regression, Random Forest, Gradient Boosting, etc. The model is evaluated using various metrics such as precision, recall, f1-score, and accuracy. The results are compared and the best model is selected.

The project is implemented in Python using Jupyter Notebook. The libraries used are pandas, numpy, matplotlib, seaborn, scikit-learn, and xgboost.

## Table of Contents

1. [Installation](#installation)
2. [Prerequisites](#prerequisites)
3. [Requirements](#requirements)

## Installation

```bash
$ git clone

## Prerequisites

What things you need to have to be able to run:

  * Python 3.6 +
  * Pip 3+
  * VirtualEnvWrapper is recommended but not mandatory

## Requirements 

```bash
$ pip install requirements.txt
```

## Explanable AI

This project uses SHAP for explainable AI. SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.