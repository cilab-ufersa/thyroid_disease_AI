import sys
sys.path.append('thyroid_disease_AI')
import lightgbm as lgb
import pandas as pd
from utils import *
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

if __name__ == '__main__':
    #Carregando o dataset
    dataset = pd.read_csv("thyroid_disease_AI\datasets\hypothyroid\hypothyroid_dataset_clean.csv")
    dataset = dataset.drop_duplicates()
    output_label_dataset = dataset['binaryClass']
    dataset = dataset.drop(['binaryClass'], axis=1)
    
    #Balanceamento dos dados
    dataset_res, output_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)
    
    #Dividindo os dados em treino (train) e teste (test)
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset_res, output_label)

    #treinando o modelo
    
    model = lgb.LGBMClassifier(
        learning_rate = 0.3,
        max_depth = 15,
        n_estimators = 5,
        num_leaves = 15, 
        subsample = 0.5 
    )
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 4, 5, 6],
    #     'n_estimators': [50, 100, 200],
    #     'num_leaves': [10, 20, 30],
    #     'subsample': [0.8, 0.9, 1.0]
    # }
    # model = GridSearchCV(estimator = model_lgb, param_grid = param_grid, cv=5, scoring='accuracy')
    model.fit(input_train, output_train)
    #print(model.best_estimator_)

    joblib.dump(model, 'thyroid_disease_AI\models_file\LightGBM.sav')

    #realizando predições
    output_model_decision = model.predict(input_test)
    
    print("\n\n\n\n\n")
    
    plot_confusion_matrix(output_test, output_model_decision, model, title = 'Matriz Confusão')

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    roc(output_test, output_model_decision) #plotando a curva ROC
    
    print("\n\n\n\n\n")

    #plotando a curva de erro
    miss_classification(input_train, output_train, input_test, output_test, model)

    #learning_curves(input_train, output_train, input_test, output_test, model)