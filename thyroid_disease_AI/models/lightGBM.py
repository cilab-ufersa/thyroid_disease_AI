import sys
sys.path.append('thyroid_disease_AI')
import lightgbm as lgb
import pandas as pd
from utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib

if __name__ == '__main__':
    #Carregando o dataset
    #80 % para treino e 20% para teste
    input_train = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\input_train.csv')
    input_train = input_train.values
    input_test = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\input_test.csv')
    input_test = input_test.values
    output_train = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\output_train.csv')
    output_test = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\output_test.csv')
    
    '''
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'n_estimators': [50, 100, 200],
        'num_leaves': [10, 20, 30],
        'subsample': [0.8, 0.9, 1.0]
    }
    model_lgb = lgb.LGBMClassifier()
    grid = GridSearchCV(estimator = model_lgb, param_grid = param_grid, cv=5, scoring='accuracy')
    grid.fit(input_train, output_train)
    print(grid.best_params_)

    {'learning_rate': 0.2, 'max_depth': 5, 'num_leaves': 31}
    {'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 200, 'num_leaves': 30, 'subsample': 0.8}
    '''

    # Carregando o modelo
    model = lgb.LGBMClassifier(
        learning_rate = 3.000001,
        max_depth = 5,
        n_estimators = 100,
        num_leaves = 15, 
        subsample = 0.8,
    )
    model.fit(input_train, output_train) # Treinando o modelo

    # joblib.dump(model, 'thyroid_disease_AI\models_file\LightGBM.sav')

    #realizando predições
    output_model_light = model.predict_proba(input_test)[:, 1]

    auc = roc_auc_score(output_test, output_model_light)

    print(f'AUC: {auc}')
    
    print("\n\n\n\n\n")
    
    plot_confusion_matrix(output_test, output_model_light, model, title = 'Matriz Confusão')


    accuracy(output_test, output_model_light) #Pontuação de acurácia
    
    precision(output_test, output_model_light) #Pontuação de precisão

    recall(output_test, output_model_light) #Pontuação de recall

    f1(output_test, output_model_light)
    
    roc(output_test, output_model_light) #plotando a curva ROC
    
    print("\n\n\n\n\n")

    #plotando a curva de erro
    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model)

    #learning_curves(input_train, output_train, input_test, output_test, model)