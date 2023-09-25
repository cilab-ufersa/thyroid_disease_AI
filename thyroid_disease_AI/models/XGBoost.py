import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
from xgboost import XGBClassifier #Para criar o modelo de árvore de decisão
import matplotlib.pyplot as plt 
from utils import *
from sklearn.model_selection import GridSearchCV
import joblib

file_name = "XGBoost.joblib"

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
    # Definindo o espaço de busca 
    param_grid_xgb = { 
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        }
    model = XGBClassifier()
    model_grid = GridSearchCV(model, param_grid_xgb, cv=5)
    model_grid.fit(input_train, output_train)
    
    print("Melhores parâmetros:", model_grid.best_params_)
    print("Melhor pontuação:", model_grid.best_score_)
    
    {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 200, 'subsample': 0.8}
    {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8, 'learning_rate': 0.1, 'n_estimators': 100}
    '''

    
    model = XGBClassifier(
        colsample_bytree = 0.1,
        gamma = 0.5,
        learning_rate = 0.04,
        max_depth = 9,
        min_child_weight = 5,
        n_estimators = 200,
        subsample = 0.1)
    model.fit(input_train, output_train) #Treinamento

    joblib.dump(model, 'thyroid_disease_AI\models_file\XGBoostClassifier.sav')

    # Fazer a classificação
    output_model_decision = model.predict(input_test)

    #Plotando
    plot_confusion_matrix(output_test.values, output_model_decision, model, title = 'Matriz Confusão')

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    roc(output_test, output_model_decision) #plotando a curva ROC
 
    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model)     #plotando a curva de erro