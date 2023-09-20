import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
from xgboost import XGBClassifier #Para criar o modelo de árvore de decisão
import matplotlib.pyplot as plt 
#import seaborn as sns
from utils import *
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold
#import joblib

file_name = "XGBoost.joblib"

if __name__ == '__main__':

    #Carregando o dataset
    #80 % para treino e 20% para teste
    input_train = pd.read_csv('C:\\Users\\Usuario\\Documents\\Projects\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\input_train.csv')
    input_train = input_train.values
    input_test = pd.read_csv('C:\\Users\\Usuario\\Documents\\Projects\\thyroid_disease_AI\\thyroid_disease_AI\datasets\\hypothyroid\\input_test.csv')
    input_test = input_test.values
    output_train = pd.read_csv('C:\\Users\\Usuario\\Documents\\Projects\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\output_train.csv')
    output_test = pd.read_csv('C:/Users/Usuario/Documents/Projects/thyroid_disease_AI/thyroid_disease_AI/datasets/hypothyroid/output_test.csv')

    '''
    # Definindo o espaço de busca 
    param_grid_xgb = { 
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]

        }
    model_grid = GridSearchCV(model, param_grid_xgb, cv=5)
    model_grid.fit(input_train, output_train)
    
    print("Melhores parâmetros:", model_grid.best_params_)
    print("Melhor pontuação:", model_grid.best_score_)
    '''
    model = XGBClassifier(
        colsample_bytree = 0.8,
        learning_rate = 0.01,
        max_depth = 3,
        n_estimators = 50,
        subsample = 0.8)
    model.fit(input_train, output_train) #Treinamento

    #joblib.dump(model, 'thyroid_disease_AI\models_file\XGBoostClassifier.sav')

    # Fazer a classificação
    output_model_decision = model.predict(input_test)

    #pickle.dump(model, open(file_name, "wb"))
    #model.save_model('XGBoost.sav')

    #Plotando
    plot_confusion_matrix(output_test.values, output_model_decision, model, title = 'Matriz Confusão')

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    roc(output_test, output_model_decision) #plotando a curva ROC
 
    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model)     #plotando a curva de erro