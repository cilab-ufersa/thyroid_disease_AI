import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import RandomForestClassifier #Para criar o modelo de árvore de decisão
from utils import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import randint
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
    param_dist = {
        'n_estimators': [10, 200],
        'max_depth': [1, 20],
        'min_samples_split': [2, 20],
        'min_samples_leaf': [1, 20],
        'max_features': ['auto', 'sqrt', 'log2', None],
        }
          
    model_random = RandomForestClassifier()
    model_grid = RandomizedSearchCV(model_random, param_distributions=param_dist, n_iter=100, scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
    model_grid.fit(input_train, output_train) #Treinamento

    print(model_grid.best_params_)
    {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 20}
    '''
    model = RandomForestClassifier(
                                n_estimators=12,
                                max_depth=3,
                                max_features ='log2',
                                min_samples_split = 2,
                                min_samples_leaf = 1,
                                )
    
    model.fit(input_train, output_train) #Treinamento
    
    joblib.dump(model, 'thyroid_disease_AI\models_file\RandomCV.sav')

    # Fazer a classificação 
    output_model_random = model.predict_proba(input_test)[:, 1]

    auc = roc_auc_score(output_test, output_model_random)

    print(f'AUC: {auc}')

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_random, model, title='Matriz confusão')

    accuracy(output_test, output_model_random) # Pontuação de acurácia
    
    precision(output_test, output_model_random) # Pontuação de precisão

    recall(output_test, output_model_random) # Pontuação de recall

    f1(output_test, output_model_random) # Pontuação de F1
    
    roc(output_test, output_model_random) # Plotando a curva ROC

    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model) # Plotando a curva de erro