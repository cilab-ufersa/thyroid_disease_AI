import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import ExtraTreesClassifier #Para criar o modelo de árvore de decisão
from utils import *
from sklearn.model_selection import GridSearchCV
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
    'n_estimators': [50, 100, 200],           
    'max_depth': [None, 10, 20, 30],        
    'min_samples_split': [2, 5, 10],    
    'min_samples_leaf': [1, 2, 4],
    }

    model = ExtraTreesClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(input_train, output_train) #Treinamento
    print(grid_search.best_params_)

    {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
    '''


    # Carregando o modelo
    model = ExtraTreesClassifier(
                                max_depth=30,
                                min_samples_leaf=4,
                                min_samples_split=4,
                                n_estimators=50,
                                    )
    model.fit(input_train, output_train) # Treinando o modelo
    
    joblib.dump(model, 'thyroid_disease_AI\models_file\ExtraTrees.sav')

    # Fazer a classificação 
    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model, title='Matriz confusão')

    accuracy(output_test, output_model_decision) # Pontuação de acurácia
    
    precision(output_test, output_model_decision) # Pontuação de precisão

    recall(output_test, output_model_decision) # Pontuação de recall

    f1(output_test, output_model_decision) # Pontuação de F1
    
    roc(output_test, output_model_decision) # Plotando a curva ROC

    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model) # Plotando a curva de erro