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
    dataset = pd.read_csv('thyroid_disease_AI\datasets\hypothyroid\hypothyroid_dataset_clean.csv')  
    dataset = dataset.drop_duplicates()
    output_label_dataset = dataset['binaryClass']
    dataset = dataset.drop(['binaryClass'], axis=1) 

    #Balanceamento dos dados 
    dataset_res, output_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset_res, output_label=output_label)

    # param_grid = {
    # 'n_estimators': [50, 100, 200],           
    # 'max_depth': [None, 10, 20, 30],        
    # 'min_samples_split': [2, 5, 10],    
    # 'min_samples_leaf': [1, 2, 4],       
    # 'class_weight': [None, 'balanced'],
    # 'criterion': ['gini', 'entropy'],
    # 'random_state': [42]
    # }

    model = ExtraTreesClassifier(class_weight='balanced', 
                                criterion='entropy',
                                n_estimators=100, 
                                random_state=50,
                                max_depth = 15)
    model.fit(input_train, output_train)
    joblib.dump(model, 'thyroid_disease_AI\models_file\ExtraTrees.sav')

    # grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(input_train, output_train) #Treinamento

    # Fazer a classificação 
    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model, title='Matriz confusão')

    accuracy(output_test, output_model_decision) # Pontuação de acurácia
    
    precision(output_test, output_model_decision) # Pontuação de precisão

    recall(output_test, output_model_decision) # Pontuação de recall

    f1(output_test, output_model_decision) # Pontuação de F1
    
    roc(output_test, output_model_decision) # Plotando a curva ROC

    miss_classification(input_train, output_train, input_test, output_test, model) # Plotando a curva de erro