import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import RandomForestClassifier #Para criar o modelo de árvore de decisão
from utils import *
from sklearn.model_selection import GridSearchCV
import joblib


if __name__ == '__main__':

    #Carregando o dataset
    dataset = pd.read_csv('thyroid_disease_AI\datasets\hypothyroid\hypothyroid_dataset_clean.csv')  
    output_label_dataset = dataset['binaryClass']
    # dataset.drop(['binaryClass'], axis=1) 
    # print(output_label_dataset.value_counts())

    #Balanceamento dos dados 
    dataset_res, output_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

    print(dataset_res['binaryClass'].value_counts())

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset_res, output_label=output_label)

    '''
    parametros = {
        'n_estimators': [50, 100, 200], # Valores que o hiperparâmetro deve assumir durante a busca em grade
        'max_depth': [None, 10, 20], # Controla a profundidade maxima das árvores nas florestas
        'min_samples_split': [2, 5, 8],
        'min_samples_leaf': np.arange(1, 5, 1, dtype=int)}
          
    model_random = RandomForestClassifier(criterion='entropy', max_features ='sqrt', class_weight='balanced', max_depth=5, n_estimators=20, min_samples_split=5, min_samples_leaf=1, bootstrap=True, random_state=10)
    model_grid = GridSearchCV(model_random, parametros, cv = 5, scoring='accuracy')
    model_grid.fit(input_train, output_train) #Treinamento

    print(model_grid.best_estimator_)
    '''
    model = RandomForestClassifier(class_weight='balanced', 
                                criterion='entropy',
                                n_estimators=50, 
                                random_state=10)
    
    model.fit(input_train, output_train) #Treinamento
    
    joblib.dump(model, 'thyroid_disease_AI\models_file\RandomCV.sav')

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