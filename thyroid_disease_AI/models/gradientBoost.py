import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import GradientBoostingClassifier #Para criar o modelo de árvore de decisão
from sklearn.model_selection import GridSearchCV
import joblib
from utils.utils import * 

if __name__ == '__main__':

    # Carregando dataset
    dataset =  pd.read_csv('thyroid_disease_AI\datasets\hypothyroid\hypothyroid_dataset_clean.csv')
    output_label_dataset = dataset['binaryClass']
    dataset = dataset.drop(['binaryClass'], axis=1)

    # Balanceando os dados
    dataset_res, output_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)
    
    # Dividindo os dados em 80% para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset_res, output_label)
    '''
    parametros = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'subsample': [0.8, 0.9, 1.0],
        'random_state': [42]
    }'''

    #Criando o modelo
    model= GradientBoostingClassifier(n_estimators=50,
                                      max_depth=5,
                                      random_state=42,
                                      subsample=0.9,
                                      max_features='sqrt')
    model.fit(input_train, output_train) #Treinamento

    # grid = GridSearchCV(estimator=model, param_grid=parametros, cv=5)
    # print(grid.best_params_)
    
    joblib.dump(model, 'thyroid_disease_AI\models_file\GradientBoost.sav')

    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model, title='Matriz confusão')

    accuracy(output_test, output_model_decision) # Pontuação de acurácia
    
    precision(output_test, output_model_decision) # Pontuação de precisão

    recall(output_test, output_model_decision) # Pontuação de recall

    f1(output_test, output_model_decision) # Pontuação de F1
    
    roc(output_test, output_model_decision) # Plotando a curva ROC

    miss_classification(input_train, output_train, input_test, output_test, model) # Plotando a curva de erro
   