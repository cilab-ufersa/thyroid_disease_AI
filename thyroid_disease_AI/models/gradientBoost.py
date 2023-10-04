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
    #80 % para treino e 20% para teste
    input_train = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\input_train.csv')
    input_train = input_train.values
    input_test = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\input_test.csv')
    input_test = input_test.values
    output_train = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\output_train.csv')
    output_test = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\output_test.csv')
    
    '''
    parametros = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5]
                            }
    model = GradientBoostingClassifier()
    grid = GridSearchCV(estimator=model, param_grid=parametros, cv=5, scoring='accuracy')
    grid.fit(input_train, output_train.values.ravel())
    print(grid.best_params_)

    {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
    '''

    # Criando o modelo
    model= GradientBoostingClassifier(n_estimators = 20,
                                      max_depth = 5,
                                      learning_rate = 0.02,
                                      subsample = 0.09,
                                      max_features = 'sqrt',
                                      )
    model.fit(input_train, output_train) # Treinamento

    joblib.dump(model, 'thyroid_disease_AI\models_file\GradientBoost.sav')

    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model, title='Matriz confusão')

    accuracy(output_test, output_model_decision) # Pontuação de acurácia
    
    precision(output_test, output_model_decision) # Pontuação de precisão

    recall(output_test, output_model_decision) # Pontuação de recall

    f1(output_test, output_model_decision) # Pontuação de F1
    
    roc(output_test, output_model_decision) # Plotando a curva ROC

    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model) # Plotando a curva de erro
   