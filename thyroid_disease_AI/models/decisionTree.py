import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.tree import DecisionTreeClassifier #Para criar o modelo de árvore de decisão
from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
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
        'criterion': ['gini', 'entropy'],          
        'splitter': ['best', 'random'],            
        'max_depth': [None, 10, 20, 30, 40],      
        'min_samples_split': [2, 5, 10],          
        'min_samples_leaf': [1, 2, 4],            
        'max_features': ['auto', 'sqrt', 'log2'], 
        'random_state': [42] 
    }  
    model = DecisionTreeClassifier()
    grid = GridSearchCV(estimator=model, param_grid=parametros, cv=5, scoring='accuracy')
    grid.fit(input_train, output_train)
    print(grid.best_params_)
    
    'criterion': 'entropy', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'random_state': 42, 'splitter': 'best'
    '''
                                                 
    #Criando o modelo de árvore de decisão
    model= DecisionTreeClassifier(
                                criterion='entropy', 
                                max_depth=None, 
                                max_features='sqrt', 
                                min_samples_leaf=1, 
                                min_samples_split=24, 
                                random_state=42, 
                                splitter='best'
                                  )
    
    model.fit(input_train, output_train) #Treinamento

    joblib.dump(model, 'thyroid_disease_AI\models_file\DecisionTree.sav')

    output_model_decision = model.predict_proba(input_test)[:, 1]

    auc = roc_auc_score(output_test, output_model_decision)

    print(f'AUC: {auc}')

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model, title='Matriz confusão')

    accuracy(output_test, output_model_decision) # Pontuação de acurácia
    
    precision(output_test, output_model_decision) # Pontuação de precisão

    recall(output_test, output_model_decision) # Pontuação de recall

    f1(output_test, output_model_decision) # Pontuação de F1
    
    roc(output_test, output_model_decision) # Plotando a curva ROC

    miss_classification(input_train, output_train['binaryClass'], input_test, output_test['binaryClass'], model) # Plotando a curva de erro



    #Plotando a curva de aprendizado
    # plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    # scoring='misclassification error')
    # plt.show()