import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.tree import DecisionTreeClassifier #Para criar o modelo de árvore de decisão
from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import GridSearchCV
import joblib
from utils.utils import * 

if __name__ == '__main__':

    # Carregando dataset
    dataset =  pd.read_csv('thyroid_disease_AI\datasets\hypothyroid\hypothyroid_dataset_clean.csv')
    dataset = dataset.drop_duplicates()
    output_label_dataset = dataset['binaryClass']
    dataset = dataset.drop(['binaryClass'], axis=1)

    # Balanceando os dados
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)
    
    # Dividindo os dados em 80% para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
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
    '''
                                                 
    #Criando o modelo de árvore de decisão
    model= DecisionTreeClassifier(criterion='gini', max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, random_state=42, splitter='best')
    model.fit(input_train, output_train) #Treinamento

    # grid = GridSearchCV(estimator=model, param_grid=parametros, cv=5, scoring='accuracy')
    # print(grid.best_params_)
    
    joblib.dump(model, 'thyroid_disease_AI\models_file\DecisionTree.sav')

    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model, title='Matriz confusão')

    accuracy(output_test, output_model_decision) # Pontuação de acurácia
    
    precision(output_test, output_model_decision) # Pontuação de precisão

    recall(output_test, output_model_decision) # Pontuação de recall

    f1(output_test, output_model_decision) # Pontuação de F1
    
    roc(output_test, output_model_decision) # Plotando a curva ROC

    miss_classification(input_train, output_train, input_test, output_test, model) # Plotando a curva de erro



    #Plotando a curva de aprendizado
    # plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    # scoring='misclassification error')
    # plt.show()


    