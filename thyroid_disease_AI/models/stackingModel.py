import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd #Para trabalhar com dataframes               
import numpy as np #Para trabalhar com arrays
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.tree import DecisionTreeClassifier #Para criar o modelo de árvore de decisão
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from utils import * 
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

    estimators = [ 
        ('xgb', XGBClassifier(
            colsample_bytree = 0.1,
            gamma = 0.5,
            learning_rate = 0.04,
            max_depth = 9,
            min_child_weight = 5,
            n_estimators = 200,
            subsample = 0.1)), 
        ('random', RandomForestClassifier(
            n_estimators=12,
            max_depth=3,
            max_features ='log2',
            min_samples_split = 2,
            min_samples_leaf = 1,)),
        ('lg', lgb.LGBMClassifier(
            learning_rate = 3.000001,
            max_depth = 5,
            n_estimators = 100,
            num_leaves = 15, 
            subsample = 0.8,)),
        ('GDBoost', GradientBoostingClassifier(
            n_estimators = 20,
            max_depth = 5,
            learning_rate = 0.02,
            subsample = 0.09,
            max_features = 'sqrt',)),
        ('extraTrees', ExtraTreesClassifier(
            max_depth=30,
            min_samples_leaf=4,
            min_samples_split=4,
            n_estimators=50,)),
        ('decision', DecisionTreeClassifier(
            criterion='entropy', 
            max_depth=None, 
            max_features='sqrt', 
            min_samples_leaf=1, 
            min_samples_split=24, 
            random_state=42, 
            splitter='best'))
    ]
    model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(
            n_estimators=12,
            max_depth=3,
            max_features ='log2',
            min_samples_split = 2,
            min_samples_leaf = 1,)
    )
    model.fit(input_train, output_train) 

    # joblib.dump(model, 'thyroid_disease_AI\models_file\StackingModel.sav')

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