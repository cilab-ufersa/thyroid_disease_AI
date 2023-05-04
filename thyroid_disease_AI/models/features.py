import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd
from imblearn.over_sampling import SMOTE #para o balanceamento de dados
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Lendo os dados
dataset = pd.read_csv('thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\hypothyroid.csv')

#print(dataset.info())
#transformando os dados em dados categoricos 
for index in dataset.columns.values:
        dataset[index]= dataset[index].astype("category").cat.codes.values

#processo de limpeza do dataset(removendo linhas com dados faltantes)
for i in dataset.columns.values:
    dataset.drop(dataset[dataset[i] == '?'].index, inplace=True)
dataset = dataset.drop('TBG', axis=1)
#print(dataset[dataset == '?'].count())
#print(dataset)

#verificando discrepancia entre a quantidade de pacientes doentes e nao doentes
#print(dataset['binaryClass'].value_counts())
#dividindo o dataset
output_label_dataset = dataset['binaryClass']
dataset = dataset.drop(['binaryClass'], axis=1) #dados que serão introduzidos no modelo
#balanceando os dados
sm = SMOTE(k_neighbors=5)

dataset_res, output_label = sm.fit_resample(dataset, output_label_dataset)
#print(output_label.value_counts())
#print(dataset)

#Treinando o modelo com todas as features
model = RandomForestClassifier()
model.fit(dataset, output_label_dataset)

#Selecionando as variáveis mais importantes
selector = RFE(model, n_features_to_select=10, step=1) #Criando um objeto RFE (Recursive Feature Elimination) com o modelo e os parâmetros desejados
selector = selector.fit(dataset, output_label_dataset) #Ajustando o seletor de recursos ao conjunto de dados e aos rótulos de saída
selected_features = dataset.columns[selector.support_] #Selecionando as características escolhidas pelo seletor de recursos e armazenando seus nomes em uma lista

print(selected_features)

#resultado = dataset['age', 'sex', 'on thyroxine', 'thyroid surgery', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'referral source']