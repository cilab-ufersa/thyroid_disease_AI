import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd
from imblearn.over_sampling import SMOTE #para o balanceamento de dados

# Lendo os dados
dataset = pd.read_csv('thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\hypothyroid.csv')


selected_features = ['age', 'sex', 'on thyroxine', 'thyroid surgery', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'binaryClass']
dataset = dataset[selected_features]
#dataset.info()
#transformando os dados em dados categoricos 
dataset['sex'] = dataset['sex'].astype('category').cat.codes.values #1 para F e 2 para M
dataset['on thyroxine'] = dataset['on thyroxine'].astype('category').cat.codes.values #0 para F e 1 para T
dataset['thyroid surgery'] = dataset['thyroid surgery'].astype('category').cat.codes.values #0 para F
dataset['binaryClass'] = dataset['binaryClass'].astype('category').cat.codes.values #1 para P e 2 para N
#print(dataset)

#processo de limpeza do dataset(removendo linhas com dados faltantes)
#print(dataset[dataset == '?'].count())
dataset.drop(dataset[dataset['age'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['TSH'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['T3'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['TT4'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['T4U'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['FTI'] == '?'].index, inplace=True)
#print(dataset[dataset == '?'].count())

#verificando discrepancia entre a quantidade de pacientes doentes e nao doentes
#print(dataset['binaryClass'].value_counts())
#dividindo o dataset
output_label_dataset = dataset['binaryClass']
dataset = dataset.drop(['binaryClass'], axis=1) #dados que serão introduzidos no modelo
#balanceando os dados
sm = SMOTE(k_neighbors=5)

dataset_res, output_label = sm.fit_resample(dataset, output_label_dataset)
print(output_label.value_counts())
print(dataset)