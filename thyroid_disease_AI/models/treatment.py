import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd
from imblearn.over_sampling import SMOTE #para o balanceamento de dados

# Lendo os dados
dataset = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\hypothyroid.csv')

# Selecionando features finais
selected_features = ['TT4', 'TT4 measured', 'T4U measured', 'T3 measured', 'FTI', 'T3', 'TSH', 'T4U', 'pregnant', 'I131 treatment', 'binaryClass']
dataset = dataset[selected_features]
# dataset.info()

#transformando os dados categoricos em numericos
dataset['TT4 measured'] = dataset['TT4 measured'].astype('category').cat.codes.values
dataset['T4U measured'] = dataset['T4U measured'].astype('category').cat.codes.values
dataset['T3 measured'] = dataset['T3 measured'].astype('category').cat.codes.values
dataset['pregnant'] = dataset['pregnant'].astype('category').cat.codes.values
dataset['I131 treatment'] = dataset['I131 treatment'].astype('category').cat.codes.values
dataset['binaryClass'] = dataset['binaryClass'].astype('category').cat.codes.values
# print(dataset)

#processo de limpeza do dataset(removendo linhas com dados faltantes)
dataset.drop(dataset[dataset['TT4'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['FTI'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['T3'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['TSH'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['T4U'] == '?'].index, inplace=True)
# print(dataset[dataset == '?'].count())

#verificando discrepancia entre a quantidade de pacientes doentes e nao doentes
# print(dataset['binaryClass'].value_counts())

#dividindo o dataset
output_label_dataset = dataset['binaryClass']
dataset = dataset.drop(['binaryClass'], axis=1) #dados que serão introduzidos no modelo
#balanceando os dados
sm = SMOTE(k_neighbors=5)

dataset_res, output_label = sm.fit_resample(dataset, output_label_dataset)
# print(output_label.value_counts())
# print(dataset)

dataset_balanced = pd.DataFrame(dataset_res, columns=dataset.columns)
dataset_balanced['binaryClass'] = output_label

hypothyroid_features_final = 'thyroid_disease_AI\datasets\hypothyroid\hypothyroid_features_final.csv'
dataset_balanced.to_csv(hypothyroid_features_final, index=False)

print('Novo dataset criado ', hypothyroid_features_final)