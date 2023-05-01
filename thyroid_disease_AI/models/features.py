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
dataset['sex'] = dataset['sex'].astype('category').cat.codes.values
dataset['on thyroxine'] = dataset['on thyroxine'].astype('category').cat.codes.values
dataset['query on thyroxine'] = dataset['query on thyroxine'].astype('category').cat.codes.values
dataset['on antithyroid medication'] = dataset['on antithyroid medication'].astype('category').cat.codes.values
dataset['sick'] = dataset['sick'].astype('category').cat.codes.values 
dataset['pregnant'] = dataset['pregnant'].astype('category').cat.codes.values
dataset['thyroid surgery'] = dataset['thyroid surgery'].astype('category').cat.codes.values
dataset['I131 treatment'] = dataset['I131 treatment'].astype('category').cat.codes.values
dataset['query hypothyroid'] = dataset['query hypothyroid'].astype('category').cat.codes.values
dataset['query hyperthyroid'] = dataset['query hyperthyroid'].astype('category').cat.codes.values
dataset['lithium'] = dataset['lithium'].astype('category').cat.codes.values
dataset['goitre'] = dataset['goitre'].astype('category').cat.codes.values
dataset['tumor'] = dataset['tumor'].astype('category').cat.codes.values
dataset['hypopituitary'] = dataset['hypopituitary'].astype('category').cat.codes.values
dataset['psych'] = dataset['psych'].astype('category').cat.codes.values
dataset['TSH measured'] = dataset['TSH measured'].astype('category').cat.codes.values
dataset['T3 measured'] = dataset['T3 measured'].astype('category').cat.codes.values
dataset['TT4 measured'] = dataset['TT4 measured'].astype('category').cat.codes.values
dataset['T4U measured'] = dataset['T4U measured'].astype('category').cat.codes.values
dataset['FTI measured'] = dataset['FTI measured'].astype('category').cat.codes.values
dataset['TBG measured'] = dataset['TBG measured'].astype('category').cat.codes.values
dataset['referral source'] = dataset['referral source'].astype('category').cat.codes.values
dataset['binaryClass'] = dataset['binaryClass'].astype('category').cat.codes.values
# print(dataset)

#processo de limpeza do dataset(removendo linhas com dados faltantes)
#print(dataset[dataset == '?'].count())
dataset.drop(dataset[dataset['age'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['TSH'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['T3'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['TT4'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['T4U'] == '?'].index, inplace=True)
dataset.drop(dataset[dataset['FTI'] == '?'].index, inplace=True)
dataset['TBG'].replace('?', -1, inplace=True)
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