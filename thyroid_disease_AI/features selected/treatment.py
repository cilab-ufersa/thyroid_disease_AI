import sys
sys.path.append('thyroid_disease_AI')
import pandas as pd
from imblearn.over_sampling import SMOTE #para o balanceamento de dados
from utils import *

# Lendo os dados
dataset = pd.read_csv('C:\\Users\\caiom\\Desktop\\Sist Hypo\\thyroid_disease_AI\\thyroid_disease_AI\\datasets\\hypothyroid\\hypothyroid.csv')

# Selecionando features finais
selected_features = ['TT4', 'TT4 measured', 'T4U measured', 'T3 measured', 'FTI', 'T3', 'TSH', 'T4U', 'pregnant', 'I131 treatment', 'binaryClass']
dataset = dataset[selected_features]
dataset = dataset.drop_duplicates()
# dataset.info()

#transformando os dados categoricos em numericos
dataset['TT4 measured'] = dataset['TT4 measured'].astype('category').cat.codes.values
dataset['T4U measured'] = dataset['T4U measured'].astype('category').cat.codes.values
dataset['T3 measured'] = dataset['T3 measured'].astype('category').cat.codes.values
dataset['pregnant'] = dataset['pregnant'].astype('category').cat.codes.values
dataset['I131 treatment'] = dataset['I131 treatment'].astype('category').cat.codes.values
dataset['binaryClass'] = dataset['binaryClass'].astype('category').cat.codes.values
# print(dataset)

# #processo de limpeza do dataset(removendo linhas com dados faltantes)
mean_TT4_1 = dataset['TT4'].loc[(dataset['TT4'] != '?') & (dataset['binaryClass'] ==1)].astype('float').mean()
mean_TT4_0 = dataset['TT4'].loc[(dataset['TT4'] != '?') & (dataset['binaryClass'] ==0)].astype('float').mean()
mean_FTI_1 = dataset['FTI'].loc[(dataset['FTI'] != '?') & (dataset['binaryClass'] ==1)].astype('float').mean()
mean_FTI_0 = dataset['FTI'].loc[(dataset['FTI'] != '?') & (dataset['binaryClass'] ==0)].astype('float').mean()
mean_T3_1 = dataset['T3'].loc[(dataset['T3'] != '?') & (dataset['binaryClass'] ==1)].astype('float').mean()
mean_T3_0 = dataset['T3'].loc[(dataset['T3'] != '?') & (dataset['binaryClass'] ==0)].astype('float').mean()
mean_TSH_1 = dataset['TSH'].loc[(dataset['TSH'] != '?') & (dataset['binaryClass'] ==1)].astype('float').mean()
mean_TSH_0 = dataset['TSH'].loc[(dataset['TSH'] != '?') & (dataset['binaryClass'] ==0)].astype('float').mean()
mean_T4U_1 = dataset['T4U'].loc[(dataset['T4U'] != '?') & (dataset['binaryClass'] ==1)].astype('float').mean()
mean_T4U_0 = dataset['T4U'].loc[(dataset['T4U'] != '?') & (dataset['binaryClass'] ==0)].astype('float').mean()

dataset.loc[(dataset['TT4']== '?') & (dataset['binaryClass'] ==0), 'TT4'] = mean_TT4_0
dataset.loc[(dataset['TT4']== '?') & (dataset['binaryClass'] ==1), 'TT4'] = mean_TT4_1
dataset.loc[(dataset['FTI']== '?') & (dataset['binaryClass'] ==0), 'FTI'] = mean_FTI_0
dataset.loc[(dataset['FTI']== '?') & (dataset['binaryClass'] ==1), 'FTI'] = mean_FTI_1
dataset.loc[(dataset['T3']== '?') & (dataset['binaryClass'] ==0), 'T3'] = mean_T3_0
dataset.loc[(dataset['T3']== '?') & (dataset['binaryClass'] ==1), 'T3'] = mean_T3_1
dataset.loc[(dataset['TSH']== '?') & (dataset['binaryClass'] ==0), 'TSH'] = mean_TSH_0
dataset.loc[(dataset['TSH']== '?') & (dataset['binaryClass'] ==1), 'TSH'] = mean_TSH_1
dataset.loc[(dataset['T4U']== '?') & (dataset['binaryClass'] ==0), 'T4U'] = mean_T4U_0
dataset.loc[(dataset['T4U']== '?') & (dataset['binaryClass'] ==1), 'T4U'] = mean_T4U_1
# print(dataset[dataset == '?'].count())


# verificando discrepancia entre a quantidade de pacientes doentes e nao doentes
# print(dataset['binaryClass'].value_counts())

#dividindo o dataset
output_label_dataset = dataset['binaryClass']
dataset = dataset.drop(['binaryClass'], axis=1) #dados que serão introduzidos no modelo
#balanceando os dados
sm = SMOTE(k_neighbors=5, random_state=42)
   
dataset_res, output_label = sm.fit_resample(dataset, output_label_dataset)
input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset_res, output_label)

dataset_test = pd.DataFrame(input_test, columns=dataset.columns)
dataset_train = pd.DataFrame(input_train, columns=dataset.columns)

#dataset_test['binaryClass'] = output_test
#dataset_train['binaryClass'] = output_train

output_test = pd.DataFrame(output_test, columns=['binaryClass'])
output_train = pd.DataFrame(output_train, columns=['binaryClass'])

hypothyroid_dataset_test = 'thyroid_disease_AI\datasets\hypothyroid\hypothyroid_input_test.csv'
dataset_test.to_csv(hypothyroid_dataset_test, index=False)
output_test.to_csv('thyroid_disease_AI\datasets\hypothyroid\hypothyroid_output_test.csv', index=False)

hypothyroid_dataset_train = 'thyroid_disease_AI\datasets\hypothyroid\hypothyroid_input_train.csv'
dataset_train.to_csv(hypothyroid_dataset_train, index=False)
output_train.to_csv('thyroid_disease_AI\datasets\hypothyroid\hypothyroid_output_train.csv', index=False)


