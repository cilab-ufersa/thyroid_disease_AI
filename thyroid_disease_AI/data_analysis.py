import pandas as pd
import matplotlib.pyplot as plt

input_train = pd.read_csv('/workspaces/thyroid_disease_AI/thyroid_disease_AI/datasets/hypothyroid/input_train.csv')
output_train = pd.read_csv('/workspaces/thyroid_disease_AI/thyroid_disease_AI/datasets/hypothyroid/output_train.csv')

# analysis the distribution of the output_train
print(output_train['binaryClass'].value_counts())

#scatter plot of output_train with smote 
plt.scatter(input_train['TSH'], output_train['binaryClass'])
plt.show()


