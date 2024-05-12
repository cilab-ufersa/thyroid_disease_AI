import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn import metrics 
from mlxtend.plotting import plot_learning_curves 

def prepare_dataset(path_origin, columns_labels, path_destiny):
    """prepare a dataset

    Args:
        path_origin (string): origin path
        columns_labels (list): columns label
        path_destiny (string): path destiny
    """
    dataframe = pd.read_csv(path_origin)
    dataframe.columns=columns_labels
    dataframe.to_csv(path_destiny, index=False)


def plot_confusion_matrix(output_test, output_model_decision, model, title, labelx='Prediction', labely='Real', display_labels=['Normal', 'Sick']):
    """plot confusion matrix

    Args:

        output_test (list): dataset for test
        output_model_decision (list): dataset for train
        model (string): model name
        title (string): title
    Returns:
        ConfusionMatrixDisplay: confusion matrix
    """

    confusionmatrix = confusion_matrix(output_test, output_model_decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(3,3))
    disp.plot(ax=ax, colorbar=False, cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    disp.ax_.set_xlabel(labelx)
    disp.ax_.set_ylabel(labely)
    return disp


def accuracy(output_test, output_model_decision):
    """show accuracy

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("\nA acurácia é de: ", accuracy_score(output_test, output_model_decision))



def precision(output_test, output_model_decision):
    """show precision

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A precisão é de: ", precision_score(output_test, output_model_decision))



def recall(output_test, output_model_decision):
    """show recall

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A pontuação de recall é de: ", recall_score(output_test, output_model_decision))


def f1(output_test, output_model_decision):
    """show F1

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A pontuação de F1 é de: ", f1_score(output_test, output_model_decision)) #Pontuação do F1



def roc(output_test, output_model, title = "Curva ROC"):
    """ploting ROC curve

    Args:
        output_test (list): dataset for test
        output_model (list): dataset for train
    """
    fp, tp, _ = metrics.roc_curve(output_test, output_model)
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(fp, tp, label = "ROC", linewidth = 2, linestyle = "--")
    plt.legend()
    plt.title(title)
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.grid(True)
    fig = plt.gcf()
    return fig


def miss_classification(input_train, output_train, input_test, output_test, model, title='Curva de erro', labelx='Training Set', labely='Classification error', label1='Training error', label2='Test error'):
    """ Plot miss classification error

    Args:
        input_train (array): input train
        output_train (array): output train
        input_test (array): input test
        output_test (array): output test
        model (object): model
    Return:
    """
    training_errors, test_errors = plot_learning_curves(X_train=input_train, y_train=output_train, X_test=input_test, y_test=output_test, clf=model, print_model=False)
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(np.arange(10, 101, 10), training_errors, label=label1, linewidth=2, linestyle='--')
    plt.plot(np.arange(10, 101, 10), test_errors, label=label2, linewidth=2)
    plt.legend()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.grid(True)
    fig = plt.gcf()
    plt.show()
    return fig

def learning_curves(input_train, output_train, input_test, output_test, model, title = 'Curva de aprendizado',  labelx='Conjunto de treinamento', labely='Acurácia'):
    """  Plot learning curves

    Args:

        input_train (array): input train
        output_train (array): output train
        input_test (array): input test
        output_test (array): output test
        model (object): model
    Return:
        fig: plot    
    """

    training_errors, test_errors = plot_learning_curves(X_train=input_train, y_train=output_train, X_test=input_test, y_test=output_test, clf=model, scoring='accuracy')
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(np.arange(10, 101, 10), training_errors, label='Treinamento', linewidth=2, linestyle='--')
    plt.plot(np.arange(10, 101, 10), test_errors, label='Teste', linewidth=2)
    plt.legend()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.grid(True)
    fig = plt.gcf()
    return fig

def balance_dataset_smote(dataset, output_label, random_state=42, k_neighbors=5):
    """balance dataset

    Args:
        dataset (pandas dataframe): dataset
        output_label (string): output label
        random_state (int, optional): random state. Defaults to 42.
        k_neighbors (int, optional): k neighbors. Defaults to 5.

    Returns:
        pandas dataframe: dataset balanced
    """
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    dataset_res, ouput_label = sm.fit_resample(dataset, output_label)

    return dataset_res, ouput_label

def slipt_and_standardize_dataset(dataset, output_label, test_size=0.2, random_state=23):
    """slipt and standardize dataset

    Args:
        dataset (pandas dataframe): dataset
        output_label (string): output label
        test_size (float, optional): test size. Defaults to 0.2.
        random_state (int, optional): random state. Defaults to 23.

    Returns:
        pandas dataframe: input train
        pandas dataframe: input test
        pandas dataframe: output train
        pandas dataframe: output test
    """
    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = train_test_split(dataset, output_label, test_size=test_size, random_state=random_state)

    # Padronizando os dados
    scaler = StandardScaler()
    input_train = scaler.fit_transform(input_train)
    input_test = scaler.transform(input_test)

    return input_train, input_test, output_train, output_test

def remove_outliers_rows(df, z_threshold=3, target_column=None):
    """
    Remove rows with outliers from a DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame.
    - z_threshold (float): Z-score threshold for outlier detection. Defaults to 3.
    - target_column (str): Name of the target label column to exclude from outlier removal. Defaults to None.

    Returns:
    - DataFrame: DataFrame with rows containing outliers removed.
    """
    df_copy = df.copy()  
    for column in df_copy.columns:
        if column == target_column[0] or column == target_column[1]:
            continue  # Skip outlier removal for the target class column
        # Convert column values to numeric, ignoring non-numeric values like '?'
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        z_scores = zscore(df_copy[column])
        outlier_indices = np.where(np.abs(z_scores) >= z_threshold)[0]
        df_copy = df_copy.drop(outlier_indices)
    
    return df_copy


def impute_missing_with_means(dataset, column, binary_class):
    """
    Impute missing values in a column with the mean of the corresponding binary class.

    Parameters:
    - dataset (DataFrame): Input DataFrame.
    - column (str): Column name.
    - binary_class (str): Binary class column name.

    Returns:
    - None
    """
    dataset[column] = dataset[column].replace('?', np.nan)  # Replace '?' with NaN
    mean_1 = dataset[column].loc[(~dataset[column].isnull()) & (dataset['binaryClass'] == 1)].astype(float).mean().round(2)
    mean_0 = dataset[column].loc[(~dataset[column].isnull()) & (dataset['binaryClass'] == 0)].astype(float).mean().round(2)
    dataset.loc[(dataset[column].isnull()) & (dataset['binaryClass'] == 0), column] = mean_0
    dataset.loc[(dataset[column].isnull()) & (dataset['binaryClass'] == 1), column] = mean_1    
