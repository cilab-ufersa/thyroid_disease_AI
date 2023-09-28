import csv
from collections import Counter
import concurrent.futures

# Função para contar ocorrências de palavras no arquivo CSV
def count_word_occurrences(file_path, word_list):
    word_counts = Counter()
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Para ignorar a linha cabeçalho
        next(reader)
        
        for row in reader:
            for cell in row[1:]:
                words = cell.split(',')
                word_counts.update(word.strip() for word in words)
    
    result_counts = {word: word_counts[word] for word in word_list}
    return result_counts

# Função para processar os arquivos em paralelo
def processar_arquivos(arquivos, word_list):
    resultados = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Mapeia a função count_word_occurrences para cada arquivo em paralelo
        future_to_file = {executor.submit(count_word_occurrences, arquivo, word_list): arquivo for arquivo in arquivos}
        for future in concurrent.futures.as_completed(future_to_file):
            arquivo = future_to_file[future]
            try:
                word_counts = future.result()
                resultados[arquivo] = word_counts
            except Exception as exc:
                print(f'Erro ao processar o arquivo {arquivo}: {exc}')
    return resultados

if __name__ == '__main__':
    # Exemplo de uso:
    arquivos = ['clustering.csv', 'RFE.csv', 'correlação.csv', 'geral.csv']
    target_words = ['age', 'sex', 'on thyroxine', 'query on thyroxine',
           'on antithyroid medication', 'sick', 'pregnant','thyroid surgery', 
           'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
           'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH', 
           'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U', 
           'FTI measured', 'FTI', 'referral source']

    resultados = processar_arquivos(arquivos, target_words)

    # Para ordene os word_counts em ordem decrescente por ocorrência
    for arquivo, word_counts in resultados.items():
        sorted_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
        print(f'Dados do arquivo {arquivo}:')
        for word, count in sorted_counts.items():
            print(f"'{word}' aparece {count} vezes, ")
        print('---')

'''
Resultados das 10 melhores features:
    - Clustering:
        ['TT4 measured', 'T4U measured', 'TT4', 'T3 measured', 'FTI', 'T3', 'pregnant', 'I131 treatment', 'psych', 'sick']
    - RFE:
        ['TSH', 'FTI', 'T3', 'TT4', 'age', 'T4U', 'sex', 'on thyroxine', 'referral source', 'query hypothyroid']
    - Correlação:
        ['THS measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U', 'FTI measured', 'FTI']
    - Geral:
        ['TT4', 'TT4 measured', 'T4U measured', 'T3 measured', 'FTI', 'T3', 'TSH', 'T4U', 'pregnant', 'I131 treatment'] 
'''