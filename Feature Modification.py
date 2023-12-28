import pandas as pd


data = pd.read_csv('Dataset/iris.csv')

label_column_name = data.columns[-1]

label_mapping = {label: idx for idx, label in enumerate(data[label_column_name].unique())}

data[label_column_name] = data[label_column_name].map(label_mapping)

output_file_path = 'Dataset/modified_iris.csv'
data.to_csv(output_file_path, index=False)
