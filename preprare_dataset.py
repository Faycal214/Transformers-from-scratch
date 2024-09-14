from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
ds = load_dataset("PaulineSanchez/Translation_words_and_sentences_english_french", split= 'train')

# Save the dataset to a CSV file
csv_file_path = "french_english.csv"
ds.to_csv(csv_file_path)

print(f"Dataset saved to {csv_file_path}")

# open dataset
df = pd.read_csv('french_english.csv')
input = df['English words/sentences']
target = df['French words/sentences']

# spliting data
input_train, input_val, tar_train, tar_val = train_test_split(input, target, test_size= 0.3, shuffle= True)

# transform the featurs into lists
input_train = input_train.to_list()
input_val = input_val.to_list()
target_train = tar_train.to_list()
target_val = tar_val.to_list()

# exemple
# print(f"English : {input_train[0]} ==> French : {output_train[0]}")
# print(f"English : {input_val[0]} ==> French : {output_val[0]}")

