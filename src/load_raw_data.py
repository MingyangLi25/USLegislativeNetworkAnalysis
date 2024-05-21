from datasets import load_dataset
import pandas as pd
import os

# Load dataset from Huggingface
dataset_committees = load_dataset("dreamproit/bill_committees_us")
dataset_labels = load_dataset("dreamproit/bill_labels_us")

# Keep only id and commttees
df_committees = dataset_committees['train'].to_pandas()
df_committees = df_committees[['id', 'committees']]

# Drop text columns
df_labels = dataset_labels['train'].to_pandas()
df_labels = df_labels.drop(columns=['text', 'sections', 'sections_length'])

# Outer join two dataframes
merged_df = pd.merge(df_committees, df_labels, on='id', how='outer')

# Save data into CSV file
csv_file_path = os.path.join('..', 'data', 'raw', 'raw_dataset.csv')
merged_df.to_csv(csv_file_path, index=False)
print(f"Modified dataset saved as {csv_file_path}")