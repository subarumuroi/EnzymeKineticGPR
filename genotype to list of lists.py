# -*- coding: utf-8 -*-
"""
@author: Subaru
"""

file = r"C:\Users\uqkmuroi\Desktop\Van3_results.xlsx"

import numpy as np
import pandas as pd
import tensorflow as tf

import ast
import re

np.random.seed(42)
tf.random.set_seed(42)

#this bit maybe unnecessary
'''
def LoadData(csvfile):
        df=pd.read_csv(csvfile)
        df_inputs = df.iloc[1:426, 115:116]
        
        return df_inputs
    
X_train = LoadData(file)
'''

df = pd.read_excel(r'C:\Users\uqkmuroi\Desktop\Van3_results.xlsx')

# Access the specific column
column_data = df['Genotype']

# Convert to a list if needed
data_list = column_data.tolist()




def convert_to_list_of_lists(data_list):
    
    list_of_lists = []

    for item in data_list:
        # Remove brackets and split by comma
        cleaned_line = item.strip()[1:-1] if item.startswith('[') and item.endswith(']') else item  # Remove the brackets
        elements = [elem.strip() for elem in cleaned_line.split(',')]
        list_of_lists.append(elements)

    return list_of_lists

gtype_lists= convert_to_list_of_lists(data_list)

# Initialize an empty list to hold the individual categories
all_gtype = []

# Function to preprocess the entry
def preprocess_entry(entry):
    # Check if the entry looks like a list but is missing quotes
    if entry.startswith('[') and entry.endswith(']'):
        # Replace unquoted entries with quoted ones
        entry = re.sub(r'(\w+)', r'"\1"', entry)  # Add quotes around each word
    return entry
# Loop through each entry in the data
for entry in data_list:
    # Check if entry is a valid string and not empty
    if isinstance(entry, str) and entry.strip():  # Check if entry is a non-empty string
        if entry == 'REF':
            all_gtype.append("REF")
        else:
            entry = preprocess_entry(entry)  # Preprocess the entry
            try:
                # Convert the string representation of the list into an actual list
                categories = ast.literal_eval(entry)
                if isinstance(categories, list):  # Ensure it's a list
                    all_gtype.extend(categories)  # Add categories to the list
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing entry '{entry}': {e}")


# Count individual categories
gtype_counts = set(all_gtype)

#Initialize dictionary to hold counts
counts_dict = {genotype: 0 for genotype in gtype_counts}

#count occurences
for sublist in gtype_lists:
    for genotype in sublist:
        if genotype in counts_dict:
            counts_dict[genotype] += 1
            
# Convert counts dictionary to a list of lists 
counts_list = [[genotype,count] for genotype ]

