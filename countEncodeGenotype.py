# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:13:44 2024

@author: Haxby
"""

import pandas as pd
import re
 
dataframe = pd.read_excel(r'C:\Users\uqkmuroi\Desktop\Van3_results.xlsx')
dataframe
 
def list_split(row):
    str_list = row['Genotype']
    result = re.sub(r'[\[\] ]', '', str_list)
    return result.split(',')
   
 
dataframe['Genotype_list_split'] = dataframe.apply(list_split, axis=1)
dataframe['Genotype_list_split']
 
full_list = []
for row in dataframe['Genotype_list_split']:
    full_list = full_list + row
 
unique_genes = list(set(full_list))
 
def one_hot(row):
    output = [0] * len(unique_genes)
    for gene in row['Genotype_list_split']:
        output[unique_genes.index(gene)] += 1
    return output
 
dataframe['genotype_one_hot'] = dataframe.apply(one_hot, axis=1)
output = dataframe[['Genotype', 'genotype_one_hot']]
output.to_csv('one-hotted.csv')
 