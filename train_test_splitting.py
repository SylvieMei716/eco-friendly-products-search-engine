"""Combining the .csv annotation files"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

df_final = pd.DataFrame()
# assign directory
directory = './annotated_files'
 
# iterate over files in that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        temp = pd.read_csv(f)
        df_final = df_final.append(temp, ignore_index=True)
        
df_final.to_csv('all_queries.csv', index=False)

#Splitting data into train and test sets with a 70:30 ratio
data = pd.read_csv('./all_queries.csv')
train, test = train_test_split(data, test_size=0.3)
train.to_csv('./training_set.csv', index=False)
test.to_csv('./testing_set.csv', index=False)
