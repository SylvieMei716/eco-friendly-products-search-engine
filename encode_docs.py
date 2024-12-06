"""
Encode dataset to embeddings using given biencoder model
Author: Pooja Thakur

"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
import jsonlines

#Enter the name of the bi-encoder model to use
model_name = ""

#Enter the name and location of file where Beauty_and_Fashon data is stored
data_file = ""

#Enter filename to save encoded data
filename_encoded_data = ""
#Enter filename to save row_to_doc list
filename_ordering_data = ""

biencoder_model = SentenceTransformer(model_name)

#Read file data
with jsonlines.open(data_file) as file:
    data = list(file.iter())

print('Processing data...')
row_to_doc = []
#Iterates through all product data and creates embeddings for a combined string of title and description
for i in range(len(data)):
    if i==0:
        combined_input = data[i]['title'] + " " + data[i]['description']
        embedding = biencoder_model.encode(combined_input)
        embedded_mat = np.zeros((len(data),len(embedding)))
        embedded_mat[i] = embedding
        row_to_doc.append(data[i]['docid'])
    else:
        combined_input = data[i]['title'] + " " + data[i]['description']
        embedding = biencoder_model.encode(combined_input)
        embedded_mat[i] = embedding
        row_to_doc.append(data[i]['docid'])

np.save(filename_encoded_data, embedded_mat)
np.save(filename_ordering_data, np.array(row_to_doc))
print('Data saved.')




