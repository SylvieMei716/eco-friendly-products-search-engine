"""
Calculate cosine similarity between keyword embeddings and document embeddings
Author: Pooja Thakur

"""

import numpy as np
from sentence_transformers import SentenceTransformer
from indexing import Indexer, InvertedIndex, BasicInvertedIndex

class Similarity:
    def __init__(self, index: InvertedIndex, docid, keyword_embed: np.ndarray, embedded_data: np.ndarray, 
                 row_to_doc: list, all_keywords: bool):
        """This class has a few similarity measuring functions that aim to calculate the similarity
        between the given set of keywords and documents
        
        Args:
        index: Index of the product data
        keyword_embed: keyword embeddings being used
        model_name: Name of bi-encoder model
        embedded_data: Doc-embeddings of product data
        row_to_doc: Mappings from row of embedded_data to docid """

        self.index = index
        self.docid = docid
        self.embedded_data = embedded_data
        self.all_keywords = all_keywords
        
        #Mapping out row_to_doc as a dictionary to reduce computation time
        self.row_mappings = {}
        for i in range(len(row_to_doc)):
            self.row_mappings[row_to_doc[i]] = i

        #self.keyword_embed = self.biencoder_model.encode(keywords)
        self.doc_embed = embedded_data[self.row_mappings[self.docid]]
        self.embeddings = keyword_embed

    def Cosine_similarity(self):
        """Calculates cosine similarity between keywords and the document embeddings and returns vector/matrix
        of the values.
        For testing whether it is better to use the entire lot of keywords at a time or use them word by word,
        this function has the requirements for both these configurations."""

        if self.all_keywords:
            #Using all keywords at once
            sim = np.dot(self.keyword_embed, self.doc_embed)
        else:
            #Using one keyword at a time
            sim = []
            for e in self.embeddings:
                sim.append(np.dot(e,self.doc_embed))
        return sim
    
