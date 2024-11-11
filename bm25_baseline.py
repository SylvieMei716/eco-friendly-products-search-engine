from indexing import BasicInvertedIndex, IndexType, InvertedIndex, PositionalInvertedIndex
from collections import Counter
import gzip
import json
import jsonlines
from document_preprocessor import RegexTokenizer
from ranker import BM25, Ranker
import os
import pandas as pd
from relevance import run_relevance_tests

"""Combining the .csv annotation files"""
df_final = pd.DataFrame()
# assign directory with all the annotated files
directory = './annotated_files'

# Parsing through each file and appending to data frame 
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        print(f)
        temp = pd.read_csv(f)
        df_final = df_final.append(temp, ignore_index=True)
        
df_final.to_csv('all_queries.csv', index=False)

"""Creating a separate index for eco-friedly products"""
class eco_friendly_indexer:

    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor, stopwords: set[str],
                     minimum_word_frequency: int, keywords: set[str],
                     max_docs: int = None) -> InvertedIndex:
        def filter_stopwords(tokens, stopwords):
            return [token for token in tokens if token not in stopwords]

        def filter_by_frequency(tokens, min_freq, u_tokens):
            token_counts = Counter(tokens)
            return [token for token in u_tokens if token_counts[token] >= min_freq]
        
        if index_type.name == 'BasicInvertedIndex':
            index = BasicInvertedIndex()
        elif index_type.name == 'PositionalIndex':
            index = PositionalInvertedIndex()

        if '.gz' in dataset_path:
            with gzip.open(dataset_path,'rt') as file:
                for line in file:
                    data = json.loads(line)
        else:
            with jsonlines.open(dataset_path) as file:
                data = list(file.iter())

        tokenizer = document_preprocessor
        all_tokens = []

        if not max_docs:
            max_docs = len(data)

        for i in range(max_docs):
            tokens = tokenizer.tokenize(data[i]['description'])
            #tokens.extend(tokenizer.tokenize(data[i]['details']['Material Feature']))
            if keywords in tokens:
                index.add_doc(data[i]["docid"], tokens)
                all_tokens.extend(tokens)

        unique_tokens = index.get_u_tokens()

        if stopwords:
            unique_tokens = filter_stopwords(unique_tokens, stopwords)

        if minimum_word_frequency > 0:
            unique_tokens = filter_by_frequency(all_tokens, minimum_word_frequency, unique_tokens)

        index.statistics['unique_tokens'] = list(unique_tokens)

        return index
    
#Make eco friendly index
keywords_string = "sustainable, organic, biodegradable, recyclable, compostable, recycled, toxic, renewable, vegan, cruelty, FSC-certified, carbon, Fair Trade, climate, upcycled, responsibly sourced, pesticide, ethical, toxin, eco"
tokenizer = RegexTokenizer('\w+')
keywords = tokenizer.tokenize(keywords_string)

path = './Beauty_and_Fashion.jsonl' #path to the combined beauty and fashion file
stopwords_file = './stopwords.txt'
stopwords = set()
with open(stopwords_file, 'r') as f:
    for line in f:
        stopwords.add(line.strip())

eco_index = eco_friendly_indexer.create_index(IndexType.BasicInvertedIndex, path, RegexTokenizer('\w+'), stopwords, 3, set(keywords), max_docs=493293)

beauty_queries = ["Matte pink lipstick",
"Clear lip gloss",
"Hydrating lip oil",
"Pastel color eyeshadow palette",
"Bronzer powder",
"Eyebrow gel",
"Eyelash curler",
"White waterline eyeliner",
"Blush brush",
"Makeup blender sponge",
"Hydrating face serum",
"Organic lip balm",
"Sunscreen spf 50",
"Matte foundation",
"Hair repair oil",
"Anti-aging night cream for sensitive skin",
"Cruelty-free makeup set",
"Gentle facial cleanser with natural ingredients",
"Long-lasting waterproof mascara",
"Shampoo and conditioner set for curly hair",
]

fashion_queries = ["Leg warmers",
"Frilly shorts",
"Cardigan sweaters for women",
"High waisted jeans",
"Knee high boots women",
"Platform heels",
"Pink purse crossbody",
"Leather jacket men",
"Ripped jeans men",
"T-shirt dress women",
"Maxi dress",
"Crop top",
"V-neck t-shirt",
"Gray baggy jeans",
"Wool scarf",
"Running shoes with cushions",
"Lightweight travel backpack",
"High-waisted leggings with pockets",
"Casual blazer for men in slim fit style",
"Kids’ winter coat waterproof",
]

eco_ranker = Ranker(eco_index, tokenizer, stopwords, BM25(eco_index))

#Parse through all queries using ranker to check if any query has less than 10 documents (this is
#needed for the scorer, so just making sure)
for beauty_query in beauty_queries:
    doc_lst = eco_ranker.query(beauty_query)
    if len(doc_lst)<10: print('This query does not have enough docs', beauty_query)
        
for beauty_query in fashion_queries:
    doc_lst = eco_ranker.query(beauty_query)
    if len(doc_lst)<10: print('This query does not have enough docs', beauty_query)

#Return the MAP@10 and NDCG@10 scores now
filename = './all_queries.csv'
d = run_relevance_tests(filename, eco_ranker)
print(d)