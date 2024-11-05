"""
This file is to get most relevant 50-200 items to each query by 
BM25 (and annotate them afterwards).

Date: Nov 2, 2024
"""
import gzip
import json
import os
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import *
import pickle
import pandas as pd


DATA_PATH = 'data/'  # TODO: Set this to the path to your data folder
CACHE_PATH = '__pycache__/'  # Set this to the path of the cache folder

BEAUTY_PATH = DATA_PATH + 'meta_All_Beauty.jsonl.gz'
FASHION_PATH = DATA_PATH + 'meta_Amazon_Fashion.jsonl.gz'
COMBINE_PATH = DATA_PATH + 'Beauty_and_Fashion.jsonl.gz'
STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
MAIN_INDEX = 'main_index'
TITLE_INDEX = 'title_index'
N_DOC_NEEDED = 50
DOCID_TO_TITLE_PATH = CACHE_PATH + 'docid_to_title.pkl'
DOCID_TO_LINK_PATH = CACHE_PATH + 'docid_to_link.pkl'

# Load stopwords
stopwords = set()
with open(STOPWORD_PATH, 'r') as f:
    for line in f:
        stopwords.add(line.strip())
        
print('Loaded', len(stopwords), 'stopwords.')

# Load two categories' items into one dataset (running time: 30s)
item_cnt = 0
keys_to_keep = ["main_category", "title", "average_rating", "rating_number", "price", "images", "details"]

def process_dataset(input_path, output_file, item_cnt):
    with gzip.open(input_path, 'rt') as infile:
        for line in infile:
            data = json.loads(line)
            if data['description'] == [] and data['features'] == []:
                continue
            item_cnt += 1
            filtered_data = {key:data[key] for key in keys_to_keep if key in data}
            filtered_data['docid'] = item_cnt
            filtered_data['description'] = " ".join(data['features'] + data['description'])
            filtered_data['link'] = "https://www.amazon.com/dp/" + data['parent_asin']
            output_file.write(json.dumps(filtered_data) + '\n')
            
    return item_cnt

with gzip.open(COMBINE_PATH, 'wt') as outfile:
    item_cnt = process_dataset(BEAUTY_PATH, outfile, item_cnt)
    N_BEAUTY = item_cnt
    item_cnt = process_dataset(FASHION_PATH, outfile, item_cnt)
    N_FASHION = item_cnt - N_BEAUTY
    
print(f'Added {item_cnt} items in total to {COMBINE_PATH} from both Beauty and Fashion.')

# Indexing docs and titles (running time: main_index 105min, title_index 320min)
print('Loading indexes...')
preprocessor = RegexTokenizer('\w+')

if not os.path.exists(MAIN_INDEX):
    main_index = Indexer.create_index(
        IndexType.BasicInvertedIndex, COMBINE_PATH, preprocessor,
        stopwords, 3, text_key='description', max_docs=493293
    )
    main_index.save(MAIN_INDEX)
else:
    main_index = BasicInvertedIndex()
    main_index.load(MAIN_INDEX)

if not os.path.exists(TITLE_INDEX):
    title_index = Indexer.create_index(
        IndexType.BasicInvertedIndex, COMBINE_PATH, preprocessor, 
        stopwords, 2, max_docs=-1,
        text_key='title'
    )
    title_index.save(TITLE_INDEX)
else:
    title_index = BasicInvertedIndex()
    title_index.load(TITLE_INDEX)

# Store docid-title mapping (running time: 3s)
if not os.path.exists(DOCID_TO_TITLE_PATH):
    docid_to_title = {}
    with gzip.open(COMBINE_PATH, mode = 'rt', newline = '') as f:
        for line in f:
            data = json.loads(line)
            docid_to_title[data['docid']] = data['title']
    pickle.dump(docid_to_title,
                open(DOCID_TO_TITLE_PATH, 'wb')
    )
else:
    docid_to_title = pickle.load(open(DOCID_TO_TITLE_PATH, 'rb'))

# Store docid-link mapping (running time: 3s)
if not os.path.exists(DOCID_TO_LINK_PATH):
    docid_to_link = {}
    with gzip.open(COMBINE_PATH, mode = 'rt', newline = '') as f:
        for line in f:
            data = json.loads(line)
            docid_to_link[data['docid']] = data['link']
    pickle.dump(docid_to_link,
                open(DOCID_TO_LINK_PATH, 'wb')
    )
else:
    docid_to_link = pickle.load(open(DOCID_TO_LINK_PATH, 'rb'))

# BM25 ranking to get top 50 docs
ranker = Ranker(main_index, preprocessor, stopwords, BM25(main_index))

beauty_queries = ["Hydrating face serum",
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
fashion_queries = ["Maxi dress",
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

for beauty_query in beauty_queries:
    doc_lst = ranker.query(beauty_query)[:N_DOC_NEEDED]
    df = pd.DataFrame(columns=['query','title','docid','link','rel'])
    for i in range(len(doc_lst)):
        df.loc[i] = [beauty_query, docid_to_title[doc_lst[i][0]], doc_lst[i][0], docid_to_link[doc_lst[i][0]], None]
    df.to_csv(beauty_query+'.csv', index=False)

for fashion_query in fashion_queries:
    doc_lst = ranker.query(fashion_query)[:N_DOC_NEEDED]
    df = pd.DataFrame(columns=['query','title','docid','link','rel'])
    for i in range(len(doc_lst)):
        df.loc[i] = [fashion_query, docid_to_title[doc_lst[i][0]], doc_lst[i][0], docid_to_link[doc_lst[i][0]], None]
    df.to_csv(fashion_query+'.csv', index=False)