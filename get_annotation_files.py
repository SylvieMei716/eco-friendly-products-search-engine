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
import requests
import time
import random


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
"Face lotion with sunscreen",
"Face sheet mask",
"Makeup remover wipes",
"Liquid eyeliner",
"Plumping lip gloss",
"Nail polish",
"Body spray",
"Setting spray",
"Deodorant for sensitive skin",
"Makeup brush set"
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
"Womens skinny jeans",
"Womens sweaters warm",
"Lightweight jacket for men",
"Hiking boots for women",
"Gloves for women",
"Kids t-shirts",
"Womens crop top",
"Cardigan sweater with pockets",
"Sundress",
"Swim trunks"
]

def check_amazon_item_exists(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    time.sleep(random.uniform(2, 5))
    response = requests.get(url, headers=headers)

    # Check if the page exists based on the HTTP status code
    if response.status_code == 200:
        if "currently unavailable" in response.text or "couldn't find that page" in response.text:
            return False
        else:
            return True
    elif response.status_code == 404:
        return False
    else:
        return False

# for beauty_query in beauty_queries:
#     doc_lst = ranker.query(beauty_query)[:N_DOC_NEEDED]
#     df = pd.DataFrame(columns=['query','title','docid','link','rel'])
#     for i in range(len(doc_lst)):
#         df.loc[i] = [beauty_query, docid_to_title[doc_lst[i][0]], doc_lst[i][0], docid_to_link[doc_lst[i][0]], None]
#     df.to_csv(beauty_query+'.csv', index=False)
    
for beauty_query in beauty_queries:
    doc_lst = ranker.query(beauty_query)
    df = pd.DataFrame(columns=['query', 'title', 'docid', 'link', 'rel'])

    valid_docs_count = 0
    for i in range(len(doc_lst)):
        docid = doc_lst[i][0]
        url = docid_to_link[docid]
        
        # Check if the URL is valid
        if check_amazon_item_exists(url):
            # Add the document to the DataFrame if URL is valid
            df.loc[valid_docs_count] = [beauty_query, docid_to_title[docid], docid, url, None]
            valid_docs_count += 1
            
            # Stop if we reach the required number of valid docs
            if valid_docs_count >= N_DOC_NEEDED:
                break

    # Save to CSV after collecting enough valid documents
    df.to_csv(f"{beauty_query}.csv", index=False)

# for fashion_query in fashion_queries:
#     doc_lst = ranker.query(fashion_query)[:N_DOC_NEEDED]
#     df = pd.DataFrame(columns=['query','title','docid','link','rel'])
#     for i in range(len(doc_lst)):
#         df.loc[i] = [fashion_query, docid_to_title[doc_lst[i][0]], doc_lst[i][0], docid_to_link[doc_lst[i][0]], None]
#     df.to_csv(fashion_query+'.csv', index=False)
    
for fashion_query in fashion_queries:
    doc_lst = ranker.query(fashion_query)
    df = pd.DataFrame(columns=['query', 'title', 'docid', 'link', 'rel'])

    valid_docs_count = 0
    for i in range(len(doc_lst)):
        docid = doc_lst[i][0]
        url = docid_to_link[docid]
        
        # Check if the URL is valid
        if check_amazon_item_exists(url):
            # Add the document to the DataFrame if URL is valid
            df.loc[valid_docs_count] = [fashion_query, docid_to_title[docid], docid, url, None]
            valid_docs_count += 1
            
            # Stop if we reach the required number of valid docs
            if valid_docs_count >= N_DOC_NEEDED:
                break

    # Save to CSV after collecting enough valid documents
    df.to_csv(f"{fashion_query}.csv", index=False)