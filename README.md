# eco-friendly-products-search-engine
### Nov 5:
#### Updates: 
- implemented get_possible_items.py
- indexed dataset
#### TODO:
- copy & paste your homework code `document_preprocessor.py`, `indexing.py`, `ranker.py` in root folder
- copy & paste main index in /main_index (the file is too large to push to github)
- copy & paste `stopwords.txt`, `meta_All_Beauty.jsonl.gz`, `meta_Amazon_Fashion.jsonl.gz` to /data
- run `python get_possible_items.py` to get annotation files (remember to replace with your own queries)


- `python dataset_preprocessor.py` to get full dataset
- `python get_annotation_files.py` to get csv files to annotate
- `python baseline_bm25.py` to run bm25 baseline system
- `python baseline_naive.py` to run naive system
- `python pipeline.py` to run demo search
- `python uvicorn app:app.py` to get search engine