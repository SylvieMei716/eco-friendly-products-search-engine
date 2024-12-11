# eco-friendly-products-search-engine
## Introduction
This is the GitHub repository for an eco-friendly beauty & fashion products search engine. 

Due to the GitHub file size limitation, the dataset and cache files were not uploaded to this 
repo. The original dataset could be found via https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023.

## How to run
To directly run the search algorithm or launch the search engine (Note: since we set 
random sleep in item link checking part, it might take around 45 seconds to show the 
results when choosing sort by relevance only; while re-sorting by relevance and 
rating will take longer, approximately several minutes):

- Modify the query in main function of `pipeline.py`, and run `python pipeline.py`, 
to get the top results for your query

OR

- run `python uvicorn app:app.py` to launch the search engine

To start from scratch:

1. `python dataset_preprocessor.py` to get full dataset
2. `python get_annotation_files.py` to get csv files to annotate
3. `python train_test_splitting.py` to split training and testing files from annotated files
4. `python baseline_bm25.py` to run bm25 baseline system
5. `python baseline_naive.py` to run naive system
6. `python multimodal.py` to prepare the image embeddings
7. `python pipeline.py` to run demo search
8. `python uvicorn app:app.py` to launch search engine
