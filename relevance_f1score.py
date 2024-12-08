"""
Edited by: Pooja Thakur

NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""
import math
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd


import pandas as pd
def map_score(search_result_relevances: list[int], cut_off: int) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    if len(search_result_relevances)<cut_off:
        cut_off = len(search_result_relevances)
    relevant_docs = 0
    precision_sum = 0
    i = 0
    while i<cut_off:
        if search_result_relevances[i]==1:
            relevant_docs += 1
            precision_sum += relevant_docs / (i+1)
        i+=1
    map = precision_sum/cut_off
    return map


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_of: int):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    if len(search_result_relevances)==len(ideal_relevance_score_ordering):
        search_result_relevances = np.array(search_result_relevances[0:cut_of])
        ideal_relevance_score_ordering = np.array(ideal_relevance_score_ordering[0:cut_of])
    else:
        cut_of = min(len(ideal_relevance_score_ordering),len(search_result_relevances))
        search_result_relevances = np.array(search_result_relevances[0:cut_of])
        ideal_relevance_score_ordering = np.array(ideal_relevance_score_ordering[0:cut_of])
        
    l = len(search_result_relevances)
    log2i = np.log2(range(2, l + 1))
    dcg = search_result_relevances[0] + (search_result_relevances[1:] / log2i).sum()
    idcg = ideal_relevance_score_ordering[0] + (ideal_relevance_score_ordering[1:]/log2i).sum()
    ndcg = dcg/idcg
    return ndcg

def f1_score(binary_relevances, actual_relevances, relevant_docs):
    """Calculates the F1 score given binary relevances, the ranked documents' list, and all relevant documents. (i.e.,
    documents with relevance of/above 4 among the annotated documents of a given query). 
    F1 score = 2*(Precision*Recall)/(Precision+Recall)
    
    Args:
        binary_relevances: A list of 0/1 values for whether each search result returned by the
            ranking function is relevant
        actual_relevances: A list of the ranked documents as returned by the ranking function
        relevant_docs: A list of documents with relevances of/above 4 among the annotated documents
            for a given query.
            
    Returns:
        F1 score"""
    # Calculate recall
    recall_docs = sum(1 for doc,_,_ in actual_relevances if doc in relevant_docs)
    if len(relevant_docs)!=0:
        recall = recall_docs / len(relevant_docs)
    else:
        return 0

    # Calculate precision
    precision_docs = sum(binary_relevances)
    precision = precision_docs / len(actual_relevances)

    # Handle division by zero in F1 calculation
    if precision + recall == 0:
        return 0.0

    # Calculate F1-score
    return (2 * precision * recall) / (precision + recall)


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset
    data = pd.read_csv(relevance_data_filename, encoding='cp850')
    cut_off = 10
    
    # Group data by queries
    grouped = data.groupby('query')
    
    # Initialize lists to store MAP and NDCG for each query
    map_scores = []
    ndcg_scores = []
    f1_scores = []
    
    for i, (query, group) in enumerate(grouped):
        #if i>10:
            #break
       
        docs = group['docid'].tolist()
        rels = group['rel'].tolist()
        
        
        docid_to_rel = {docid: rel for docid, rel in zip(docs, rels)}
        
        
        ranked_docs = ranker.query(query,docid_to_price)[:cut_off]
        
        binary_rels = [1 if docid_to_rel.get(doc, 0) >= 4 else 0 for doc,_,_ in ranked_docs]
        
        actual_rels = [docid_to_rel.get(doc, 0) for doc,_,_ in ranked_docs]
        
        #Making a list of relevant docs for given query to calculate recall
        relevant_docs_ = []
        for doc in docid_to_rel.keys():
            if docid_to_rel[doc]>=4:
                relevant_docs_.append(doc)
        
        f1_q = f1_score(binary_rels, ranked_docs, relevant_docs_)
        f1_scores.append(f1_q)

        map_q = map_score(binary_rels, cut_off)
        map_scores.append(map_q)
  
        ideal_rels = sorted(rels, reverse=True)[:cut_off]
        print('Size of ideal:',len(ideal_rels),'Size of actual',len(actual_rels))
        ndcg_q = ndcg_score(actual_rels, ideal_rels, cut_off)
        ndcg_scores.append(ndcg_q)

    avg_map = np.mean(map_scores) if map_scores else 0.0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_f1_score = np.mean(f1_scores) if f1_scores else 0.0
    
    #'f1_score': avg_f1_score,
    #'f1_list': f1_scores

    return {'map': avg_map, 'ndcg': avg_ndcg, 'f1_score': avg_f1_score, 'map_list': map_scores, 'ndcg_list': ndcg_scores, 'f1_list': f1_scores}
