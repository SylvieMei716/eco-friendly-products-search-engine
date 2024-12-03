'''
Edited by Lindsey Dye

Author: Prithvijit Dasgupta
Modified by: Zim Gong

This is the FastAPI start index. Currently it has 4 paths

1. GET / -> Fetches the test bench HTML file. Used by browsers
2. POST /search -> This is the main search API which is responsible for perform the search across the index
3. GET /cache/:query/page/:page -> This path is meant to be a cached response for pagination purposes.
4. GET /experiment -> Run a relevance experiment
'''
# importing external modules
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from threading import Timer
import math
from typing import List

# importing internal modules
from models import QueryModel, APIResponse, PaginationModel, SearchResponse
from pipeline import initialize
from relevance import run_relevance_tests

algorithm = initialize()
pagination_cache = {}
timer_mgr = {}

# Some global configurations
PAGE_SIZE = 10
CACHE_TIME = 3600

# this is the FastAPI application
app = FastAPI()

# Establish eco-friendly and non-friendly keywords
ecofriendly_keywords = [
    'sustainable', 'organic', 'biodegradable', 'recyclable', 'compostable', 'recycled', 
    'non-toxic', 'renewable', 'plant-based', 'vegan', 'low-impact', 'zero-waste', 
    'green', 'cruelty-free', 'FSC-certified', 'carbon-neutral', 'Energy Star', 'Fair Trade', 
    'eco-conscious', 'climate-positive', 'upcycled', 'responsibly sourced', 'energy-efficient', 
    'plastic-free', 'pesticide-free', 'natural', 'ethical', 'eco-label', 'water-saving', 
    'low-carbon', 'toxin-free', 'green-certified', 'eco-safe'
]

nonfriendly_keywords = ['non-recyclable', 'disposable', 'single-use', 'harmful', 'polluting', 'toxic', 'unsustainable']

# Helper function to add eco-friendliness tag based on keyword matching
def eco_friendly_tag(product):
    """Tag products as eco-friendly or non eco-friendly based on certain criteria."""
    product_name = product['name'].lower()  # Convert the product name to lowercase for keyword matching
    
    # Check for eco-friendly keywords
    for keyword in ecofriendly_keywords:
        if keyword in product_name:
            return "Eco-friendly"
    
    # Check for non-friendly keywords
    for keyword in nonfriendly_keywords:
        if keyword in product_name:
            return "Not eco-friendly"
    
    # If no keyword matches, return a neutral tag
    return "Eco-friendly (Uncertain)"

# cache deletion function used to delete cache entries after a set timeout.
def delete_from_cache(query):
    global pagination_cache
    if query in pagination_cache:
        del pagination_cache[query]
        del timer_mgr[query]

# API paths begin here
@app.get('/', response_class=HTMLResponse)
async def home():
    with open('./web/home.html') as f:
        return f.read()

@app.post('/search')
async def doSearch(body: QueryModel) -> APIResponse:
    request_query = body.query
    response = algorithm.search(request_query)

    # Add eco-friendliness tag and sort by price (low to high)
    sorted_response = sorted(response, key=lambda x: x['price'])
    for product in sorted_response:
        product['eco_friendly_tag'] = eco_friendly_tag(product)

    global pagination_cache
    pagination_cache[request_query] = sorted_response
    pagination_cache[f'{request_query}_max_page'] = math.floor(len(sorted_response) / PAGE_SIZE)

    global timer_mgr
    t = Timer(CACHE_TIME, delete_from_cache, [request_query])
    timer_mgr[request_query] = t
    t.start()

    # Pagination links
    next_page = 1 if len(sorted_response) > PAGE_SIZE else 0
    prev_page = 0

    return APIResponse(
        results=[SearchResponse(**product) for product in sorted_response[:PAGE_SIZE]],
        page=PaginationModel(prev=f'/cache/{request_query}/page/{prev_page}',
                             next=f'/cache/{request_query}/page/{next_page}')
    )


@app.get('/experiment')
async def runExperiment() -> APIResponse:
    results = run_relevance_tests(algorithm)
    return APIResponse(results=results, page=None)


@app.get('/cache/{query}/page/{page}')
async def getCache(query: str, page: int) -> APIResponse:
    if query in pagination_cache:
        if page < 0:
            page = 0
        max_page = pagination_cache[f'{query}_max_page']
        if page > max_page:
            page = max_page

        prev_page = page - 1 if page > 0 else 0
        next_page = page + 1 if page < max_page else max_page

        # Paginated results
        results = pagination_cache[query][page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        return APIResponse(
            results=[SearchResponse(**product) for product in results],
            page=PaginationModel(prev=f'/cache/{query}/page/{prev_page}',
                                 next=f'/cache/{query}/page/{next_page}')
        )
    else:
        return await doSearch(QueryModel(query=query))


# python does not have a way to gracefully handle timeout handlers. This is an attempt to ensure graceful shutdown
# does not work a few times
# TODO find a more graceful solution or fix the bug
@app.on_event('shutdown')
def timer_shutddown():
    [timer_mgr[key].cancel() for key in timer_mgr]
