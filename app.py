'''
DO NOT EDIT

Author: Prithvijit Dasgupta
Modified by: Sylvie Mei

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

# importing internal modules
from models import QueryModel, APIResponse, PaginationModel
from pipeline import initialize
from relevance import run_relevance_tests

# Some global variables
# TODO Remove global variables

algorithm = initialize()

pagination_cache = {}
timer_mgr = {}

# doc_data = {
#     "1": {"image": "https://example.com/image1.jpg", "ecoFriendly": "High", "price": 25.99},
#     "2": {"image": "https://example.com/image2.jpg", "ecoFriendly": "Medium", "price": 15.49},
# }

# Some global configurations
PAGE_SIZE = 10
CACHE_TIME = 3600

# this is the FastAPI application
app = FastAPI()


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
    sort_option = body.sort_option
    response = algorithm.search(request_query, sort_option)
    global pagination_cache
    enriched_results = []
    for res in response:
        # docid = res["docid"]
        enriched_results.append(res)
        # enriched_results.append({
        #         "docid": result[0],
        #         "title": doc.get("title", "Unknown"),
        #         "description": doc.get("description", "No description available"),
        #         "score": result[1],
        #         "price": doc.get("price", "N/A"),
        #         "eco_friendly": self.docid_to_ecolabel.get(result[0], "Unknown"),
        #         "image": self.docid_to_image.get(result[0], ""),
        #         "avg_rating": doc.get("average_rating", 0),
        #     })
    pagination_cache[request_query] = enriched_results
    pagination_cache[f'{request_query}_max_page'] = math.floor(
        len(response) / PAGE_SIZE)
    global timer_mgr
    t = Timer(CACHE_TIME, delete_from_cache, [request_query])
    timer_mgr[request_query] = t
    t.start()
    return APIResponse(results=response[:PAGE_SIZE],
                       page=PaginationModel(prev=f'/cache/{request_query}/page/0',
                                            next=f'/cache/{request_query}/page/1'))


@app.get('/experiment')
async def runExperiment() -> APIResponse:
    results = run_relevance_tests(algorithm)
    return APIResponse(results=results, page=None)


@app.get('/cache/{query}/page/{page}')
async def getCache(query: str, page: int) -> APIResponse:
    if query in pagination_cache:
        if page < 0:
            page = 0
        if page == 0:
            prev_page = page
        else:
            prev_page = page-1
        if pagination_cache[f'{query}_max_page'] == page:
            next_page = page
        else:
            next_page = page+1
        return APIResponse(results=pagination_cache[query][page*PAGE_SIZE:(page+1)*PAGE_SIZE],
                           page=PaginationModel(prev=f'/cache/{query}/page/{prev_page}',
                                                next=f'/cache/{query}/page/{next_page}'))
    else:
        return await doSearch(QueryModel(query=query))


# python does not have a way to gracefully handle timeout handlers. This is an attempt to ensure graceful shutdown
# does not work a few times
# TODO find a more graceful solution or fix the bug
@app.on_event('shutdown')
def timer_shutddown():
    [timer_mgr[key].cancel() for key in timer_mgr]
