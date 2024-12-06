'''
Edited by Lindsey Dye

Author: Prithvijit Dasgupta

This file contains the base models required by the service to function
'''
from pydantic import BaseModel

class QueryModel(BaseModel):
    query:str
    sort_option: str

class SearchResponse(BaseModel):
    id: int
    docid: int
    score: float
    image: str | None  # URL to the product image
    link: str | None   # URL to the product page
    eco_friendly_tag: str | None  # Eco-friendliness tag

class PaginationModel(BaseModel):
    prev: str
    next: str

class APIResponse(BaseModel):
    results: list[SearchResponse]
    page: PaginationModel | None

class ExperimentResponse(BaseModel):
    ndcg: float
    query: str

class BaseSearchEngine():
    def __init__(self, path: str) -> None:
        pass

    def index(self):
        pass

    def search(self, query: str) -> list[SearchResponse]:
        pass
