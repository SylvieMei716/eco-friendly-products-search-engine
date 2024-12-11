'''
Edited by Sylvie Mei

Author: Prithvijit Dasgupta

This file contains the base models required by the service to function
'''
from pydantic import BaseModel

class QueryModel(BaseModel):
    query:str
    sort_option: str = "relevance"

class SearchResponse(BaseModel):
    docid: int
    link: str
    eco_friendly: str | None
    price: float | None
    title: str
    description: str
    image: str
    avg_rating: float | None

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
