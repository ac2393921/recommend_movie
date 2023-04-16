from typing import Dict, List

from pandera.typing import DataFrame
from pydantic import BaseModel

from src.dataset.shema import MoviesSchema


class Dataset(BaseModel):
    train: DataFrame[MoviesSchema]
    test: DataFrame[MoviesSchema]
    test_user2items: Dict[int, List[int]]
    item_content: DataFrame[MoviesSchema]
