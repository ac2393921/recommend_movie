from typing import Any, Dict, List

from pydantic import BaseModel


class RecommendResult(BaseModel):
    rating: Any
    user2items: Dict[int, List[int]]
