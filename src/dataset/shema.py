import numpy as np
from pandera import Field, SchemaModel
from pandera.typing import Object, Series


class MoviesBaseSchema(SchemaModel):
    """映画データのベーススキーマ"""

    movie_id: Series[np.int64]
    title: Series[Object]
    genres: Series[Object]


class MoviesSchema(SchemaModel):
    """映画データのスキーマ"""

    movie_id: Series[np.int64]
    title: Series[Object]
    genres: Series[Object]
    tag: Series[Object] = Field(nullable=True, coerce=True)


class TagsBaseSchema(SchemaModel):
    """タグデータのベーススキーマ"""

    user_id: Series[np.int64]
    movie_id: Series[np.int64]
    tag: Series[Object] = Field(nullable=True, coerce=True)
    timestamp: Series[np.int64]


class RatingsBaseSchema(SchemaModel):
    """評価データのベーススキーマ"""

    user_id: Series[np.int64]
    movie_id: Series[np.int64]
    rating: Series[np.float64]
    timestamp: Series[np.int64]


class MoviesRatingSchema(SchemaModel):
    """映画評価データのスキーマ"""

    movie_id: Series[np.int64]
    title: Series[Object]
    genres: Series[Object]
    rating: Series[np.float64]
    timestamp: Series[np.int64]
