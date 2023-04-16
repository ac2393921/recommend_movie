import numpy as np
import pandas as pd
from pandera import Column, Field, SchemaModel, SeriesSchema
from pandera.typing import Bool, DataFrame, Object, Series
from pydantic import BaseModel


class MoviesBaseSchema(SchemaModel):
    movie_id: Series[np.int64]
    title: Series[Object]
    genres: Series[Object]


class MoviesSchema(SchemaModel):
    movie_id: Series[np.int64]
    title: Series[Object]
    genres: Series[Object]
    tag: Series[Object] = Field(nullable=True, coerce=True)


class TagsBaseSchema(SchemaModel):
    user_id: Series[np.int64]
    movie_id: Series[np.int64]
    tag: Series[Object] = Field(nullable=True, coerce=True)
    timestamp: Series[np.int64]


class RatingsBaseSchema(SchemaModel):
    user_id: Series[np.int64]
    movie_id: Series[np.int64]
    rating: Series[np.float64]
    timestamp: Series[np.int64]


class MoviesRatingSchema(SchemaModel):
    movie_id: Series[np.int64]
    title: Series[Object]
    genres: Series[Object]
    rating: Series[np.float64]
    timestamp: Series[np.int64]
