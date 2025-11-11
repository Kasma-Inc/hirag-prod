from typing import List, Union

from pgvector import HalfVector, Vector
from pgvector.sqlalchemy import HALFVEC, VECTOR

from configs.functions import get_envs

PGVector = Union[HalfVector, Vector, List[float]]
PGVECTOR = (
    HALFVEC(get_envs().EMBEDDING_DIMENSION)
    if get_envs().USE_HALF_VECTOR
    else VECTOR(get_envs().EMBEDDING_DIMENSION)
)
