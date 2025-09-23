from typing import List, Union

from pgvector import HalfVector, Vector
from pgvector.sqlalchemy import HALFVEC, VECTOR

from hirag_prod.configs.functions import get_init_config

INIT_ENVS = get_init_config()
dim, use_halfvec = INIT_ENVS.EMBEDDING_DIMENSION, INIT_ENVS.USE_HALF_VEC
PGVector = Union[HalfVector, Vector, List[float]]
PGVECTOR = HALFVEC(dim) if use_halfvec else VECTOR(dim)
