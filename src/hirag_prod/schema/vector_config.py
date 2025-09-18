from pgvector.sqlalchemy import HALFVEC, Vector

from hirag_prod.configs.functions import get_init_config

INIT_ENVS = get_init_config()
dim, use_halfvec = INIT_ENVS.EMBEDDING_DIMENSION, INIT_ENVS.USE_HALF_VEC
vec_type = HALFVEC(dim) if use_halfvec else Vector(dim)
