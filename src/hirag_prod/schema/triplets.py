import os
from datetime import datetime
from typing import List

import dotenv
from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import Column, DateTime, String, Text

from hirag_prod.schema.base import Base

# read halfvec and dim from env
dotenv.load_dotenv()
dim = int(os.getenv("EMBEDDING_DIMENSION", 1536))
use_halfvec = bool(os.getenv("USE_HALF_VEC", True))

vec_type = HALFVEC(dim) if use_halfvec else Vector(dim)


class Triplets(Base):
    __tablename__ = "Triplets"

    source: str = Column(String, nullable=False)
    target: str = Column(String, nullable=False)
    description: str = Column(Text, primary_key=True, nullable=False)
    knowledgeBaseId: str = Column(String, primary_key=True, nullable=False)
    workspaceId: str = Column(String, primary_key=True, nullable=False)
    fileName: str = Column(String, nullable=False)
    documentId: str = Column(String, nullable=False)
    vector: List[float] = Column(vec_type, nullable=False)
    updatedAt: datetime = Column(DateTime, default=datetime.now, nullable=False)
