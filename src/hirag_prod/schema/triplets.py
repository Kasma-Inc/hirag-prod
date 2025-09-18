from datetime import datetime
from typing import List

from sqlalchemy import Column, DateTime, String, Text

from hirag_prod.schema.base import Base
from hirag_prod.schema.vector_config import vec_type


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
