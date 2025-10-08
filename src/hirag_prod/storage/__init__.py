"""
HiRAG Storage Management Module

Provides utilities for managing various storage backends including vector databases,
graph databases, and Redis-based document processing state management.
"""

from hirag_prod.storage.base_vdb import BaseVDB

__all__ = [
    "BaseVDB",
]
