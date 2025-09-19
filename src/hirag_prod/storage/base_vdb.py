from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Union


class BaseVDB(ABC):
    embedding_func: Callable

    @abstractmethod
    async def _init_vdb(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def upsert_texts(
        self,
        texts_to_upsert: List[str],
        properties_list: List[dict],
        table_name: str,
        with_tokenization: bool = False,
        with_translation: bool = False,
        mode: Literal["append", "overwrite"] = "append",
    ):
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        query: Union[str, List[str]],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        topk: Optional[int] = None,
        uri_list: Optional[List[str]] = None,
        require_access: Optional[Literal["private", "public"]] = None,
        columns_to_select: Optional[List[str]] = None,
        distance_threshold: Optional[float] = None,
        topn: Optional[int] = None,
        rerank: bool = False,
    ) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    async def query_by_keys(
        self,
        key_value: List[str],
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
        key_column: str = "documentKey",
        columns_to_select: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    async def get_existing_document_keys(
        self,
        uri: str,
        workspace_id: str,
        knowledge_base_id: str,
        table_name: str,
    ) -> List[str]:
        raise NotImplementedError
