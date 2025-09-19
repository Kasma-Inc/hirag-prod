from typing import List

from pydantic import BaseModel


class ProcessSearchResponse(BaseModel):
    synonym_list: List[str]
    is_english: bool
    translation_list: List[str]
