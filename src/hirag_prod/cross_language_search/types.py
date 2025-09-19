from typing import List

from pydantic import BaseModel


class TranslationResponse(BaseModel):
    translation_list: List[str]


class SynonymResponse(BaseModel):
    synonym_list: List[str]
