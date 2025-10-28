from typing import Any, Dict, List, Optional

from humps import camelize
from pydantic import BaseModel, ConfigDict, Field, validator


class ProcessSearchResponse(BaseModel):
    synonym_list: List[str]
    is_english: bool
    translation_list: List[str]


def to_camel(string: str) -> str:
    return camelize(string)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")


class RectangularHighlight(CamelModel):
    """Represents a rectangular highlight region for PDF or Image files."""


class RectangularBbox(CamelModel):

    x1: float = Field(..., description="X-coordinate of the top-left corner")
    y1: float = Field(..., description="Y-coordinate of the top-left corner")
    x2: float = Field(..., description="X-coordinate of the bottom-right corner")
    y2: float = Field(..., description="Y-coordinate of the bottom-right corner")

    @validator("x2")
    def check_x2(cls, v, values):
        if "x1" in values and v <= values["x1"]:
            raise ValueError("x2 must be > x1")
        return v

    @validator("y2")
    def check_y2(cls, v, values):
        if "y1" in values and v >= values["y1"]:
            raise ValueError("y2 must be < y1")
        return v


class PDFBbox(RectangularBbox):

    pass


class PDFHighlight(RectangularHighlight):

    page_number: int = Field(..., ge=0, description="Page number in the PDF (0-based)")
    width: float = Field(..., ge=0, description="Width of the page")
    height: float = Field(..., ge=0, description="Height of the page")
    bboxes: List[PDFBbox] = Field(..., description="List of bounding boxes")


class MarkdownBbox(CamelModel):

    from_idx: int = Field(..., ge=0, description="Start index of the highlighted text")
    to_idx: int = Field(..., ge=0, description="End index of the highlighted text")

    @validator("to_idx")
    def check_end_idx(cls, v, values):
        if "from_idx" in values and v < values["from_idx"]:
            raise ValueError("to_idx must be >= from_idx")
        return v


class MarkdownHighlight(CamelModel):

    bboxes: List[MarkdownBbox] = Field(..., description="List of text ranges")


class ExcelBbox(CamelModel):

    col: Optional[int] = Field(..., description="column index (0-based: 0, 1, 2, ...)")
    row: Optional[int] = Field(..., description="row index (0-based: 0, 1, 2, ...)")


class ExcelHighlight(CamelModel):

    bboxes: List[ExcelBbox] = Field(..., description="List of cells")


class ImageBbox(RectangularBbox):

    pass


class ImageHighlight(RectangularHighlight):

    width: float = Field(..., ge=0, description="Width of the image")
    height: float = Field(..., ge=0, description="Height of the image")
    bboxes: List[ImageBbox] = Field(..., description="List of bounding boxes")
