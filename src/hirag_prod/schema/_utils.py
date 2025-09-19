from hirag_prod.schema.chunk import Chunk
from hirag_prod.schema.file import File
from hirag_prod.schema.item import Item


def file_to_chunk(
    file: File, documentKey: str, text: str, documentId: str, chunkIdx
) -> Chunk:
    new_chunk = Chunk(
        # Given
        documentKey=documentKey,
        text=text,
        documentId=documentId,
        chunkIdx=chunkIdx,
    )

    # Copy
    for col in dict(file):
        if hasattr(new_chunk, col):
            # Only copy if attr is none in new_chunk
            if getattr(new_chunk, col) is None:
                setattr(new_chunk, col, getattr(file, col))
    return new_chunk


def item_to_chunk(item: Item) -> Chunk:
    new_chunk = Chunk()
    for col in dict(item):
        if hasattr(new_chunk, col):
            attr = getattr(item, col)
            if col == "bbox":
                attr = [attr] if attr is not None else None
            elif col == "pageNumber":
                attr = [attr] if attr is not None else None
            setattr(new_chunk, col, attr)
    return new_chunk


def file_to_item(
    file: File, documentKey: str, text: str, documentId: str, chunkIdx
) -> Item:
    new_item = Item(
        # Given
        documentKey=documentKey,
        text=text,
        documentId=documentId,
        chunkIdx=chunkIdx,
    )
    # Copy
    for col in dict(file):
        if hasattr(new_item, col):
            # Only copy if attr is none in new_item
            if getattr(new_item, col) is None:
                setattr(new_item, col, getattr(file, col))
    return new_item
