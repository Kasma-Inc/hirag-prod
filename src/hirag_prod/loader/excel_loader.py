import asyncio
import os
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd

from hirag_prod._utils import compute_mdhash_id
from hirag_prod.configs.functions import get_llm_config
from hirag_prod.exceptions import HiRAGException
from hirag_prod.loader.utils import route_file_path
from hirag_prod.prompt import PROMPTS
from hirag_prod.resources.functions import get_chat_service
from hirag_prod.schema import Chunk, File, Item, file_to_item, item_to_chunk


def _keep_sheet(name: str) -> bool:
    s = (name or "").lower()
    return ("cache" not in s) and ("detail" not in s)


async def _summarize_excel_sheet(sheet_name: str, latex: str) -> str:
    system_prompt = PROMPTS["summary_excel_en"].format(
        sheet_name=sheet_name, latex=latex
    )
    try:
        return await get_chat_service().complete(
            prompt=system_prompt, model=get_llm_config().model_name
        )
    except Exception as e:
        raise HiRAGException(f"Failed to summarize excel sheet {sheet_name}") from e


async def load_and_chunk_excel(
    document_path: str,
    document_meta: Dict,
) -> Tuple[List[Chunk], File, List[Item]]:
    try:
        if document_path.startswith("file://"):
            p = urlparse(document_path)
            raw_path = f"/{p.netloc}{p.path}" if p.netloc else p.path
            local_path = os.path.abspath(unquote(raw_path))
        else:
            try:
                local_path = route_file_path("excel_loader", document_path)
            except Exception:
                local_path = document_path
        all_sheets: Dict[str, pd.DataFrame] = pd.read_excel(local_path, None)

        filtered_sheets = [
            (name, df) for name, df in all_sheets.items() if _keep_sheet(name)
        ]

        document_id = document_meta.get("documentKey", "")
        file_name = document_meta.get("fileName", os.path.basename(local_path))

        generated_md = File(
            documentKey=document_id,
            text=file_name,
            type=document_meta.get("type", "xlsx"),
            pageNumber=len(filtered_sheets),
            fileName=file_name,
            uri=document_meta.get("uri", document_path),
            private=bool(document_meta.get("private", False)),
            uploadedAt=document_meta.get("uploadedAt", datetime.now()),
            knowledgeBaseId=document_meta.get("knowledgeBaseId", ""),
            workspaceId=document_meta.get("workspaceId", ""),
        )

        latex_list: List[str] = []
        sheet_names: List[str] = []
        for sheet_name, df in filtered_sheets:
            try:
                latex_list.append(df.to_latex(index=False))
            except Exception:
                latex_list.append(df.to_string(index=False))
            sheet_names.append(sheet_name)

        captions = await asyncio.gather(
            *[
                _summarize_excel_sheet(sheet_name, latex)
                for sheet_name, latex in zip(sheet_names, latex_list)
            ]
        )

        items: List[Item] = []
        chunks: List[Chunk] = []

        for idx, (name, latex, caption) in enumerate(
            zip(sheet_names, latex_list, captions), start=1
        ):
            sheet_key = compute_mdhash_id(f"{document_id}:{name}", prefix="chunk-")

            item = file_to_item(
                generated_md,
                documentKey=sheet_key,
                text=(latex or "").strip(),
                documentId=document_id,
                chunkIdx=idx,
            )
            item.caption = (caption or "None").strip()
            item.chunkType = "excel_sheet"

            chunk = item_to_chunk(item)
            chunk.text = (latex or "None").strip()
            chunk.caption = (caption or "None").strip()
            chunk.chunkType = "excel_sheet"

            items.append(item)
            chunks.append(chunk)

        return chunks, generated_md, items

    except Exception as e:
        raise HiRAGException("Failed to load and chunk excel document") from e
