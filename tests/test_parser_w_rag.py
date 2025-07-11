"""Quick test for LLM services"""

import asyncio
import os
import time
from unittest.mock import patch

import pytest

from hirag_prod._llm import APIConfig, ChatCompletion, EmbeddingService, TokenUsage

# This is a quickstart script for the HiRAG system.
import asyncio
import logging

from hirag_prod import HiRAG
from hirag_prod.prompt import PROMPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv("/chatbot/.env")


async def index():
    index = await HiRAG.create()
    # document_path = f"benchmark/2wiki/2wiki_subcorpus.txt"
    # content_type = "text/plain"
    # document_meta = {
    #     "type": "txt",
    #     "filename": "2wiki_subcorpus.txt",
    #     "uri": document_path,
    #     "private": False,
    # }
    # await index.insert_to_kb(
    #     document_path=document_path,
    #     content_type=content_type,
    #     document_meta=document_meta,
    # )


    query = "What is the place of birth of the performer of song Changed It?"

    raw_data = await index.query_all(query)
    # print(raw_data)

    # parsed_chunks = await index.parse_chunks(
    #     raw_data["chunks"],
    #     keep_attr=["text", "document_id"],
    # )
    parsed_dict = await index.parse_dict({"chunks": raw_data["chunks"]})
    # print(parsed_dict)

    # follow the prompt format to generate a prompt
    prompt_temp = PROMPTS["NAIVE_RAG_PROMPT"]

    # the temp includes {context}, {chat_history}, {question}

    prompt = prompt_temp.format(
        context=parsed_dict,
        chat_history="<history> </history>",
        question=query,
    )
    print(prompt)

    chat_service = ChatCompletion() 

    # chat using the prompt
    try:
        response = await chat_service.complete(
            model="gpt-4o-mini",
            prompt=prompt,
            timeout=30.0,
        )
        print("Response:", response)
    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        print("An error occurred during chat completion.")


if __name__ == "__main__":
    asyncio.run(index())
