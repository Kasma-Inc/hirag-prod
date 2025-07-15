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


async def Naive_RAG():
    """
    A simple RAG example using HiRAG
    This prompting uses the prompt to let the LLM give out the references it used directly.
    Not a robust way, just a quick test.

    Uses the RAG chunking system to retrieve relevant chunks from the knowledge base.
    """

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

    raw_data = await index.query_chunks(query)
    # print(raw_data)

    # parsed_chunks = await index.parse_chunks(
    #     raw_data["chunks"],
    #     keep_attr=["text", "document_id"],
    # )
    parsed_dict = await index.parse_dict({"chunks": raw_data})
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

async def Naive_RAG_with_reference():
    """
    A simple RAG example using HiRAG with references.
    This prompting uses the prompt to let the LLM give out which places are using references.
    Then, it uses the ReferenceSeparator and similarity search to fill in the references.
    Should be more robust than the previous one.
    """

    index = await HiRAG.create()
    query = "Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?"

    raw_data = await index.query_chunks(query)
    parsed_dict = await index.parse_dict({"chunks": raw_data})

    prompt_temp = PROMPTS["NAIVE_RAG_PROMPT_NO_ID"]
    prompt = prompt_temp.format(
        context=parsed_dict,
        chat_history="<history> </history>",
        question=query,
        place_holder_begin=PROMPTS["PLACE_HOLDER_BEGIN"],
        place_holder_end=PROMPTS["PLACE_HOLDER_END"],
    )
    
    print(prompt)

    chat_service = ChatCompletion()

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

    # Extract references from the response
    references = await index.reference_separate(response)
    if not references:
        print("No references found in the response.")
        return
    print("References found:", references)

    # Get similar chunks in the vdb, one reference at a time
    similar_chunk_ids = []
    for ref in references:
        found = False
        chunks_ref = await index.query_chunks(ref)
        if chunks_ref:
            for this_chunk in chunks_ref:
                similar_chunk_ids.append(this_chunk["document_key"])
                found = True
                break
        
        if not found:
            print(f"No similar chunks found for reference: {ref}")
            similar_chunk_ids.append("")
            continue

    print("Similar chunks found:", similar_chunk_ids)

    # Fill the placeholders in the response with the references
    filled_response = await index.reference_fill(
        text=response,
        references=similar_chunk_ids
    )

    print("Filled response:", filled_response)

if __name__ == "__main__":
    # asyncio.run(Naive_RAG())
    asyncio.run(Naive_RAG_with_reference())
