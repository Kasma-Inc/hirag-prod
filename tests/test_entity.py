import pytest

from hirag_prod._llm import create_chat_service
from hirag_prod.entity.vanilla import VanillaEntity
from hirag_prod.schema import Chunk, Entity, Relation


@pytest.mark.asyncio
async def test_vanilla_entity():
    entity_handler = VanillaEntity.create(
        extract_func=create_chat_service().complete,
        llm_model_name="gpt-4o-mini",
    )

    chunks = [
        Chunk(
            id="chunk-5b8421d1da0999a82176b7836b795235",
            metadata={
                "type": "pdf",
                "chunk_type": "text",
                "filename": "Guide-to-U.S.-Healthcare-System.pdf",
                "page_number": 4,
                "chunk_idx": 0,
                "document_id": "doc-02492a6d371e70c7acd2196f7d8ce6d6",
                "private": False,
                "uri": "/chatbot/Sagi/src/Sagi/mcp_server/hirag_mcp/tests/Guide-to-U.S.-Healthcare-System.pdf",
            },
            page_content="A Very General Overview of How the U.S. Health Care System Works \nThe United States is considered a free market health care system with privatized and some \ngovernment insurance providers. Basically, it is a pay-as-you-can-afford system. The private \ninsurance industry offers individual and group policies.  Health care providers (physicians, \nhospitals, pharmacies, diagnostic facilities, therapeutic facilities, nursing care facilities, and \nso on) sign contracts with insurance providers. Private",
        ),
        Chunk(
            id="chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            metadata={
                "type": "pdf",
                "chunk_type": "text",
                "filename": "Guide-to-U.S.-Healthcare-System.pdf",
                "page_number": 4,
                "chunk_idx": 1,
                "document_id": "doc-02492a6d371e70c7acd2196f7d8ce6d6",
                "private": False,
                "uri": "/chatbot/Sagi/src/Sagi/mcp_server/hirag_mcp/tests/Guide-to-U.S.-Healthcare-System.pdf",
            },
            page_content=") sign contracts with insurance providers. Private insurance companies then use the \nvolume of insured patients that they control in these plans to restrict payment to the health \ncare providers who have agreed by contract to take a fixed fee for each service.  \n \nAfter a person receives care, the providers send the bill to either the patient's insurance \nprovider, or, if the patient has no insurance, to the patient.  \n \nThe insurance company will pay the provider \nall, some, or none of what is ",
        ),
    ]

    entities = await entity_handler.entity(chunks)
    assert isinstance(entities[0], Entity)


@pytest.mark.asyncio
# @pytest.mark.skip(reason="Skipping relation extraction test")
async def test_vanilla_relation():
    chunks = [
        Chunk(
            id="chunk-5b8421d1da0999a82176b7836b795235",
            metadata={
                "type": "pdf",
                "chunk_type": "text",
                "filename": "Guide-to-U.S.-Healthcare-System.pdf",
                "page_number": 4,
                "chunk_idx": 0,
                "document_id": "doc-02492a6d371e70c7acd2196f7d8ce6d6",
                "private": False,
                "uri": "/chatbot/Sagi/src/Sagi/mcp_server/hirag_mcp/tests/Guide-to-U.S.-Healthcare-System.pdf",
            },
            page_content="A Very General Overview of How the U.S. Health Care System Works \nThe United States is considered a free market health care system with privatized and some \ngovernment insurance providers. Basically, it is a pay-as-you-can-afford system. The private \ninsurance industry offers individual and group policies.  Health care providers (physicians, \nhospitals, pharmacies, diagnostic facilities, therapeutic facilities, nursing care facilities, and \nso on) sign contracts with insurance providers. Private",
        ),
        Chunk(
            id="chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            metadata={
                "type": "pdf",
                "chunk_type": "text",
                "filename": "Guide-to-U.S.-Healthcare-System.pdf",
                "page_number": 4,
                "chunk_idx": 1,
                "document_id": "doc-02492a6d371e70c7acd2196f7d8ce6d6",
                "private": False,
                "uri": "/chatbot/Sagi/src/Sagi/mcp_server/hirag_mcp/tests/Guide-to-U.S.-Healthcare-System.pdf",
            },
            page_content=") sign contracts with insurance providers. Private insurance companies then use the \nvolume of insured patients that they control in these plans to restrict payment to the health \ncare providers who have agreed by contract to take a fixed fee for each service.  \n \nAfter a person receives care, the providers send the bill to either the patient's insurance \nprovider, or, if the patient has no insurance, to the patient.  \n \nThe insurance company will pay the provider \nall, some, or none of what is ",
        ),
    ]

    entities = [
        Entity(
            id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
            metadata={
                "entity_type": "GEO",
                "description": [
                    "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage."
                ],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
            page_content="UNITED STATES",
        ),
        Entity(
            id="ent-5a28a79d61d9ba7001246e3fdebbe108",
            metadata={
                "entity_type": "EVENT",
                "description": [
                    "The Health Care System in the United States refers to the organized provision of medical services, which relies on a combination of privatized and government insurance. This system encompasses a variety of healthcare providers and services aimed at delivering medical care to the population, ensuring access to needed health resources through different forms of insurance coverage."
                ],
                "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
            },
            page_content="HEALTH CARE SYSTEM",
        ),
        Entity(
            id="ent-2a422318fc58c5302a5ba9365bcbc0be",
            metadata={
                "entity_type": "ORGANIZATION",
                "description": [
                    "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans."
                ],
                "chunk_ids": [
                    "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                    "chunk-5b8421d1da0999a82176b7836b795235",
                ],
            },
            page_content="INSURANCE COMPANIES",
        ),
        Entity(
            id="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
            metadata={
                "entity_type": "ORGANIZATION",
                "description": [
                    "Health Care Providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured."
                ],
                "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
            },
            page_content="HEALTH CARE PROVIDERS",
        ),
    ]
    entity_handler = VanillaEntity.create(
        extract_func=create_chat_service().complete,
        llm_model_name="gpt-4o-mini",
    )
    relations = await entity_handler.relation(chunks, entities)
    assert isinstance(relations[0], Relation)
