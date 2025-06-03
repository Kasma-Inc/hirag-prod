import pytest

from hirag_prod._llm import gpt_4o_mini_complete
from hirag_prod.schema import Entity, Relation
from hirag_prod.storage.networkx import NetworkXGDB


@pytest.mark.asyncio
async def test_networkx_gdb():
    relations = [
        Relation(
            source=Entity(
                id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
                page_content="UNITED STATES",
                metadata={
                    "entity_type": "GEO",
                    "description": "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            target=Entity(
                id="ent-5a28a79d61d9ba7001246e3fdebbe108",
                page_content="HEALTH CARE SYSTEM",
                metadata={
                    "entity_type": "EVENT",
                    "description": "The Health Care System in the United States refers to the organized provision of medical services, which relies on a combination of privatized and government insurance. This system encompasses a variety of healthcare providers and services aimed at delivering medical care to the population, ensuring access to needed health resources through different forms of insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            properties={
                "description": "The United States operates a free market health care system, which defines its overall structure and operation.",
                "weight": 9.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source=Entity(
                id="ent-5a28a79d61d9ba7001246e3fdebbe108",
                page_content="HEALTH CARE SYSTEM",
                metadata={
                    "entity_type": "EVENT",
                    "description": "The Health Care System in the United States refers to the organized provision of medical services, which relies on a combination of privatized and government insurance. This system encompasses a variety of healthcare providers and services aimed at delivering medical care to the population, ensuring access to needed health resources through different forms of insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            target=Entity(
                id="ent-2a422318fc58c5302a5ba9365bcbc0be",
                page_content="INSURANCE COMPANIES",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                    "chunk_ids": [
                        "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                        "chunk-5b8421d1da0999a82176b7836b795235",
                    ],
                },
            ),
            properties={
                "description": "The health care system in the U.S. is heavily influenced by insurance companies that provide policies to consumers and sign contracts with healthcare providers.",
                "weight": 8.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source=Entity(
                id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
                page_content="UNITED STATES",
                metadata={
                    "entity_type": "GEO",
                    "description": "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            target=Entity(
                id="ent-2a422318fc58c5302a5ba9365bcbc0be",
                page_content="INSURANCE COMPANIES",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                    "chunk_ids": [
                        "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                        "chunk-5b8421d1da0999a82176b7836b795235",
                    ],
                },
            ),
            properties={
                "description": "Insurance companies operate within the framework of the U.S. health care system, affecting how services are delivered and financed.",
                "weight": 7.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source=Entity(
                id="ent-2a422318fc58c5302a5ba9365bcbc0be",
                page_content="INSURANCE COMPANIES",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                    "chunk_ids": [
                        "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                        "chunk-5b8421d1da0999a82176b7836b795235",
                    ],
                },
            ),
            target=Entity(
                id="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
                page_content="HEALTH CARE PROVIDERS",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Health Care Providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured.",
                    "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
                },
            ),
            properties={
                "description": "Insurance companies restrict payment to health care providers based on contracts that set fixed fees for services.",
                "weight": 8.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
    ]

    gdb = NetworkXGDB.create(path="test.gpickle", llm_func=gpt_4o_mini_complete)
    for relation in relations:
        await gdb.upsert_relation(relation)
    await gdb.dump()


@pytest.mark.asyncio
async def test_merge_node():
    gdb = NetworkXGDB.create(path="test.gpickle", llm_func=gpt_4o_mini_complete)
    description1 = "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage."
    description2 = "The medical system in the United States is a complex network of hospitals, clinics, and other healthcare providers that provide medical care to the population."
    node1 = Entity(
        id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
        page_content="UNITED STATES",
        metadata={
            "entity_type": "GEO",
            "description": description1,
            "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
        },
    )
    node2 = Entity(
        id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
        page_content="UNITED STATES",
        metadata={
            "entity_type": "GEO",
            "description": description2,
            "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
        },
    )
    await gdb.upsert_node(node1)
    await gdb.upsert_node(node2)

    node = await gdb.query_node(node1.id)
    assert node.metadata.description != description1
    assert node.metadata.description != description2
    assert isinstance(node.metadata.description, str)
    assert len(node.metadata.description) > 0


@pytest.mark.asyncio
async def test_query_one_hop():
    gdb = NetworkXGDB.create(path="test.gpickle", llm_func=gpt_4o_mini_complete)

    relations = [
        Relation(
            source=Entity(
                id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
                page_content="UNITED STATES",
                metadata={
                    "entity_type": "GEO",
                    "description": "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            target=Entity(
                id="ent-5a28a79d61d9ba7001246e3fdebbe108",
                page_content="HEALTH CARE SYSTEM",
                metadata={
                    "entity_type": "EVENT",
                    "description": "The Health Care System in the United States refers to the organized provision of medical services, which relies on a combination of privatized and government insurance. This system encompasses a variety of healthcare providers and services aimed at delivering medical care to the population, ensuring access to needed health resources through different forms of insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            properties={
                "description": "The United States operates a free market health care system, which defines its overall structure and operation.",
                "weight": 9.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source=Entity(
                id="ent-5a28a79d61d9ba7001246e3fdebbe108",
                page_content="HEALTH CARE SYSTEM",
                metadata={
                    "entity_type": "EVENT",
                    "description": "The Health Care System in the United States refers to the organized provision of medical services, which relies on a combination of privatized and government insurance. This system encompasses a variety of healthcare providers and services aimed at delivering medical care to the population, ensuring access to needed health resources through different forms of insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            target=Entity(
                id="ent-2a422318fc58c5302a5ba9365bcbc0be",
                page_content="INSURANCE COMPANIES",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                    "chunk_ids": [
                        "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                        "chunk-5b8421d1da0999a82176b7836b795235",
                    ],
                },
            ),
            properties={
                "description": "The health care system in the U.S. is heavily influenced by insurance companies that provide policies to consumers and sign contracts with healthcare providers.",
                "weight": 8.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source=Entity(
                id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
                page_content="UNITED STATES",
                metadata={
                    "entity_type": "GEO",
                    "description": "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            target=Entity(
                id="ent-2a422318fc58c5302a5ba9365bcbc0be",
                page_content="INSURANCE COMPANIES",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                    "chunk_ids": [
                        "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                        "chunk-5b8421d1da0999a82176b7836b795235",
                    ],
                },
            ),
            properties={
                "description": "Insurance companies operate within the framework of the U.S. health care system, affecting how services are delivered and financed.",
                "weight": 7.0,
                "chunk_id": "chunk-5b8421d1da0999a82176b7836b795235",
            },
        ),
        Relation(
            source=Entity(
                id="ent-2a422318fc58c5302a5ba9365bcbc0be",
                page_content="INSURANCE COMPANIES",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Insurance Companies are private entities that offer health insurance coverage and establish payment processes for healthcare services based on contracts with providers. They play a crucial role in the healthcare system by managing risk and ensuring that individuals have access to necessary medical services through their insurance plans.",
                    "chunk_ids": [
                        "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
                        "chunk-5b8421d1da0999a82176b7836b795235",
                    ],
                },
            ),
            target=Entity(
                id="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
                page_content="HEALTH CARE PROVIDERS",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Health Care Providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured.",
                    "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
                },
            ),
            properties={
                "description": "Insurance companies restrict payment to health care providers based on contracts that set fixed fees for services.",
                "weight": 8.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
        Relation(
            source=Entity(
                id="ent-8ac4883b1b6f421ea5f0196eb317b2ba",
                page_content="HEALTH CARE PROVIDERS",
                metadata={
                    "entity_type": "ORGANIZATION",
                    "description": "Health Care Providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured.",
                    "chunk_ids": ["chunk-d66c81e0b32e3d4e6777f0dfbabe81a8"],
                },
            ),
            target=Entity(
                id="ent-3ff39c0f9a2e36a5d47ded059ba14673",
                page_content="UNITED STATES",
                metadata={
                    "entity_type": "GEO",
                    "description": "The United States is a country characterized by a free market health care system that encompasses a diverse array of insurance providers and health care facilities. This system allows for competition among various organizations, which can lead to a wide range of options for consumers seeking medical care and insurance coverage.",
                    "chunk_ids": ["chunk-5b8421d1da0999a82176b7836b795235"],
                },
            ),
            properties={
                "description": "Health care providers are the professionals or facilities that offer medical treatments and services to patients, regardless of their insurance status, whether they are insured or uninsured.",
                "weight": 8.0,
                "chunk_id": "chunk-d66c81e0b32e3d4e6777f0dfbabe81a8",
            },
        ),
    ]

    gdb = NetworkXGDB.create(path="test.gpickle", llm_func=gpt_4o_mini_complete)

    for relation in relations:
        await gdb.upsert_relation(relation)
    neighbors, edges = await gdb.query_one_hop("ent-8ac4883b1b6f421ea5f0196eb317b2ba")
    assert len(neighbors) == 2
    assert len(edges) == 2
    assert set([n.id for n in neighbors]) == {
        "ent-3ff39c0f9a2e36a5d47ded059ba14673",
        "ent-2a422318fc58c5302a5ba9365bcbc0be",
    }
    assert set([e.source.id for e in edges]) == {
        "ent-8ac4883b1b6f421ea5f0196eb317b2ba",
        "ent-8ac4883b1b6f421ea5f0196eb317b2ba",
    }
    assert set([e.target.id for e in edges]) == {
        "ent-3ff39c0f9a2e36a5d47ded059ba14673",
        "ent-2a422318fc58c5302a5ba9365bcbc0be",
    }
