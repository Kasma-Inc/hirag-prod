# Use the reference parser's parse_references
import asyncio

from hirag_prod.parser.reference_parser import ReferenceParser

text = "The standard for the U.S. healthcare system is characterized as a free market system that features both privatized and some government insurance providers. It operates on a pay-as-you-can-afford basis, meaning that health care access and costs can vary significantly depending on an individual's ability to pay [|REFERENCE|]. \n\nIn this system, health care providers, such as physicians and hospitals, enter into contracts with private insurance companies, which negotiate fixed fees for services. The ultimate costs for patients depend on their insurance coverage and the terms of their contracts with insurance providers. Patients may be responsible for all, some, or none of the costs incurred, depending on their insurance plans [|REFERENCE|]. \n\nFurthermore, individuals, particularly those on J-1 visas and their dependents, are required by law to have health insurance meeting specific criteria [|REFERENCE|]. However, many medical providers in the U.S. do not bill foreign insurance entities directly, which may require patients to pay upfront for services and seek reimbursement later [|REFERENCE|]. This complex framework underlines significant variability in healthcare access and costs across the U.S. [|REFERENCE|]."
reference_placeholder = "[|REFERENCE|]"

parser = ReferenceParser()
references = asyncio.run(parser.parse_references(text, reference_placeholder))
for i, ref in enumerate(references):
    print(f"Reference {i+1}: {ref}")
