import json
from typing import Any, Dict, List

import pytest

from hirag_prod.loader.dots_ocr import DotsOCRClient


class TestDotsTreeGeneration:
    """Test class for Dots OCR tree generation functionality"""

    def test_generate_tree_with_json_file(self):
        """Test the generate_tree function with a real JSON file"""
        # Path to the test JSON file
        json_file_path = "/chatbot/tmp/PGhandbook2025.pdf.json"

        # Load the JSON file
        with open(json_file_path, "r") as f:
            parsed_json = json.load(f)

        # Create DotsOCRClient instance
        # We don't need the API credentials for this test, so we'll use dummy values
        try:
            client = DotsOCRClient()
        except ValueError:
            # If environment variables are not set, create with dummy values
            client = DotsOCRClient(api_url="http://dummy.url", auth_token="dummy_token")

        # Call the generate_tree method
        tree = client.generate_tree(parsed_json)

        # Assertions
        assert isinstance(tree, list), "Tree should be a list"
        assert len(tree) > 0, "Tree should not be empty"

        # Check structure of tree elements
        for element in tree:
            assert isinstance(element, dict), "Each tree element should be a dictionary"
            assert "page_no" in element, "Element should have page_no"
            assert "type" in element, "Element should have type"
            assert "text" in element, "Element should have text"
            assert "bbox" in element, "Element should have bbox"
            assert "level" in element, "Element should have level"
            assert "children" in element, "Element should have children"

            # Check that only Parent level elements are included
            assert element["type"] in [
                "Title",
                "Section-header",
            ], f"Only Title and Section-header should be in tree, found: {element['type']}"

            # Check that level is properly calculated for markdown headers
            if element["text"].startswith("#"):
                expected_level = len(element["text"]) - len(element["text"].lstrip("#"))
                assert (
                    element["level"] == expected_level
                ), f"Level mismatch for '{element['text']}': expected {expected_level}, got {element['level']}"

            # Check children structure recursively
            self._validate_children(element["children"])

        # Print some basic statistics for manual verification
        total_elements = self._count_tree_elements(tree)
        print(f"\nTree generation test completed:")
        print(f"- Total top-level elements: {len(tree)}")
        print(f"- Total elements (including children): {total_elements}")

        # Print tree in markdown format
        if tree:
            md_tree = client.build_tree_in_md_format(tree)
            print(f"\nTree in Markdown format:")
            print(md_tree)

    def _validate_children(self, children: List[Dict[str, Any]]) -> None:
        """Recursively validate children structure"""
        for child in children:
            assert isinstance(child, dict), "Each child should be a dictionary"
            assert "page_no" in child, "Child should have page_no"
            assert "type" in child, "Child should have type"
            assert "text" in child, "Child should have text"
            assert "bbox" in child, "Child should have bbox"
            assert "level" in child, "Child should have level"
            assert "children" in child, "Child should have children"

            # Recursively validate nested children
            self._validate_children(child["children"])

    def _count_tree_elements(self, tree: List[Dict[str, Any]]) -> int:
        """Count total number of elements in the tree including children"""
        count = len(tree)
        for element in tree:
            count += self._count_tree_elements(element["children"])
        return count

    def test_generate_tree_with_sample_data(self):
        """Test the generate_tree function with manually created sample data"""
        # Create sample data that mimics the JSON structure
        sample_json = [
            {
                "page_no": 0,
                "full_layout_info": [
                    {
                        "bbox": [93, 103, 725, 137],
                        "category": "Page-header",
                        "text": "Document Header",
                    },
                    {
                        "bbox": [93, 205, 509, 241],
                        "category": "Section-header",
                        "text": "# Main Title",
                    },
                    {
                        "bbox": [93, 307, 613, 344],
                        "category": "Section-header",
                        "text": "## Subtitle",
                    },
                    {
                        "bbox": [93, 407, 295, 443],
                        "category": "Text",
                        "text": "Some regular text content",
                    },
                    {
                        "bbox": [93, 507, 295, 543],
                        "category": "Section-header",
                        "text": "### Sub-subtitle",
                    },
                ],
            }
        ]

        # Create DotsOCRClient instance
        try:
            client = DotsOCRClient()
        except ValueError:
            client = DotsOCRClient(api_url="http://dummy.url", auth_token="dummy_token")

        # Call the generate_tree method
        tree = client.generate_tree(sample_json)

        # Should have 1 top-level element (# Main Title)
        assert len(tree) == 1, f"Expected 1 top-level element, got {len(tree)}"

        # Check the main title
        main_title = tree[0]
        assert main_title["text"] == "# Main Title"
        assert main_title["level"] == 1
        assert main_title["type"] == "Section-header"

        # Should have 1 child (## Subtitle)
        assert (
            len(main_title["children"]) == 1
        ), f"Expected 1 child, got {len(main_title['children'])}"

        subtitle = main_title["children"][0]
        assert subtitle["text"] == "## Subtitle"
        assert subtitle["level"] == 2
        assert subtitle["type"] == "Section-header"

        # Should have 1 grandchild (### Sub-subtitle)
        assert (
            len(subtitle["children"]) == 1
        ), f"Expected 1 grandchild, got {len(subtitle['children'])}"

        sub_subtitle = subtitle["children"][0]
        assert sub_subtitle["text"] == "### Sub-subtitle"
        assert sub_subtitle["level"] == 3
        assert sub_subtitle["type"] == "Section-header"

        print("\nSample data test completed successfully!")
        print(
            f"Tree structure: {main_title['text']} -> {subtitle['text']} -> {sub_subtitle['text']}"
        )

        if tree:
            md_tree = client.build_tree_in_md_format(tree)
            print(f"\nTree in Markdown format:")
            print(md_tree)
