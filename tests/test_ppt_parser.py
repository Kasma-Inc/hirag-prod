import os

import pytest

from hirag_prod.loader.ppt_parser import PPTParser


@pytest.fixture
def test_file_path():
    """Get the path to the test PPT file."""
    return os.path.join(os.path.dirname(__file__), "Beamer.pptx")


@pytest.fixture
def work_dir():
    """Get the working directory for test output."""
    work_dir = os.path.join(os.path.dirname(__file__), "ppt_templates", "Beamer")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def test_ppt_parser(test_file_path, work_dir):
    """Test the basic functionality of PPTParser."""
    parser = PPTParser(work_dir)

    presentation, ppt_image_folder = parser.parse_pptx(test_file_path)

    assert presentation is not None
    assert os.path.exists(ppt_image_folder)
    assert len(os.listdir(ppt_image_folder)) > 0

    slide_induction = parser.analyze_slide_structure(presentation, ppt_image_folder)

    assert slide_induction is not None
    assert os.path.exists(os.path.join(work_dir, "slide_induction.json"))
