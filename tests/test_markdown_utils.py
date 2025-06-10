import importlib.util
from pathlib import Path
from unittest.mock import patch

# Load module without executing hirag_prod __init__
module_path = Path(__file__).resolve().parents[1] / "src/hirag_prod/markdown_utils.py"
spec = importlib.util.spec_from_file_location("markdown_utils", module_path)
markdown_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(markdown_utils)
store_markdown_images = markdown_utils.store_markdown_images


def test_store_markdown_images(tmp_path):
    md = "![](http://localhost:20926/images/sample.png)"
    image_data = b"dummy"

    def fake_open(url):
        assert url == "http://markify:20926/images/sample.png"

        class Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def read(self_inner):
                return image_data

        return Resp()

    with patch.object(markdown_utils, "urlopen", fake_open):
        out = store_markdown_images(md, str(tmp_path), base_url="http://markify:20926")

    expected_path = Path(tmp_path) / "sample.png"
    assert expected_path.exists()
    assert expected_path.read_bytes() == image_data
    expected_md = f"![]({expected_path.parent.name}/sample.png)"
    assert out == expected_md
