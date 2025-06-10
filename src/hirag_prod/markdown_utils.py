import os
import re
from typing import Optional
from urllib.request import urlopen

IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)")


def store_markdown_images(
    markdown: str, image_dir: str, base_url: Optional[str] = None
) -> str:
    """Download images referenced in markdown and update the links.

    Parameters
    ----------
    markdown : str
        Markdown text containing image links.
    image_dir : str
        Directory where images will be stored.
    base_url : Optional[str]
        If provided, replace the scheme/host part of each image URL with this
        base URL before downloading.

    Returns
    -------
    str
        Updated markdown with local image paths.
    """
    os.makedirs(image_dir, exist_ok=True)
    updated = markdown

    for match in IMAGE_PATTERN.finditer(markdown):
        alt = match.group("alt")
        url = match.group("url")
        if base_url and url.startswith("http://localhost:20926"):
            url = url.replace("http://localhost:20926", base_url, 1)
        filename = os.path.basename(url.split("?")[0])
        local_path = os.path.join(image_dir, filename)
        with urlopen(url) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        updated_link = (
            f"![{alt}]({os.path.join(os.path.basename(image_dir), filename)})"
        )
        updated = updated.replace(match.group(0), updated_link)

    return updated
