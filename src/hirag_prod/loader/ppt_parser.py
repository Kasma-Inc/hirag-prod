import hashlib
import json
import os
import pickle
import shutil
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pptagent.induct as induct
import torch
from pptagent.llms import LLM, AsyncLLM
from pptagent.model_utils import ModelManager
from pptagent.multimodal import ImageLabler
from pptagent.presentation import Presentation
from pptagent.utils import Config, get_logger, pjoin, ppt_to_images

from hirag_prod.schema import File, FileMetadata

# Configure logger
logger = get_logger("PPTParser")


class PPTParser:
    """
    A class for parsing PowerPoint templates and analyzing their structure.

    This class handles the parsing of PowerPoint templates and performs slide induction
    to understand the template's structure and content patterns.

    Args:
        work_dir (str): Working directory for intermediate files
        async_flag (bool): set true to use AsyncLLM, default is false
    """

    def __init__(self, work_dir: str, async_flag: bool = False):
        self.work_dir: str = work_dir
        self.pptx_config: Config = Config(self.work_dir)
        self.models = ModelManager()

        api_base = os.environ.get("API_BASE", None)
        language_model_name = os.environ.get("LANGUAGE_MODEL", "gpt-4.1")
        vision_model_name = os.environ.get("VISION_MODEL", "gpt-4.1")
        text_model_name = os.environ.get("TEXT_MODEL", "text-embedding-3-small")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.language_model = (
            AsyncLLM(language_model_name, api_base)
            if async_flag
            else LLM(language_model_name, api_base)
        )
        self.vision_model = (
            AsyncLLM(vision_model_name, api_base)
            if async_flag
            else LLM(vision_model_name, api_base)
        )
        self.text_model = (
            AsyncLLM(text_model_name, api_base)
            if async_flag
            else LLM(text_model_name, api_base)
        )

        # Create necessary directories
        os.makedirs(self.work_dir, exist_ok=True)

    def get_template_files(self) -> List[File]:
        """
        Return a list of File objects, each representing a template in parsed PPT.

        Returns:
            List[File]: List of File objects, one per template in parsed PPT.
        """
        slide_induction_path = os.path.join(self.work_dir, "slide_induction.json")
        with open(slide_induction_path, "r", encoding="utf-8") as f:
            slide_induction = json.load(f)

        templates = []
        for key, template in slide_induction.items():
            if key == "functional_keys":
                continue
            template_id = template.get("template_id")
            slide_id = template.get("slides")[0]
            if template_id is None:
                continue
            # Generate a unique id using template_id and work_dir
            unique_str = f"{self.work_dir}-{template_id}"
            unique_id = hashlib.md5(unique_str.encode("utf-8")).hexdigest()
            template_obj = File(
                id=unique_id,
                page_content=json.dumps(template, ensure_ascii=False, indent=2),
                metadata=FileMetadata(
                    type="pptx",
                    page_number=slide_id,
                    uri=self.work_dir,
                    private=False,
                ),
            )
            templates.append(template_obj)
        return templates

    def _parse_pptx(self, pptx_path: str) -> Tuple[Presentation, str]:
        """
        Parse the PowerPoint template and extract slide information

        Args:
            pptx_path (str): Path to the PowerPoint template

        Returns:
            Tuple[Presentation, str]: The parsed presentation and path to slide images

        Raises:
            Exception: If PPTX parsing fails or models are not set
        """
        # Create a directory for slide images
        ppt_image_folder = pjoin(self.work_dir, "slide_images")
        os.makedirs(ppt_image_folder, exist_ok=True)
        presentation_cache_path = pjoin(self.work_dir, "presentation.pkl")

        if os.path.exists(presentation_cache_path):
            try:
                with open(presentation_cache_path, "rb") as f:
                    presentation = pickle.load(f)
                logger.info(f"√ Loaded Presentation successfully from cache")
                return presentation, ppt_image_folder
            except Exception as e:
                logger.warning(f"Cache load faild for Presentatio: {e}")

        logger.info("PowerPoint Template Parsing")

        # Copy template to the analysis directory
        template_dest = pjoin(self.work_dir, "source.pptx")
        shutil.copy(pptx_path, template_dest)
        logger.info(f"✓ Template copied to {template_dest}")

        # Parse the presentation
        presentation = Presentation.from_file(template_dest, self.pptx_config)

        logger.info(f"✓ Parsed {len(presentation)} slides from the template")

        # Convert slides to images for visual analysis
        ppt_to_images(template_dest, ppt_image_folder)
        logger.info(f"✓ Generated slide images in {ppt_image_folder}")

        # Handle any error slides
        if presentation.error_history:
            logger.warning(
                f"Found {len(presentation.error_history)} problematic slides"
            )
            for err_idx, err_msg in presentation.error_history:
                logger.warning(f"Slide {err_idx}: {err_msg}")
                err_image = pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg")
                if os.path.exists(err_image):
                    os.remove(err_image)

        # Renumber slide indices
        for i, slide in enumerate(presentation.slides, 1):
            slide.slide_idx = i
            old_path = pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg")
            new_path = pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg")
            if os.path.exists(old_path) and old_path != new_path:
                os.rename(old_path, new_path)

        logger.info("✓ Slide indices renumbered")

        # Caption images in the slides using vision model
        logger.info("Captioning images in the template slides...")
        labler = ImageLabler(presentation, self.pptx_config)
        labler.caption_images(self.vision_model)

        # Save image captions and statistics
        json.dump(
            labler.image_stats,
            open(pjoin(self.work_dir, "image_stats.json"), "w"),
            ensure_ascii=False,
            indent=4,
        )
        logger.info("✓ Image captions saved to image_stats.json")

        try:
            with open(presentation_cache_path, "wb") as f:
                pickle.dump(presentation, f)
            logger.info(f"√ Presentation cache save successfully")
        except Exception as e:
            logger.warning(f"Presentation cache save failed: {str(e)}")

        return presentation, ppt_image_folder

    def _analyze_slide_structure(
        self, presentation: Presentation, ppt_image_folder: str
    ) -> Dict[str, Any]:
        """
        Analyze the slide template and induct its structure

        Args:
            presentation (Presentation): The parsed presentation
            ppt_image_folder (str): Path to the folder containing slide images

        Returns:
            Dict[str, Any]: The slide induction results

        Raises:
            Exception: If slide induction fails or models are not set
        """

        logger.info("Slide Template Analysis")

        # Check if slide induction cache exists
        slide_induction_cache_path = pjoin(self.work_dir, "slide_induction.json")
        if os.path.exists(slide_induction_cache_path):
            try:
                with open(slide_induction_cache_path, "r", encoding="utf-8") as f:
                    slide_induction = json.load(f)
                logger.info("✓ Loaded slide induction results successfully from cache")
                return slide_induction
            except Exception as e:
                logger.warning(f"Failed to load slide induction from cache: {str(e)}")

        # Save a layout-only version of the template
        layout_template = pjoin(self.work_dir, "template.pptx")
        template_images_folder = pjoin(self.work_dir, "template_images")

        # Skip if both template and images folder already exist
        if not (
            os.path.exists(layout_template)
            and os.path.exists(template_images_folder)
            and os.listdir(template_images_folder)
        ):  # Make sure folder is not empty
            # Create layout-only template if it doesn't exist
            if not os.path.exists(layout_template):
                deepcopy(presentation).save(layout_template, layout_only=True)
                logger.info(f"✓ Layout-only template saved to: {layout_template}")
            else:
                logger.info(f"✓ Using existing layout-only template: {layout_template}")

            # Create images of the layout-only template
            os.makedirs(template_images_folder, exist_ok=True)
            ppt_to_images(layout_template, template_images_folder)
            logger.info(
                f"✓ Template layout images generated in: {template_images_folder}"
            )
        else:
            logger.info(
                f"✓ Using existing layout template and images from: {template_images_folder}"
            )

        # Induct the slide layout
        logger.info("Analyzing template layout structure...")
        slide_inducter = induct.SlideInducter(
            presentation,
            ppt_image_folder,
            template_images_folder,
            self.pptx_config,
            image_models=self.models.image_model,
            language_model=self.language_model,
            vision_model=self.vision_model,
        )

        logger.info("Performing layout induction...")
        layout_induction = slide_inducter.layout_induct()

        logger.info("Performing content induction...")
        slide_induction = slide_inducter.content_induct(layout_induction)

        # Save the slide induction results
        json.dump(
            slide_induction,
            open(slide_induction_cache_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        logger.info(f"✓ Slide induction results saved to: {slide_induction_cache_path}")
        return slide_induction

    def parse_pptx(self, pptx_path: str) -> List[File]:
        """
        Run the full PPT parsing and slide induction pipeline, then return work_dir as a File.

        Args:
            pptx_path (str): Path to the PowerPoint file.

        Returns:
            List[File]: A list with a single File object representing work_dir.
        """
        presentation, ppt_image_folder = self._parse_pptx(pptx_path)
        self._analyze_slide_structure(presentation, ppt_image_folder)
        return self.get_template_files()
