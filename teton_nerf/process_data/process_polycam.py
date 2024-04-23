import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import tyro
from typing_extensions import Annotated

from nerfstudio.process_data import polycam_utils

from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE
from teton_nerf.process_data.polycam_confidence_utils import process_confidence_maps, polycam_confidence_to_json


@dataclass
class ProcessPolycamConfidence(BaseConverterToNerfstudioDataset):
    """Process Polycam data into a nerfstudio dataset.

    To capture data, use the Polycam app on an iPhone or iPad with LiDAR. The capture must be in LiDAR or ROOM mode.
    Developer mode must be enabled in the app settings, this will enable a raw data export option in the export menus.
    The exported data folder is used as the input to this script.

    This script does the following:

    1. Scales images to a specified size.
    2. Converts Polycam poses into the nerfstudio format.
    """

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    use_uncorrected_images: bool = False
    """If True, use the raw images from the polycam export. If False, use the corrected images."""
    max_dataset_size: int = 600
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""
    min_blur_score: float = 25
    """Minimum blur score to use an image. If the blur score is below this value, the image will be skipped."""
    crop_border_pixels: int = 15
    """Number of pixels to crop from each border of the image. Useful as borders may be black due to undistortion."""
    use_depth: bool = False
    """If True, processes the generated depth maps from Polycam"""
    use_confidence: bool = False
    """If True, processes the generated confidence maps for the depth maps"""
    add_semantics: bool = False
    """If True, adds semantic segmentation to the dataset using pretrained detectron2 model"""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        if self.data.suffix == ".zip":
            with zipfile.ZipFile(self.data, "r") as zip_ref:
                zip_ref.extractall(self.output_dir)
                extracted_folder = zip_ref.namelist()[0].split("/")[0]
            self.data = self.output_dir / extracted_folder

        if (self.data / "keyframes" / "corrected_images").exists() and not self.use_uncorrected_images:
            polycam_image_dir = self.data / "keyframes" / "corrected_images"
            polycam_cameras_dir = self.data / "keyframes" / "corrected_cameras"
        else:
            polycam_image_dir = self.data / "keyframes" / "images"
            polycam_cameras_dir = self.data / "keyframes" / "cameras"
            if not self.use_uncorrected_images:
                CONSOLE.print("[bold yellow]Corrected images not found, using raw images.")

        if not polycam_image_dir.exists():
            raise ValueError(f"Image directory {polycam_image_dir} doesn't exist")

        if not (self.data / "keyframes" / "depth").exists():
            confidence_dir = self.data / "keyframes" / "depth"
            raise ValueError(f"Depth map directory {confidence_dir} doesn't exist")
        
        if not (self.data / "keyframes" / "confidence").exists() and self.use_confidence:
            confidence_dir = self.data / "keyframes" / "confidence"
            raise ValueError(f"Confidence map directory {confidence_dir} doesn't exist")

        (image_processing_log, polycam_image_filenames) = polycam_utils.process_images(
            polycam_image_dir,
            image_dir,
            crop_border_pixels=self.crop_border_pixels,
            max_dataset_size=self.max_dataset_size,
            num_downscales=self.num_downscales,
            verbose=self.verbose,
        )

        summary_log.extend(image_processing_log)

        polycam_depth_filenames = []
        if self.use_depth:
            polycam_depth_image_dir = self.data / "keyframes" / "depth"
            confidence_dir = self.output_dir / "depth"
            confidence_dir.mkdir(parents=True, exist_ok=True)
            (depth_processing_log, polycam_depth_filenames) = polycam_utils.process_depth_maps(
                polycam_depth_image_dir,
                confidence_dir,
                num_processed_images=len(polycam_image_filenames),
                crop_border_pixels=self.crop_border_pixels,
                max_dataset_size=self.max_dataset_size,
                num_downscales=self.num_downscales,
                verbose=self.verbose,
            )
            summary_log.extend(depth_processing_log)
        
        polycam_confidence_filenames = []
        if self.use_confidence:
            polycam_depth_image_dir = self.data / "keyframes" / "confidence"
            confidence_dir = self.output_dir / "confidence"
            confidence_dir.mkdir(parents=True, exist_ok=True)
            (confidence_processing_log, polycam_confidence_filenames) = process_confidence_maps(
                polycam_depth_image_dir,
                confidence_dir,
                num_processed_images=len(polycam_image_filenames),
                crop_border_pixels=self.crop_border_pixels,
                max_dataset_size=self.max_dataset_size,
                num_downscales=self.num_downscales,
                verbose=self.verbose,
            )
            summary_log.extend(confidence_processing_log)
            
        if self.add_semantics:
            from teton_nerf.processing_tools.detectron import SemanticSegmentor
            SS = SemanticSegmentor()
            segmentation_filenames = SS.add_segmentation(self.output_dir)
            
        summary_log.extend(
            polycam_confidence_to_json(
                image_filenames=polycam_image_filenames,
                depth_filenames=polycam_depth_filenames,
                confidence_filenames=polycam_confidence_filenames,
                segmentation_filenames=segmentation_filenames,
                cameras_dir=polycam_cameras_dir,
                output_dir=self.output_dir,
                min_blur_score=self.min_blur_score,
                crop_border_pixels=self.crop_border_pixels,
            )
        )

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()

@dataclass
class NotInstalled:
    def main(self) -> None:
        ...
       
Commands = Union[
    Annotated[ProcessPolycamConfidence, tyro.conf.subcommand(name="polycam")]
]

# Add aria subcommand if projectaria_tools is installed.
try:
    import projectaria_tools
except ImportError:
    projectaria_tools = None

if projectaria_tools is not None:
    from nerfstudio.scripts.datasets.process_project_aria import ProcessProjectAria

    # Note that Union[A, Union[B, C]] == Union[A, B, C].
    Commands = Union[Commands, Annotated[ProcessProjectAria, tyro.conf.subcommand(name="aria")]]
else:
    Commands = Union[
        Commands,
        Annotated[
            NotInstalled,
            tyro.conf.subcommand(
                name="aria",
                description="**Not installed.** Processing Project Aria data requires `pip install projectaria_tools'[all]'`.",
            ),
        ],
    ]
        
def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + e.args[0])


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # type: ignore