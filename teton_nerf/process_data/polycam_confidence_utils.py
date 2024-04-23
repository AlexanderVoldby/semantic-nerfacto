from pathlib import Path
from typing import List, Tuple

from nerfstudio.process_data import process_data_utils

def process_confidence_maps(
    polycam_confidence_dir: Path,
    confidence_dir: Path,
    num_processed_images: int,
    crop_border_pixels: int = 15,
    max_dataset_size: int = 600,
    num_downscales: int = 3,
    verbose: bool = True,
) -> Tuple[List[str], List[Path]]:
    """
    Process Depth maps from polycam only

    Args:
        polycam_depth_dir: Path to the directory containing depth maps
        depth_dir: Output directory for processed depth maps
        num_processed_images: Number of RGB processed that must match the number of depth maps
        crop_border_pixels: Number of pixels to crop from each border of the image. Useful as borders may be
                            black due to undistortion.
        max_dataset_size: Max number of images to train on. If the dataset has more, images will be sampled
                         approximately evenly. If -1, use all images.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
                        will downscale the images by 2x, 4x, and 8x.
        verbose: If True, print extra logging.
    Returns:
        summary_log: Summary of the processing.
        polycam_depth_maps_filenames: List of processed depth maps paths
    """
    summary_log = []
    polycam_confidence_maps_filenames, num_orig_confidence_maps = process_data_utils.get_image_filenames(
        polycam_confidence_dir, max_dataset_size
    )

    # Copy confidence images to output directory
    copied_depth_maps_paths = process_data_utils.copy_and_upscale_polycam_depth_maps_list(
        polycam_confidence_maps_filenames,
        depth_dir=confidence_dir,
        num_downscales=num_downscales,
        crop_border_pixels=crop_border_pixels,
        verbose=verbose,
    )

    num_processed_confidence_maps = len(copied_depth_maps_paths)

    # assert same number of images as depth maps
    if num_processed_images != num_processed_confidence_maps:
        raise ValueError(
            f"Expected same amount of confidence maps as images. "
            f"Instead got {num_processed_images} images and {num_processed_confidence_maps} depth maps"
        )

    if crop_border_pixels > 0 and num_processed_confidence_maps != num_orig_confidence_maps:
        summary_log.append(f"Started with {num_processed_confidence_maps} images out of {num_orig_confidence_maps} total")
        summary_log.append(
            "To change the size of the dataset add the argument --max_dataset_size to larger than the "
            f"current value ({crop_border_pixels}), or -1 to use all images."
        )
    else:
        summary_log.append(f"Started with {num_processed_confidence_maps} images")

    return summary_log, polycam_confidence_maps_filenames