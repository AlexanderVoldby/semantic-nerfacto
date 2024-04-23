from pathlib import Path
from typing import List, Tuple
import sys
import json

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.process_data import process_data_utils
from nerfstudio.utils import io
from nerfstudio.utils.rich_utils import CONSOLE

def polycam_confidence_to_json(
    image_filenames: List[Path],
    depth_filenames: List[Path],
    confidence_filenames: List[Path],
    cameras_dir: Path,
    output_dir: Path,
    min_blur_score: float = 0.0,
    crop_border_pixels: int = 0,
) -> List[str]:
    """Convert Polycam data into a nerfstudio dataset.

    Args:
        image_filenames: List of paths to the original images.
        depth_filenames: List of paths to the original depth maps.
        confidence_filenames: List of paths to the original confidence maps
        cameras_dir: Path to the polycam cameras directory.
        output_dir: Path to the output directory.
        min_blur_score: Minimum blur score to use an image. Images below this value will be skipped.
        crop_border_pixels: Number of pixels to crop from each border of the image.

    Returns:
        Summary of the conversion.
    """
    use_depth = len(image_filenames) == len(depth_filenames)
    use_confidence = len(confidence_filenames) == len(depth_filenames)
    data = {}
    data["camera_model"] = CAMERA_MODELS["perspective"].value
    # Needs to be a string for camera_utils.auto_orient_and_center_poses
    data["orientation_override"] = "none"

    frames = []
    skipped_frames = 0
    for i, image_filename in enumerate(image_filenames):
        json_filename = cameras_dir / f"{image_filename.stem}.json"
        frame_json = io.load_from_json(json_filename)
        if "blur_score" in frame_json and frame_json["blur_score"] < min_blur_score:
            skipped_frames += 1
            continue
        frame = {}
        frame["fl_x"] = frame_json["fx"]
        frame["fl_y"] = frame_json["fy"]
        frame["cx"] = frame_json["cx"] - crop_border_pixels
        frame["cy"] = frame_json["cy"] - crop_border_pixels
        frame["w"] = frame_json["width"] - crop_border_pixels * 2
        frame["h"] = frame_json["height"] - crop_border_pixels * 2
        frame["file_path"] = f"./images/frame_{i+1:05d}{image_filename.suffix}"
        if use_depth:
            frame["depth_file_path"] = f"./depth/frame_{i+1:05d}{depth_filenames[i].suffix}"
        if use_confidence:
            frame["confidence_file_path"] = f"./confidence/frame_{i+1:05d}{confidence_filenames[i].suffix}"
        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        frame["transform_matrix"] = [
            [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
            [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
            [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append(frame)
    data["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summary = []
    if skipped_frames > 0:
        summary.append(f"Skipped {skipped_frames} frames due to low blur score.")
    summary.append(f"Final dataset is {len(image_filenames) - skipped_frames} frames.")

    if len(image_filenames) - skipped_frames == 0:
        CONSOLE.print("[bold red]No images remain after filtering, exiting")
        sys.exit(1)

    return summary

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