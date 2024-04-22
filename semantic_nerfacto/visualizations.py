import mediapy as mp
import numpy as np
import torch
import cv2

def compare_depth_and_image(image, depth, name):
    # save the depth image and image
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    min_ = torch.where(depth == depth.min())
    max_ = torch.where(depth == depth.max())
    depth = (depth[..., None].repeat(1, 1, 3) * 255).cpu().numpy().astype(np.uint8)
    depth = cv2.circle(depth, (min_[1].item(), min_[0].item()), 10, (0, 0, 255), -1)
    cv2.circle(depth, (max_[1].item(), max_[0].item()), 10, (255, 0, 0), -1)
    mp.write_image(f"{name}_depth.png", depth)
    mp.write_image(f"{name}_image.png", image.pixel_values[0].permute(1,2,0).cpu().numpy())

def visualize_depth_before_and_after_scaling(image, lidar_depth, depth, scaled_depth, name):
    # visualize depth maps before and after scaling
    import matplotlib.pyplot as plt

    # create subfigure with three images (lidar depth, monocular depth, scaled monocular depth).
    # include a colorbar for each image.
    
    valid_mask = lidar_depth > 0

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    axs[0].imshow(lidar_depth.cpu())
    axs[0].set_title("LiDAR depth")
    axs[0].axis("off")
    plt.colorbar(axs[0].imshow(lidar_depth.cpu()), ax=axs[0])

    axs[1].imshow(depth.cpu())
    axs[1].set_title("Monocular depth")
    axs[1].axis("off")
    plt.colorbar(axs[1].imshow(depth.cpu()), ax=axs[1])

    axs[2].imshow(scaled_depth.cpu())
    axs[2].set_title("Scaled monocular depth")
    axs[2].axis("off")
    plt.colorbar(axs[2].imshow(scaled_depth.cpu()), ax=axs[2])

    # show valid mask
    axs[3].imshow(valid_mask.cpu())
    axs[3].set_title("Valid mask")
    axs[3].axis("off")
    plt.colorbar(axs[3].imshow(valid_mask.cpu()), ax=axs[3])

    # show image

    axs[4].imshow(image)
    axs[4].set_title("Image")
    axs[4].axis("off")

    # save the figure
    plt.savefig(f"{name}_alignment.png")
    plt.close()