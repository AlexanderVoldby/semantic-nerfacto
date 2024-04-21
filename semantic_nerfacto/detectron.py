# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import torch
import os, json, cv2, random, glob
import click

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class SemanticSegmentor():
    
    def __init__(self):
    
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        
        # Reduce the number of classes the model is using
        expected_things = ["chair", "couch", "bed", "dining table", "toilet", "tv"]
        
        expected_stuff = ["blanket","curtain", "door-stuff", "floor-wood",
                          "mirror-stuff", "pillow", "shelf", "stairs",
                          "wall-brick", "table", "window", "ceiling", "floor", "wall", "rug"]
        
        # self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(expected_stuff) + len(expected_things)
        # self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).set(thing_classes=expected_things, stuff_classes=expected_stuff)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.predictor = DefaultPredictor(self.cfg)
        self.num_things = len(self.metadata.thing_classes)
        self.num_stuff = len(self.metadata.stuff_classes)
        
    def predict(self, image):
        panoptic_seg, segments_info = self.predictor(image)["panoptic_seg"]
        # We are doing semantic segmentation so we simply want to map the panoptic ID to a class ID:
        panoptic_seg = panoptic_seg.cpu().numpy()
        for info in segments_info:
            try:
                if info["isthing"]:
                    panoptic_seg[panoptic_seg == info["id"]] = info["category_id"]
                else:
                    panoptic_seg[panoptic_seg == info["id"]] = info["category_id"] + self.num_things
            except Exception as e:
                print(f"Error: {e}")
        return panoptic_seg
    
    def visualize(self, image, panoptic_segmentation, segments_info):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_segmentation.to("cpu"), segments_info)
        cv2.imshow(out.get_image()[:, :, ::-1])
        
    def save_metadata(self, data):
        metadict = {
            "thing_classes": self.metadata.thing_classes,
            "stuff_classes": self.metadata.stuff_classes,
            "thing_colors": self.metadata.thing_colors,
            "stuff_colors": self.metadata.stuff_colors,
            "thing_dataset_id_to_contiguous_id": self.metadata.thing_dataset_id_to_contiguous_id,
            "stuff_dataset_id_to_contiguous_id": self.metadata.stuff_dataset_id_to_contiguous_id
        }
        
        json_file_path = f"{data}/panoptic_classes.json"

        # Save the metadata to a JSON file
        with open(json_file_path, 'w') as f:
            json.dump(metadict, f, indent=4)

        
    def add_segmentation(self, data):
        print("Generating semantics")
        # Find a way to get these from some metadata in the dataset
        image_folder_suffixes = ['', '_2', '_4', '_8']

        for suffix in image_folder_suffixes:
            # Construct the path to the current image folder
            image_folder_path = os.path.join(data, f'images{suffix}')

            # Create a new folder for the panoptic segmentations
            segmentation_folder_path = os.path.join(data, f'segmentations{suffix}')
            if not os.path.exists(segmentation_folder_path):
                os.makedirs(segmentation_folder_path)

            # Find all image files in the current image folder
            image_files = glob.glob(os.path.join(image_folder_path, '*.*'))  # Adjust the pattern if needed
            for image_file in image_files:
                try:
                    # Load the image
                    image = cv2.imread(image_file)
                    # Perform panoptic segmentation
                    semantic_segmentation = self.predict(image)
                    # Construct the path to save the segmented image
                    base_name = os.path.basename(image_file)
                    segmentation_file_path = os.path.join(segmentation_folder_path, base_name).replace(".jpg", ".png")

                    # Save the segmented image
                    cv2.imwrite(segmentation_file_path, semantic_segmentation)
                except Exception as e:
                    print(f"Error: {e}")
        
        # Add panoptic_classes.json to dataset
        self.save_metadata(data)


@click.command()
@click.option("--data", help="Path to dataset")
def main(data):
    SS = SemanticSegmentor()
    SS.add_segmentation(data)
    

if __name__ == "__main__":
    main()