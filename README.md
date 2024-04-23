# teton-nerf

A method that extends the nerfacto method from Nerfstudio to include semantic masks, depth supervision and patch-based regularization

## Installation
To run the methods with Nerfstudio, clone this repo, activate your nerfstudio virtual environment and cd into the folder with the pyproject.toml file.
Then  in the console run "pip install -e ." to install the method.

## Train
The two methods are called semantic-nerfacto and semantic-depth-nerfacto. To train either simply use the standard Nerfstudio CLI, ns-train <method>.

```
ns-train teton-nerf --data data/process-data/USZ-internal-med-L14/ 
```
The additions that have been to the nerfacto model can be toggled individually. The following flags are included as part of th training script

--pipeline.model.use-semantics: Whether to train and display semantic segmentations (default True)
--pipeline.model.use-depth: Whether to add regularization loss on the depth images (default True)
--pipeline.datamanager.use-monocular-depth: Whether to extend the current depth images using confidence maps and a pre-trained monocular depth model (DepthAnything) (default true) Warning: Currently only works if confidence maps are in the dataset

The following three flags are used for patch-based regularization on the three output types
pipeline.use-regnerf-depth-loss (default True)
pipeline.use-regnerf-rgb-loss (default True)
pipeline.use-regnerf-semantics-loss (default True)

Setting all of the above flags to flase will result in the nerfacto model.

```
ns-train teton-nerf --viewer.websocket-host 10.0.0.93 --viewer.websocket-port 8888 --pipeline.use-regnerf-depth-loss False --pipeline.use-regnerf-rgb-loss False --pipeline.use-regnerf-semantics-loss False --pipeline.model.use-semantics False --pipeline.model.use-depth False --data data/process-data/USZ-internal-med-L14/
```

[x] TODO: (rename semantic_depth_nerfacto to teton_nerfacto and implement semantics, depth, reglosses as flags).

## Process-data
Installing the method also install the new command ns-process-teton (better name necessary?). This works like ns-process-data, but has different configureable additions, such as the addition of semantic segmentations and confidence maps for the depth images. It is used similar to ns-process-data:

```
ns-process-teton polycam --data <Polycam un-processed dataset> --output-dir <output directory> --use_depth/--no_use_depth --use_confidence/--no_use_confidence --add_semantics/--no_add_semantics
```

The three flags are all True by default and setting them all with --no prefixed will perform the same as ns-process-data polycam.

[x] TODO: add scripts to download from google drive
[x] TODO: add script to ns-process-data after downloading the data 
[x] TODO: edit such that we also compute semantic segmentations and depth + format the depth maps


[x] TODO: make a new ns-process-data teton entry that is basically the same as the nerfstudio/nerfstudio/scripts/process_data.py [ProcessPolycam] but also processes the confidence maps

[x] TODO: put everything important in a single model class called TetonNerf something, and make everything a configurable flag. It is kinda confusion with nerf-depth, nerf-depth-seg, nerf-seg... 