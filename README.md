# semantic-nerfacto

A method that extends the nerfacto method from Nerfstudio to include semantic masksand various regularizers, such as a depth prior and patch-based regularization

## Installation
To run the methods with Nerfstudio, clone this repo, activate your nerfstudio virtual environment and cd into the folder with the pyproject.toml file.
Then  in the console run "pip install -e ." to install the method.

## Train
The two methods are called semantic-nerfacto and semantic-depth-nerfacto. To train either simply use the standard Nerfstudio CLI, ns-train <method>.

```
ns-train semantic-depth-nerfacto --data data/process-data/USZ-internal-med-L14/
```

semantic-nerfacto is a simple nerfacto-extension that includes semantic segmentation. It currently requires that you have added semantic segmentation to your dataset before training.
Implementing automatic segmentation is a work-in-progress

semantic-depth-nerfacto also inculdes a depth-prior and patch-based regularization. You can toggle the patch-based regularization by using the flags --use-regnerf-rgb-loss/--use-regnerf-depth-los/--use-regnerf-semantics-loss True/False
semantic-depth-nerfacto currently requires sparse lidar depth maps. It extends these with a monocular depth model to create dense depth maps.

## Process-data

TODO: add scripts to download from google drive
TODO: add script to ns-process-data after downloading the data 
TODO: edit such that we also compute semantic segmentations and depth + format the depth maps


```
ns-process-data polycam --data data/zip/USZ-internal-med-L14.zip --output-dir data/process-data/USZ-internal-med-L14
```

```
python semantic_nerfacto/detectron.py --data data/process-data/USZ-internal-med-L14
```
