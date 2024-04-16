# semantic-nerfacto
A method that extends the nerfacto method from Nerfstudio to include semantic masksand various regularizers, such as a depth prior and patch-based regularization

To run the methods with Nerfstudio, clone this repo, activate your nerfstudio virtual environment and cd into the folder with the pyproject.toml file.
Then  in the console run "pip install -e ." to install the method.

The two methods are called semantic-nerfacto and semantic-depth-nerfacto. To train either simply use the standard Nerfstudio CLI, ns-train <method>.

semantic-nerfacto is a simple nerfacto-extension that includes semantic segmentation. It currently requires that you have added semantic segmentation to your dataset before training.
Implementing automatic segmentation is a work-in-progress

semantic-depth-nerfacto also inculdes a depth-prior and patch-based regularization. You can toggle the patch-based regularization by using the flags --use-regnerf-rgb-loss/--use-regnerf-depth-los/--use-regnerf-semantics-loss True/False
semantic-depth-nerfacto currently requires sparse lidar depth maps. It extends these with a monocular depth model to create dense depth maps.
