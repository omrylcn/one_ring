**New TODO List 16-03-2024**
- [ ] Add Boundary Difference Over Union Loss For Medical Image Segmentation (https://arxiv.org/abs/2308.00220)
- [X] Add Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation 
- [x] Fix load method for train class
- [x] Fix custom object save
- [x] Fix onnx save method
- [ ] Add tensorrt integration
- [ ] Add openvino integration
- [ ] Integrate UNet+++ model
- [ ] Add Grad CAM++ integration
- [ ] Add basic test
- [ ] Add more notebook for training different scenarios like instance segmentation
- [ ] Test all exist models and fix bugs
- [ ] Add more backbone like mobilenet variants
- [ ] Add tensorboard callback with sample results
- [ ] Add universal class like yolo to easily use any model for prediction and training
- [ ] Enhance save object add best weights and add results images
- [ ] add Transunet
- [ ] Add other unet models
- [x] Add augmenation function logs for mlflow
- [X] Add theshold for metrics, default is 0.5

** Old TODO List **
- [ ] Integrate Segformer model (huggingface)
- [ ] Integrate Segformer model (Deepvision)
- [ ] Add directory for pretrained models
- [ ] Add training and inference examples
- [ ] Add onnx and openvino models for segformer
- [ ] Add UNet+++ training and inference notebooks
- [ ] Integrate YOLOv8 model for segmentation (Keras-cv-attention)
- [ ] Add training and inference examples for YOLOv8
- [ ] Add docstrings for sources
- [ ] Add licenses for necessary sources
- [ ] Add backbones for each necessary model
- [ ] Add doctring for each backbone
- [ ] Upload 3 different datasets to google cloud
- [ ] Build dataset classes for each dataset
- [ ] Add examples of segmentation training on each dataset
- [x] Add mlflow integration
- [x] Add evaluation functions
- [x] Add visualization functions and classes
- [x] Add overlay functionalities to easily plot
- [x] Add setup.py script for pip installation




- [ ] Add env folder

- [ ] Add tests folder

- [ ] Add poetry

- [ ] Add pytproject.toml file with linting and formatting settings

- [ ] Complete  **check_image_size** function

    ```python

    def check_image_size(config:Union[DictConfig,ListConfig], reference_part:str="augmentation"):
        "check_image_size all config file parts and provide all parts are same size"
        pass

    ```

- [ ] Complete **check_output_size** function

  ```python

    def check_output_size(config:Union[DictConfig,ListConfig], reference_part:str="augmentation"):
        "check_output_size all config file parts and provide all parts are same size"
    
        pass

    ```

- [ ]

- [ ] Add more learning rate schedulers like pytorch learning rate schedulers

  - check link : <https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#StepLR>

- [ ] Add threading  for deploy albumentation preprocessor  and onnx model, it is not running batch size > 1

- [ ] Add checking method  to control deploy transforms function and and validation data transforms function are same
