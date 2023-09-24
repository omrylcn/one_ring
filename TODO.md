- [ ] Integrate Segformer model (huggingface)
- [ ] Integrate Segformer model (Deepvision)
- [ ] Add directory for pretrained models
- [ ] Add training and inference examples
- [ ] Add onnx and openvino models for segformer
- [ ] Integrate UNet+++ model
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
- [ ] Add mlflow integration
- [ ] Add onnx integration for inference
- [ ] Add openvino integration for inference
- [ ] Add evaluation functions
- [ ] Add visualization functions and classes
- [ ] Add overlay functionalities to easily plot
- [ ] Add setup.py script for pip installation
- [ ] Add Grad CAM++ integration




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
