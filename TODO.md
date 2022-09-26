
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
