from typing import Tuple
#from transformers import TFSegformerForSemanticSegmentation
from tensorflow.keras.models import Model

from one_ring.base import ModelBuilder


class SegFormer(ModelBuilder):
    """
    Initializes the model builder.

    Parameters
    ----------
    input_shape : Tuple
        Shape of the input image.
    output_size : int
        The size of the output, typically this would be the number of classes for segmentation.
    backbone : str
        Which model variant to use , defaults to "nvidia/mit-b0".


        Methods
        -------
        build_model(self)
            Builds and returns the model.

    Notes
    -----
    SegFormer Article: https://arxiv.org/abs/2105.15203

    Huggingface SegFormer Documentation: https://huggingface.co/docs/transformers/main/model_doc/segformer

    """

    def __init__(
        self,
        output_size: int,
        input_shape: Tuple = (512, 512, 3),
        backbone_name: str = "nvidia/mit-b0",
    ) -> None:

        self.output_size = output_size
        self.input_shape = input_shape
        self.backbone_name = backbone_name

    def build_model(self, label_names: list) -> Model:
        """Builds the model.

        Returns
        -------
        Model : tf.keras.model.Model
            The SegFormer model.
        """
        id2label = {i: label_names[i] for i in range(self.output_size)}
        label2id = {label_names[i]: i for i in range(self.output_size)}
        model = TFSegformerForSemanticSegmentation(
            self.backbone_name,
            num_labels=self.output_size,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        return model
