import sys
sys.path.append("../")
from one_ring.deploy.inferencer import Inferencer
import matplotlib.pyplot as plt

import time

start = time.time()

inference_config = dict(
    image_size=(512, 512),
    normalizing=True,
    model_type="tf",  # onnx or tf
    model_path="../trained_models/test_resnet50_binary_road",  # model folder path , not file
    preprocessor_type="albumentations",
    preprocessor_path="tf_seg.transformers.albumentations:get_test_transform",
    postprocessor_type="vanilla", # vanilla or None
    postprocessor_path=None,
    threshold=0.9,
    seed=48,
    device="cpu" # cpu or gpu
)

image_path = "../examples/test_images/road_2.jpg"
image = plt.imread(image_path)

inf = Inferencer(**inference_config)
pred_image, pred_mask = inf.predict(image)


plt.imshow(pred_mask[0])
plt.show()


print(f"time taken: {round(time.time() - start,3)} s")