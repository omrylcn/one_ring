import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input,EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

tf.get_logger().setLevel(level="ERROR")


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        

# Parametreler
BATCH_SIZE = 4
IMAGE_SIZE = (512,512)
AUTOTUNE = tf.data.AUTOTUNE

#tfds.disable_progress_bar()

# Veri setini yükle
data, dataset_info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train_dataset = data['train']
num_classes = dataset_info.features['label'].num_classes
train_steps_per_epoch = dataset_info.splits['train'].num_examples // BATCH_SIZE

# Veri setini ön işleme fonksiyonu
def preprocess_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)  # EfficientNet için ön işleme
    return image, label

# Veri artırımı fonksiyonu
def augment_image(image, label):
   # image = tf.image.random_flip_left_right(image)
   # image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

# Veri setini ön işle ve karıştır
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Modeli oluştur
base_model = EfficientNetB1(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Modelin alt katmanlarını dondur

# fine-tuning
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

history = model.fit(train_dataset, epochs=10, steps_per_epoch=train_steps_per_epoch)