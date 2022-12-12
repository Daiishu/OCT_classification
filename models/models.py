import tensorflow as tf
from tensorflow import keras


def all_models(version='first', image_size=(256, 256, 1)):
    """
    Available model:
    - first

    :param version: type of model
    :param image_size: size of image
    :return: model
    """
    available_versions = ['first']
    if version not in available_versions:
        raise ValueError('Model not available')
    if version == 'first':
        model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 180x180 with 3 bytes colour
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=image_size),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(4, activation='sigmoid')
        ])
        return model
