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
    available_versions = ['first', 'EfficientNetV2M', 'EfficientNetV2S', 'second', 'VGG16']
    if version not in available_versions:
        raise ValueError('Model not available')
    if version == 'first':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=image_size),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(4, activation='sigmoid')
        ])
    if version == 'second':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=image_size),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='sigmoid')
        ])
    if version == 'VGG16':
        model = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size,
            pooling=None,
            classes=4,
            classifier_activation="softmax",
        )
    if version == 'EfficientNetV2S':
        model = tf.keras.applications.EfficientNetV2S(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size,
            pooling=None,
            classes=4,
            classifier_activation="softmax",
            include_preprocessing=True,
        )
    if version == 'EfficientNetV2M':
        model = tf.keras.applications.EfficientNetV2M(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size,
            pooling=None,
            classes=4,
            classifier_activation="softmax",
            include_preprocessing=True,
        )
    return model
