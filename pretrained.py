import tensorflow as tf
import tensorflow_hub as hub


model = tf.keras.applications.EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224,224,1),
    pooling=None,
    classes=4,
    classifier_activation="softmax",
)