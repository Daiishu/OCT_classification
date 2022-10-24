import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_data.data_loader import load_data_using_keras
from tensorflow import keras

train_ds, val_ds, test_ds = load_data_using_keras(path='../', path_to_original_dataset="../OCT2017",
                                                  generate_new_data=True, im_size=(180, 180))
classes_names = train_ds.class_names

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 180x180 with 3 bytes colour
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=25, verbose=1)
results = model.evaluate(test_ds)

print("test loss, test acc:", results)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
# Plot accuracy
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
# Plot Loss
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation Loss')
plt.legend(loc=0)
plt.figure()
plt.show()
