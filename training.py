import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_data.data_loader import load_data_using_keras
from tensorflow import keras
from models.models import all_models

train_ds, val_ds, test_ds = load_data_using_keras(path='../', path_to_original_dataset="../OCT2017",
                                                  generate_new_data=True, im_size=(180, 180))
classes_names = train_ds.class_names

model = all_models(version='first', image_size=(180, 180, 1))

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
