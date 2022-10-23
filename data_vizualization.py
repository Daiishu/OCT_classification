import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_data.data_loader import load_data_using_keras
from prepare_data.folder_creation import remove_all_folders_with_name

remove_all_folders_with_name(path='../')
train_ds, val_ds, test_ds = load_data_using_keras(path='../', path_to_original_dataset="../OCT2017")
classes_names = train_ds.class_names

plt.figure(figsize=(10, 20))
for images, labels in train_ds.take(1):
    for i in range(32):
        ax = plt.subplot(8, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        plt.title(classes_names[int(tf.argmax(labels[i]))])
        plt.axis("off")
    plt.show()
