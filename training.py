import os
import re

import matplotlib.pyplot as plt

from prepare_data.data_loader import load_data_using_keras
from tensorflow import keras
from models.models import all_models
from vizualization.functiones_for_vizualization import confuzion_matrix_print, plot_training_histry
import json
import numpy as np
import tensorflow as tf


def training_data(size_im=224, generate_new_data=False, model_name='first', lr=0.001, log_folder_name=None,
                  patience=3):
    if log_folder_name is None:
        log_folder_name = model_name

    train_ds, val_ds, test_ds = load_data_using_keras(path='../', path_to_original_dataset="../OCT2017",
                                                      generate_new_data=generate_new_data, im_size=(size_im, size_im),
                                                      val_size=200, batch_size=32)

    model = all_models(version=model_name, image_size=(size_im, size_im, 1))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['categorical_accuracy'])

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, verbose=1, callbacks=[callback])
    results = model.evaluate(test_ds)

    print("test loss, test acc:", results)

    plot_training_histry(history=history, save_history=True, name=log_folder_name, path='./results', load_history=False)

    confuzion_matrix_print(data_set=test_ds, model=model, dst='./results', data_name='test', test_name=log_folder_name)
    confuzion_matrix_print(data_set=val_ds, model=model, dst='./results', data_name='val', test_name=log_folder_name)

    model.save_weights('./results/' + log_folder_name + '/weights/latest')


def print_results_of_trained_models():
    train_ds, val_ds, test_ds = load_data_using_keras(path='../', path_to_original_dataset="../OCT2017",
                                                      generate_new_data=False, im_size=(224, 224),
                                                      val_size=200, batch_size=32)
    output = {}
    for i in os.listdir('./results'):
        if '.' in i:
            continue
        path = os.path.join('./results', i, 'weights', 'latest')
        model_name = re.search(r'(.*)(?=\_0\_|\s)', i, flags=re.MULTILINE)
        lrs = re.search(r'\d+$', i, flags=re.MULTILINE)
        if lrs:
            lr = float('0.' + lrs.group(0))
        else:
            lr = 0.001
        if model_name is None:
            model_name = i
        else:
            model_name = model_name.group(0)
        model = all_models(version=model_name, image_size=(224, 224, 1))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['categorical_accuracy'])
        model.load_weights(path)
        output[model_name + ' ' + str(lr) + '_test'] = model.evaluate(test_ds)
        output[model_name + ' ' + str(lr) + '_val'] = model.evaluate(val_ds)
        classes_names = test_ds.class_names
        plt.figure(figsize=(20, 10))
        k = 0
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            predicted_labels = np.argmax(preds, axis=- 1)
            true_labels = np.argmax(labels, axis=- 1)
            if k >= 8:
                break
            for j in range(len(predicted_labels)):
                if k >= 8:
                    break
                if predicted_labels[j] != true_labels[j]:
                    ax = plt.subplot(2, 4, k + 1)
                    plt.imshow(images[j].numpy().astype("uint8"), cmap='gray')
                    plt.title('T: ' + classes_names[true_labels[j]] + ' P: ' + classes_names[predicted_labels[j]])
                    plt.axis("off")
                    k += 1
        plt.savefig('/'.join(['./results', i, 'neg.svg']))
    with open('./results/output.json', 'w') as fp:
        json.dump(output, fp)


print_results_of_trained_models()
# x = ['second', 'second_0_01', 'second_0_0005', 'second_0_005', 'first', 'first_0_01', 'first_0_0005', 'first_0_005', ]
# for i in x:
#     plot_training_histry(save_history=False, name=i, path='./results', load_history=True)
