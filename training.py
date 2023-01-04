import os
import re

import matplotlib.pyplot as plt

from prepare_data.data_loader import load_data_using_keras
from tensorflow import keras
from models.models import all_models, supported_models
from vizualization.functiones_for_vizualization import confuzion_matrix_print, plot_training_histry
import json
import numpy as np
from keras_flops import get_flops
import time
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
        if lrs and lrs != 16:
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


def plot_avg_lr():
    data = {}
    with open('./results/output.json', 'r') as f:
        data = json.load(f)
    keys = sorted(data)
    lrs = ['0.01', '0.005', '0.001', '0.0005']
    models_names = ['first', 'second', 'VGG16', 'EfficientNetV2S']
    output = []
    output2 = []
    lrs_max = []
    for lr in lrs:
        x = [data[key][1] for key in keys if lr in key and 'test' in key]
        output.append(sum(x)/len(x)*100)
    for name in models_names:
        x = [((data[key][1]), key.split(' ')[1].split('_')[0]) for key in keys if name in key and 'test' in key]
        output2.append(max(x)[0]*100)
        lrs_max.append(max(x)[1])
    print(output2, lrs_max)

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(4), output, color='g')
    plt.xticks([0, 1, 2, 3], lrs)
    plt.ylabel("Średnia dokładność [%]", fontweight='bold')
    plt.xlabel("Wartośći learning rate", fontweight='bold')
    plt.ylim(95, 99.5)
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(4), output2, color='g')
    plt.xticks([0, 1, 2, 3], ['pierwszy\nlr = ' + lrs_max[0], 'drugi\nlr = ' + lrs_max[1], 'VGG16\nlr = ' + lrs_max[2],
                              'EfficientNetV2S\nlr = ' + lrs_max[3]])
    plt.ylabel("Maksymalna dokładność [%]", fontweight='bold')
    plt.xlabel("Model", fontweight='bold')
    plt.ylim(95, 99.5)
    plt.show()
    fig.savefig('./results/barploty.svg')


def print_model_summary():
    models_names = ['first', 'second', 'VGG16', 'EfficientNetV2S']
    Total = []
    Trainable = []
    Non_trainable = []
    for model_name in models_names:
        model = all_models(version=model_name, image_size=(244, 244, 1))
        model.summary(print_fn=lambda x: Total.append(x) if 'Total' in x else None)
        model.summary(print_fn=lambda x: Trainable.append(x) if 'Trainable' in x else None)
        model.summary(print_fn=lambda x: Non_trainable.append(x) if 'Non-trainable' in x else None)
    print(Total)
    print(Trainable)
    print(Non_trainable)


def get_models_flops():
    names = {
        'first': 'pierwszy',
        'second': 'drugi',
    }
    output = {}
    for model_name in supported_models:
        model = all_models(version=model_name, image_size=(224, 224, 1))

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['categorical_accuracy'])
        t0 = time.time()
        flop = round(get_flops(model, 1)/(10**9), 2)
        t1 = time.time()
        t = t1 - t0
        output[model_name] = {}
        output[model_name]['GFLOPs'] = flop
        output[model_name]['time'] = round(t, 2)
        output[model_name]['GFLOPS'] = round(flop/t, 2)

    with open('./results/flops_in_G.json', 'w') as fp:
        json.dump(output, fp)

    x = [item["GFLOPs"] for item in output.values()]
    labels = [names[item] if item in names.keys() else item for item in output.keys()]

    fig = plt.figure()
    plt.bar(np.arange(4), x, color='b')
    plt.xticks([0, 1, 2, 3], labels)
    plt.ylabel("FLOPs [G]", fontweight='bold')
    plt.xlabel("Model", fontweight='bold')
    fig.savefig('./results/GFLOPs.svg')


get_models_flops()
# print_results_of_trained_models()
# x = ['VGG16', 'VGG16_0_01', 'VGG16_0_0005', 'VGG16_0_005']
# for i in x:
#     plot_training_histry(save_history=False, name=i, path='./results', load_history=True)

# plot_avg_lr()
# print_model_summary()
