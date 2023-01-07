import os
import re
import json
import time
import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras_flops import get_flops
from models.models import all_models, supported_models
from prepare_data.data_loader import load_data_using_keras


def confusion_matrix_print(data_set, model, dst='../results', data_name=None, model_name=None):
    """
    Function for saving confusion matrix in svg and png

    :param data_set: Data set
    :param model: Compiled model
    :param dst: destination to save matrix
    :param data_name: name of dataset
    :param model_name: name of model
    :return: confusion matrix
    """
    if model_name:
        dst = dst + '/' + model_name

    true_labels = []
    predicted_labels = []

    for image, label in data_set:
        true_labels.append(np.argmax(label, axis=- 1))
        preds = model.predict(image, verbose=0)
        predicted_labels.append(np.argmax(preds, axis=- 1))

    true_labels_tensor = tf.concat([item for item in true_labels], axis=0)
    predicted_labels_tensor = tf.concat([item for item in predicted_labels], axis=0)

    con_matrix = tf.math.confusion_matrix(true_labels_tensor, predicted_labels_tensor)

    classes_names = data_set.class_names

    fig = plt.figure()
    sns.heatmap(con_matrix, annot=True, cmap=sns.color_palette("viridis"), fmt='g',
                xticklabels=classes_names, yticklabels=classes_names)
    plt.show()
    if not os.path.exists(dst):
        os.mkdir(dst)
    fig.savefig(dst + '/conf_' + data_name + '.svg')
    fig.savefig(dst + '/conf_' + data_name + '.png')

    return con_matrix


def plot_training_history(history=None, save_history=False, path='../results',
                          load_history=False, name=None):
    """
    Function for saving plot of model training history. And print number of epoch.

    :param history: Model history
    :param save_history: Path to existing model history
    :param path: Path to destination history or saved history
    :param load_history: True if history was saved before and read this history
    :param name: model name
    :return:
    """
    if history:
        history = history.history
    if name:
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(path + '/' + name):
            os.mkdir(path + '/' + name)
        path = path + '/' + name + '/trainig_history'
    else:
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/trainig_history'
    if save_history:
        with open(path, 'wb') as file_pi:
            pickle.dump(history, file_pi)
    if load_history:
        with open(path, "rb") as file_pi:
            history = pickle.load(file_pi)

    acc = history['categorical_accuracy']
    val_acc = history['val_categorical_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    if name:
        print(name + " epok: " + str(len(acc)))

    fig1 = plt.figure()
    plt.plot(epochs, acc, 'r', label='Zbiór treningowy')
    plt.plot(epochs, val_acc, 'b', label='Zbiór walidacyjny')
    plt.title('Dokładność zbioru treningowego i walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend(loc=0)
    plt.show()
    fig1.savefig(path[:-16] + '/acc.svg')

    fig2 = plt.figure()
    plt.plot(epochs, loss, 'r', label='Zbiór treningowy')
    plt.plot(epochs, val_loss, 'b', label='Zbiór walidacyjny')
    plt.title('Funkcja straty zbioru treningowego i walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Koszt')
    plt.legend(loc=0)
    plt.show()
    fig2.savefig(path[:-16] + '/loss.svg')


def save_results_of_trained_models():
    """
    Save json with results of each model and each learning rate for testing and validation dataset

    :return: None
    """
    train_ds, val_ds, test_ds = load_data_using_keras(path='../../', path_to_original_dataset="../../OCT2017",
                                                      generate_new_data=False, im_size=(224, 224),
                                                      val_size=200, batch_size=32)
    output = {}
    for i in os.listdir('../results'):
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
        plt.savefig('/'.join(['../results', i, 'neg.svg']))
    with open('../results/output.json', 'w') as fp:
        json.dump(output, fp)


def plot_avg_lr():
    """
    Save plot average accuracy for each value of learning rate and maximum accuracy for each model.

    :return: None
    """
    data = {}
    with open('../results/output.json', 'r') as f:
        data = json.load(f)
    keys = sorted(data)
    lrs = ['0.01', '0.005', '0.001', '0.0005']
    models_names = supported_models
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
    fig.savefig('../results/barploty.svg')


def save_numbers_of_params():
    """
    Save in json number of parameters in each model
    :return: None
    """
    models_names = supported_models
    Total = []
    Trainable = []
    Non_trainable = []
    output = {}
    for model_name in models_names:
        model = all_models(version=model_name, image_size=(244, 244, 1))
        model.summary(print_fn=lambda x: Total.append(x) if 'Total' in x else None)
        model.summary(print_fn=lambda x: Trainable.append(x) if 'Trainable' in x else None)
        model.summary(print_fn=lambda x: Non_trainable.append(x) if 'Non-trainable' in x else None)
        output[model_name] = {}
        output[model_name]['Total'] = Total[-1]
        output[model_name]['Trainable'] = Trainable[-1]
        output[model_name]['Non_trainable'] = Non_trainable[-1]

    with open('../results/number_of_params.json', 'w') as fp:
        json.dump(output, fp)


def get_models_flops():
    """
    Save all models flops in json and bar_plot

    :return: None
    """
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

    with open('../results/flops_in_G.json', 'w') as fp:
        json.dump(output, fp)

    x = [item["GFLOPs"] for item in output.values()]
    labels = [names[item] if item in names.keys() else item for item in output.keys()]

    fig = plt.figure()
    plt.bar(np.arange(4), x, color='b')
    plt.xticks([0, 1, 2, 3], labels)
    plt.ylabel("FLOPs [G]", fontweight='bold')
    plt.xlabel("Model", fontweight='bold')
    fig.savefig('../results/GFLOPs.svg')


def save_barplot_for_dataset_training():
    path = "../../OCT2017"
    n_dict = {}
    CNV = []
    DME = []
    DRUSEN = []
    NORMAL = []
    labels = []
    for i in os.listdir(path):
        if '.' not in i:
            n_dict[i] = {}
            labels.append(i)
            for j in os.listdir('/'.join([path, i])):
                if '.' not in j:
                    list = os.listdir('/'.join([path, i, j]))
                    new_list = [x for x in list if '.D' not in x]
                    n_dict[i].update({j: len(new_list)})
                    if j == "CNV":
                        CNV.append(len(new_list))
                    elif j == "DME":
                        DME.append(len(new_list))
                    elif j == "DRUSEN":
                        DRUSEN.append(len(new_list))
                    else:
                        NORMAL.append(len(new_list))

    fig = plt.figure()
    plt.bar(np.arange(4), n_dict['train'].values(), color='r')
    plt.xticks([0, 1, 2, 3], n_dict['train'].keys())
    plt.ylabel("Liczba obrazów", fontweight='bold')
    plt.xlabel("Klasy", fontweight='bold')
    fig.savefig('../results/training_set.svg')
