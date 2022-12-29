import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import pickle
import os


def confuzion_matrix_print(data_set, model, dst='../results', data_name=None, test_name=None):
    if test_name:
        dst = dst + '/' + test_name

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


def plot_training_histry(history=None, save_history=False, path='../results',
                         load_history=False, name=None):
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
