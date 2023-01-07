from prepare_data.data_loader import load_data_using_keras
from tensorflow import keras
from models.models import all_models, supported_models
from vizualization.functiones_for_vizualization import confusion_matrix_print, plot_training_history, get_models_flops,\
    save_results_of_trained_models, plot_avg_lr, save_numbers_of_params


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

    plot_training_history(history=history, save_history=True, name=log_folder_name, path='./results', load_history=False)

    confusion_matrix_print(data_set=test_ds, model=model, dst='./results', data_name='test', model_name=log_folder_name)
    confusion_matrix_print(data_set=val_ds, model=model, dst='./results', data_name='val', model_name=log_folder_name)

    model.save_weights('./results/' + log_folder_name + '/weights/latest')
