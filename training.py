from prepare_data.data_loader import load_data_using_keras
from tensorflow import keras
from models.models import all_models
from vizualization.functiones_for_vizualization import confuzion_matrix_print, plot_training_histry

size = 224

train_ds, val_ds, test_ds = load_data_using_keras(path='../', path_to_original_dataset="../OCT2017",
                                                  generate_new_data=False, im_size=(size, size), val_size=200,
                                                  batch_size=32)

model = all_models(version='EfficientNetV2S', image_size=(size, size, 1))

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=100, verbose=1, )
results = model.evaluate(test_ds)

print("test loss, test acc:", results)

plot_training_histry(history=history, save_history=True, name='EfficientNetV2S', path='./results', load_history=False)

confuzion_matrix_print(data_set=test_ds, model=model, dst='./results', data_name='test', test_name='EfficientNetV2S')
confuzion_matrix_print(data_set=val_ds, model=model, dst='./results', data_name='val', test_name='EfficientNetV2S')

model.save_weights('./results/EfficientNetV2S/weights/')
# model.save('./results/EfficientNetV2S/model')
