import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing

from generate_TF import GenerateTF, get_freq
from scipy.optimize import curve_fit
from prettytable import PrettyTable

from pytorchClassifiers import get_keras_nn, plot_history

avg_pool1d = keras.layers.AveragePooling1D



# %%
# Load the data
df = pd.read_pickle('./data/tf-ampl-response-82000-noise0.1.pkl')
# df.head()

# %%
# Extract the target variables
phase = df.pop('phase')
gain = df.pop('gain')

# All fmax, np should be equal
fmax = df.iloc[0].fmax
NP = df.iloc[0].np
df.drop(columns=['fmax', 'np'], inplace=True)

# %%
# target_orig is the vector with the originale phase, gain labels
target_orig = np.array((phase, gain), dtype=np.float32).T

target_scaler = preprocessing.StandardScaler().fit(target_orig)
# target is scaled, better for training
target = target_scaler.transform(target_orig)
# target = tf.convert_to_tensor(target, dtype=tf.uint8)


# %%
# our dataset is 3D
values = np.zeros((len(df), len(df.iloc[0].real), 3), dtype=np.float32)
print(values.shape)
for index, row in df.iterrows():
    # print(row)
    values[index, :, 0] = row.real
    values[index, :, 1] = row.imag
    values[index, :, 2] = row.amplitude
data = values

# %%
# Split in train and test
X_train_orig, X_test_orig, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
    data, target, target_orig, test_size=0.2, random_state=0)

# further divide X_test in test + validate
X_test_orig, X_validate_orig, y_test, y_validate, y_test_orig, y_validate_orig = \
    train_test_split(X_test_orig, y_test, y_test_orig, test_size=0.4, random_state=1)

X_train = X_train_orig[:, :, :2]
X_test = X_test_orig[:, :, :2]
X_validate = X_validate_orig[:, :, :2]

print(X_train.shape)
print(X_test.shape)
print(X_validate.shape)
print(y_train.shape)
print(y_test.shape)
print(y_validate.shape)


# %%
def phase_model_builder(hp):
    # n_inputs, n_outputs, name='phase_regressor', activation='relu',
    #              layers=[140, 110], kernel_size=5, stride=5, trim_edges=120,
    #              dropout=0.0, with_norm=False):
    # initialize model
    n_inputs = X_train.shape[1:]
    n_outputs = 1
    name = 'keras_phase_reg'
    # Define HyperParameter Space
    activation = hp.Choise('activate', values=['relu', 'gelu'])
    with_norm = hp.Boolean('norm')
    kernel_size = hp.Int('pool', min_value=3, max_value=10, step=1)
    stride = kernel_size
    trim_edges = hp.Int('crop', min_value=100, max_value=160, step=5)
    dropout = hp.Fixed('dropout', value=0.2)
    layer1 = hp.Fixed('fc1', min_value=400, max_value=600, step=20)
    layer2 = hp.Fixed('fc2', min_value=100, max_value=300, step=20)
    layers = [layer1, layer2]

    lr = hp.Choice('learning_rate', values=[0.001, 0.005, 0.01])
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Model architecture
    model = keras.models.Sequential(name=name)
    # just set input shape
    model.add(keras.Input(shape=n_inputs))
    # crop the edges
    model.add(keras.layers.Cropping1D(cropping=trim_edges, name='Crop'))
    # smoothing layer
    model.add(keras.layers.AveragePooling1D(pool_size=kernel_size, strides=stride,
                                            name='AvgPool'))
    # normalize input
    model.add(keras.layers.BatchNormalization(axis=2, name='Norm_0'))
    # flatten before the dense layers
    model.add(keras.layers.Flatten(name='Flat'))
    # for every layer
    for i, layer_size in enumerate(layers):
        # Add dense layer with activation_func
        model.add(keras.layers.Dense(layer_size, activation=activation,
                                     name=f'Dense_{i+1}'))
        # Add batch normalization
        if with_norm:
            model.add(keras.layers.LayerNormalization(
                axis=1, name=f'Norm_{i+1}'))
        # Add dropout
        if dropout > 0 and dropout < 1:
            model.add(keras.layers.Dropout(dropout, name=f'Dropout_{i+1}'))
    model.add(keras.layers.Dense(n_outputs, name=f'Output'))

    # Now compile the model
    model.compile(optimizer=optimizer, loss="mse")

    return model


# %%
import keras_tuner as kt

phase_tuner = kt.Hyperband(phase_model_builder,
                           objective='mse',
                           max_epochs=40,
                           factor=3,
                           directory='keras_phase_hyperparam',
                           project_name='otfb_lhc_ml')
                           
stop_early1 = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=5, restore_best_weights=True)


# %%
# train the network

phase_tuner.search(X_train, y_train[:, 0],
                   validation_data=(X_test, y_test[:, 0]),
                   epochs=20, callbacks=[stop_early1])

# Get the optimal hyperparameters
best_hps = phase_tuner.get_best_hyperparameters(num_trials=1)[0]


# %% [markdown]
# ## Below the definition of the gain prediction model

# %%
# n_inputs = X_train.shape[1:]
# n_outputs = 1

# model2 = get_keras_nn(n_inputs, n_outputs, name='gain_regressor', activation='gelu',
#                       layers=[420, 350], kernel_size=4, stride=4, trim_edges=120,
#                       with_norm=True)
# print(model2.summary())

# model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
#                loss="mse")

# stop_early2 = keras.callbacks.EarlyStopping(monitor='val_loss',
#                                             patience=5, restore_best_weights=True)
# save_best2 = keras.callbacks.ModelCheckpoint(filepath='models/keras/regression/gain',
#                                              monitor='val_loss', save_best_only=True)


# %%
# # train the network
# gain_history = model2.fit(X_train, y_train[:, 1],
#                            validation_data=(X_test, y_test[:, 1]),
#                            epochs=20, callbacks=[stop_early2, save_best2])


# %%
# load the models
# model1 = keras.models.load_model('models/keras/regression/phase_best')
# model2 = keras.models.load_model('models/keras/regression/gain_best')


# %%
# def curve_fit_deluxe(func, freq, sample, trim_edges=0, kernel_size=1, stride=1, **kwargs):
#     # center crop sample
#     if trim_edges > 0:
#         freq, sample = freq[trim_edges:-trim_edges], sample[trim_edges:-trim_edges]
#     # prepare the shapes for avg_pooling
#     freq = freq.reshape(1, -1, 1)
#     sample = sample.reshape(1, -1, 1)
#     # perform average pooling
#     freq = avg_pool1d(pool_size=kernel_size, strides=stride)(freq).numpy().flatten()
#     sample = avg_pool1d(pool_size=kernel_size, strides=stride)(sample).numpy().flatten()
#     # pass to curve_fit
#     return curve_fit(func, freq, sample, **kwargs)

# %%
# # Get curve fit predictions
# gen_tf = GenerateTF(fb_attn_index=3, with_noise=False)
# freq = gen_tf.frequency.astype(np.float32)
# y_optimizer = []
# for sample in X_validate_orig[:, :, 2]:
#     popt, _ = curve_fit_deluxe(gen_tf, freq, sample, trim_edges=130, kernel_size=4, stride=1,
#                                bounds=([-20, 1e-4], [20, 1e-2]), method='trf')
#     y_optimizer.append(popt)
# y_optimizer = np.array(y_optimizer)


# %%
# Get model's predictions
# y_nn_phase = model1.predict(X_validate).flatten()
# y_nn_gain = model2.predict(X_validate).flatten()
# y_nn = np.array([y_nn_phase, y_nn_gain]).T


# %%
# # Convert from category to value
# y_nn_descaled = target_scaler.inverse_transform(y_nn)
# y_nn_phase_descaled = y_nn_descaled[:, 0]
# y_nn_gain_descaled = y_nn_descaled[:, 1]

# phase_loss = model1.evaluate(X_validate, y_validate[:, 0])
# gain_loss = model2.evaluate(X_validate, y_validate[:, 1])


# %%
# from sklearn.metrics import r2_score, mean_squared_error

# r2_nn = r2_score(y_validate_orig, y_nn_descaled,
#                    multioutput='raw_values')
# mse_nn = mean_squared_error(y_validate_orig, y_nn_descaled,
#                                multioutput='raw_values')

# r2_opt = r2_score(y_validate_orig, y_optimizer,
#                   multioutput='raw_values')
# mse_opt = mean_squared_error(y_validate_orig, y_optimizer,
#                               multioutput='raw_values')

# print('R2\tPhase\tGain')
# print('NeuralNet: ', r2_nn)
# print('Optimizer:', r2_opt)

# print('MSE\tPhase\tGain')
# print('NeuralNet: ', mse_nn)
# print('Optimizer:', mse_opt)



# %%
# plt.figure(figsize=(7, 6))

# table = PrettyTable()
# table.field_names = ["idx", "param", "original", "NN", "Opt"]

# gen_tf = GenerateTF(fb_attn_index=3, with_noise=False)
# freq = gen_tf.frequency.astype(np.float32)

# for idx in np.random.choice(np.arange(0, len(X_validate)), size=3):
#     try:
#         popt, _ = curve_fit_deluxe(gen_tf, freq, X_validate_orig[idx, :, 2], trim_edges=130, kernel_size=4, stride=1,
#                                    bounds=([-20, 1e-4], [20, 1e-2]), method='trf')
#     except:
#         print(f'Scipy curve fit failed for idx: {idx}')
#         continue

#     table.add_row([idx, 'phase', y_validate_orig[idx]
#                   [0], y_nn_descaled[idx][0], popt[0]])
#     table.add_row([idx, 'gain', y_validate_orig[idx][1],
#                   y_nn_descaled[idx][1], popt[1]])

#     p = plt.plot(
#         freq, gen_tf(freq, *(y_validate_orig[idx])), label=f'real_{idx}', ls='-')
#     plt.plot(freq, gen_tf(
#         freq, *(y_nn_descaled[idx])), label=f'NN_{idx}', ls='--', color=p[0].get_color())
#     plt.plot(freq, gen_tf(freq, *popt),
#              label=f'opt_{idx}', ls=':', color=p[0].get_color())
#     # plt.plot(x, gen_tf(x, *poptModel), label=f'opt+model_{idx}', ls='-.', color=p[0].get_color())
# print(table)
# plt.legend(ncol=3)
# plt.tight_layout()


# %%



