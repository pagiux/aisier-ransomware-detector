import tensorflow as tf


# build the model
def build_model(is_train):
    n_inputs = 6
    n_hidden = [12, 24, 12]
    dropout = 0.4
    activation = tf.nn.relu
    out_activation = tf.nn.softmax

    original_inputs = tf.keras.Input(shape=(n_inputs,), name='model_input')
    x = tf.keras.layers.Dense(n_hidden[0], activation=activation, name='0_hidden')(original_inputs)
    x = tf.keras.layers.Dropout(dropout, name='0_dropout')(x)

    for i, n_neurons in enumerate(n_hidden[1:]):
        # setup hidden layers
        x = tf.keras.layers.Dense(n_neurons, activation=activation, name='%d_hidden' % (i + 1))(x)

        # add dropout
        if is_train:
            x = tf.keras.layers.Dropout(dropout, name='%d_dropout' % (i + 1))(x)

    # setup output layer
    output = tf.keras.layers.Dense(2, activation=out_activation, name='model_output')(x)

    model = tf.keras.Model(inputs=original_inputs, outputs=output, name="model")

    return model


# define training strategy
def train_model(model, dataset):
    loss = 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    metrics = ['accuracy']

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model_path_pattern = 'checkpoint.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path_pattern,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    return model.fit(dataset.X_train, dataset.Y_train,
                     batch_size=32,
                     epochs=50,
                     verbose=2,
                     validation_data=(dataset.X_val, dataset.Y_val),
                     callbacks=[model_early_stop_callback, model_checkpoint_callback])