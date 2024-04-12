from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, MaxPooling1D, Dense, concatenate
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model

def exponential_decay_fn(epoch):
    return 0.01*0.1**(epoch/300)

def create_single_stream_cnn_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape), #TODO: Look into input_shape
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(3, strides=3),

        Conv1D(128, 3, activation='relu', padding='same'),
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(3, strides=3),

        Conv1D(256, 3, activation='relu', padding='same'),
        Conv1D(256, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),

        Conv1D(512, 3, activation='relu', padding='same'),
        Conv1D(512, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),

        Conv1D(512, 3, activation='relu', padding='same'),
        Conv1D(512, 3, activation='relu', padding='same'),
        MaxPooling1D(2, strides=2),
        
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=L2(0.0001)),
        Dense(1024, activation='relu', kernel_regularizer=L2(0.0001)),
        Dense(256, activation='relu', kernel_regularizer=L2(0.0001)),
        Dense(1, activation='sigmoid')
    ])
    optimizer = SGD()
    metrics = ['accuracy', 'Precision', 'Recall']
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return model

def create_dual_stream_cnn_model(input_shape):
    # Define the inputs for each stream
    input = Input(shape=input_shape)
    
    # Stream 1
    x = Conv1D(64, 3, activation='relu', padding='same')(input)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(3, strides=3)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(3, strides=3)(x)

    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, strides=2)(x)

    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = Conv1D(512, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, strides=2)(x)
    
    # Stream 2
    y = Conv1D(64, 7, activation='relu', padding='same')(input)
    y = Conv1D(64, 7, activation='relu', padding='same')(y)
    y = MaxPooling1D(3, strides=3)(y)
    
    y = Conv1D(128, 7, activation='relu', padding='same')(y)
    y = Conv1D(128, 7, activation='relu', padding='same')(y)
    y = MaxPooling1D(3, strides=3)(y)

    y = Conv1D(256, 3, activation='relu', padding='same')(y)
    y = Conv1D(256, 3, activation='relu', padding='same')(y)
    y = MaxPooling1D(2, strides=2)(y)

    y = Conv1D(512, 3, activation='relu', padding='same')(y)
    y = Conv1D(512, 3, activation='relu', padding='same')(y)
    y = MaxPooling1D(2, strides=2)(y)
    
    y = Conv1D(512, 3, activation='relu', padding='same')(y)
    y = Conv1D(512, 3, activation='relu', padding='same')(y)
    y = MaxPooling1D(2, strides=2)(y)
    
    concatenated = concatenate([x, y])

    z = Flatten()(concatenated)
    z = Dense(1024, activation='relu', kernel_regularizer=L2(0.0001))(z)
    z = Dense(1024, activation='relu', kernel_regularizer=L2(0.0001))(z)
    z = Dense(256, activation='relu', kernel_regularizer=L2(0.0001))(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=input, outputs=z)
    optimizer = SGD()
    metrics = ['accuracy', 'Precision', 'Recall']
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    print(model.summary())
    return model
