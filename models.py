from tensorflow.python.keras.layers import Input, Conv2D, Dropout, Flatten, Dense, Reshape
from tensorflow.python.keras.models import Model


def default_linear(shape=(120, 2*160)):
    img_in = Input(shape=shape, name='img_in')
    x = img_in

    x = Reshape(target_shape=shape+(1,))(x)

    x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)

    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})

    return model

