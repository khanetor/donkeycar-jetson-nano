from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, Reshape, BatchNormalization, Cropping2D
from tensorflow.python.keras.models import Model

from donkeycar.parts.keras import KerasPilot


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


def linear_dropout(shape=(120, 2*160)):
    drop = 0.1

    img_in = Input(shape=shape, name='img_in')
    x = img_in

    x = Reshape(target_shape=shape+(1,))(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = Dropout(drop)(x)

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


def linear_cropped_dropout(shape=(120, 2*160)):
    drop = 0.1

    img_in = Input(shape=shape, name='img_in')
    x = img_in

    x = Reshape(target_shape=shape+(1,))(x)
    x = Cropping2D(cropping=((40, 0), (0, 0)))(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = Dropout(drop)(x)

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


class KerasRNN_LSTM(KerasPilot):
    def __init__(self, image_w =2*160, image_h=120, seq_length=2, num_outputs=2, *args, **kwargs):
        super(KerasRNN_LSTM, self).__init__(*args, **kwargs)
        image_shape = (image_h, image_w)
        self.model = rnn_lstm(seq_length=seq_length,
            num_outputs=num_outputs,
            image_shape=image_shape)
        self.seq_length = seq_length
        self.image_w = image_w
        self.image_h = image_h
        self.img_seq = []
        self.optimizer = "rmsprop"
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                  loss='mse')

    def run(self, img_arr):
        if img_arr.shape[2] == 3 and self.image_d == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)
        
        img_arr = np.array(self.img_seq).reshape(1, self.seq_length, self.image_h, self.image_w)
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle


def rnn_lstm(seq_length=2, num_outputs=2, image_shape=(120,2*160)):

    from tensorflow.python.keras.layers.merge import concatenate
    from tensorflow.python.keras.layers import LSTM
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD

    drop_out = 0.3

    img_seq_shape = (seq_length,) + image_shape
    img_in = Input(batch_shape=img_seq_shape, name='img_in')

    x = Sequential()
    x.add(TD(Reshape(target_shape=image_shape + (1,)), input_shape=img_seq_shape))
    x.add(TD(Cropping2D(cropping=((40,0), (0,0))), input_shape=img_seq_shape )) 
    x.add(TD(BatchNormalization()))
    x.add(TD(Conv2D(24, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Conv2D(32, (5,5), strides=(2,2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Conv2D(32, (3,3), strides=(2,2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Conv2D(32, (3,3), strides=(1,1), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(MaxPool2D(pool_size=(2, 2))))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(drop_out)))

    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_out"))
    x.add(Dropout(.1))
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(64, activation='relu'))
    x.add(Dense(10, activation='relu'))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))

    return x
