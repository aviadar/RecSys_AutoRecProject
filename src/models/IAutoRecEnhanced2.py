from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import tensorflow as tf


class IAutoRecEnhanced2:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('best_autorec_enhanced2_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    # es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
    # mc = ModelCheckpoint('best_autorec_enhanced_model.h5', monitor='loss', mode='min', save_best_only=True)

    def __init__(self, items_num, hidden_units=500, hidden_layer_factor=2, reg=0.0005, optimizer='Adam',
                 loss='mean_squared_error', learning_rate=0.0001, first_activation='elu', last_activation='elu'):
        self.items_num = items_num
        self.hidden_units = hidden_units
        self.hidden_layer_factor = hidden_layer_factor
        self.reg = reg
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.first_activation = first_activation
        self.last_activation = last_activation

    def build_opt_model(self):
        input_layer = Input(shape=(self.items_num,), name='item_rating')
        hidden_layer_encoder_1 = Dense(self.hidden_units * self.hidden_layer_factor * self.hidden_layer_factor,
                                       activation=self.first_activation,
                                       name='hidden_encoder_1', kernel_regularizer=regularizers.l2(self.reg))(
            input_layer)
        hidden_layer_encoder_2 = Dense(self.hidden_units * self.hidden_layer_factor, activation=self.first_activation,
                                       name='hidden_encoder_2', kernel_regularizer=regularizers.l2(self.reg))(
            hidden_layer_encoder_1)
        dense = Dense(self.hidden_units, activation=self.first_activation, name='latent_dim',
                      kernel_regularizer=regularizers.l2(self.reg))(hidden_layer_encoder_2)
        hidden_layer_decoder_1 = Dense(self.hidden_units * self.hidden_layer_factor, activation=self.last_activation,
                                       name='hidden_decoder_1', kernel_regularizer=regularizers.l2(self.reg))(dense)
        hidden_layer_decoder_2 = Dense(self.hidden_units * self.hidden_layer_factor * self.hidden_layer_factor,
                                       activation=self.last_activation,
                                       name='hidden_decoder_2', kernel_regularizer=regularizers.l2(self.reg))(
            hidden_layer_decoder_1)
        output_layer = Dense(self.items_num, activation=self.last_activation, name='item_pred_rating',
                             kernel_regularizer=regularizers.l2(self.reg))(hidden_layer_decoder_2)
        self.model = Model(input_layer, output_layer)

        if self.optimizer == 'Adam':
            # self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=IAutoRecEnhanced2.masked_rmse)
        else:
            raise Exception('Optimizer not implemented')

        self.model.summary()

    def model_builder(self, hp):
        # hp_hidden_units = hp.Int('hidden_units', min_value=100, max_value=400, step=100)
        hp_hidden_units = hp.Int('hidden_units', min_value=50, max_value=100, step=50)
        hp_hidden_layer_factor = hp.Choice('hidden_layer_factor', values=[2, 3])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
        hp_reg = hp.Choice('reg', values=[0.001, 0.0001])
        hp_first_activation = hp.Choice('first_activation', values=['elu', 'sigmoid', 'relu'])
        hp_last_activation = hp.Choice('last_activation', values=['elu', 'sigmoid', 'relu'])

        input_layer = Input(shape=(self.items_num,), name='item_rating')
        hidden_layer_encoder_1 = Dense(hp_hidden_units * hp_hidden_layer_factor * hp_hidden_layer_factor,
                                       activation=hp_first_activation,
                                       name='hidden_encoder_1', kernel_regularizer=regularizers.l2(hp_reg))(input_layer)
        hidden_layer_encoder_2 = Dense(hp_hidden_units * hp_hidden_layer_factor, activation=hp_first_activation,
                                       name='hidden_encoder_2', kernel_regularizer=regularizers.l2(hp_reg))(
            hidden_layer_encoder_1)
        dense = Dense(hp_hidden_units, activation=hp_first_activation, name='latent_dim',
                      kernel_regularizer=regularizers.l2(hp_reg))(hidden_layer_encoder_2)
        hidden_layer_decoder_1 = Dense(hp_hidden_units * hp_hidden_layer_factor, activation=hp_last_activation,
                                       name='hidden_decoder_1', kernel_regularizer=regularizers.l2(hp_reg))(dense)
        hidden_layer_decoder_2 = Dense(hp_hidden_units * hp_hidden_layer_factor * hp_hidden_layer_factor,
                                       activation=hp_last_activation,
                                       name='hidden_decoder_2', kernel_regularizer=regularizers.l2(hp_reg))(
            hidden_layer_decoder_1)
        output_layer = Dense(self.items_num, activation=hp_last_activation, name='item_pred_rating',
                             kernel_regularizer=regularizers.l2(hp_reg))(hidden_layer_decoder_2)
        self.model = Model(input_layer, output_layer)

        if self.optimizer == 'Adam':
            # self.model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=self.loss)
            self.model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=IAutoRecEnhanced2.masked_rmse)
        else:
            raise Exception('Optimizer not implemented')

        self.model.summary()
        return self.model

    @staticmethod
    def masked_rmse(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), dtype='float32')
        e = y_true - y_pred
        se = e * e
        se = se * mask
        mse = 1.0 * tf.reduce_sum(se) / tf.reduce_sum(mask)
        rmse = tf.math.sqrt(mse)
        return rmse

    def fit(self, rating_mat, rating_mat2, batch_size=256, epochs=500, verbose=2):
        self.hist = self.model.fit(x=rating_mat, y=rating_mat2,
                                   validation_split=0.1,
                                   batch_size=batch_size, epochs=epochs, verbose=verbose,
                                   callbacks=[IAutoRecEnhanced2.es, IAutoRecEnhanced2.mc])

    def predict(self, rating_mat, batch_size=512, verbose=1):
        self.predictions = self.model.predict(rating_mat,
                                              batch_size=batch_size, verbose=verbose)
