from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Input, Flatten
import numpy as np


class MF:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('best_mf_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    def __init__(self, users_num, items_num, latent_dim=20, optimizer='Adam', loss='mean_squared_error',
                 learning_rate=0.01):
        self.users_num = users_num
        self.items_num = items_num
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim

    def build_opt_model(self):
        user_input = Input(shape=(1,), name='user_input', dtype='int32')
        item_input = Input(shape=(1,), name='item_input', dtype='int32')

        user_embedding = Embedding(input_dim=self.users_num, output_dim=self.latent_dim, name='user_embedding')(
            user_input)
        item_embedding = Embedding(input_dim=self.items_num, output_dim=self.latent_dim, name='item_embedding')(
            item_input)

        user_latent = Flatten()(user_embedding)
        item_latent = Flatten()(item_embedding)

        pred = keras.layers.dot([user_latent, item_latent], axes=1, normalize=False)
        self.model = Model(inputs=[user_input, item_input], outputs=pred)

        if self.optimizer == 'Adam':
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
        else:
            raise Exception('Optimizer not implemented')
        self.model.summary()

    def model_builder(self, hp):
        user_input = Input(shape=(1,), name='user_input', dtype='int32')
        item_input = Input(shape=(1,), name='item_input', dtype='int32')

        hp_latent_dim = hp.Int('latent_dim', min_value=8, max_value=40, step=8)

        user_embedding = Embedding(input_dim=self.users_num, output_dim=hp_latent_dim, name='user_embedding')(
            user_input)
        item_embedding = Embedding(input_dim=self.items_num, output_dim=hp_latent_dim, name='item_embedding')(
            item_input)

        user_latent = Flatten()(user_embedding)
        item_latent = Flatten()(item_embedding)

        pred = keras.layers.dot([user_latent, item_latent], axes=1, normalize=False)
        self.model = Model(inputs=[user_input, item_input], outputs=pred)

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        if self.optimizer == 'Adam':
            self.model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=self.loss)
        else:
            raise Exception('Optimizer not implemented')
        self.model.summary()
        return self.model

    def fit(self, user_input, item_input, labels, batch_size=8168, epochs=100, verbose=1):
        self.hist = self.model.fit([np.array(user_input), np.array(item_input)],  # input
                                   np.array(labels),  # labels
                                   validation_split=0.1,
                                   batch_size=batch_size, epochs=epochs, verbose=verbose,
                                   callbacks=[MF.es, MF.mc])

    def predict(self, user_input, item_input, batch_size=8168, verbose=1):
        self.predictions = self.model.predict([np.array(user_input), np.array(item_input)],
                                              batch_size=batch_size, verbose=verbose)
