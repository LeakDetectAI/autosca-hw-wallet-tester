from keras.layers import Flatten, Dense, Input, Conv2D, AveragePooling2D, Activation, Conv1D, AveragePooling1D
from keras.models import Model
from keras.optimizers import RMSprop

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import ONED_CNN, TWOD_CNN_SQR, TWOD_CNN_RECT


class CustomLeNet5(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(CustomLeNet5, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                           model_type=ONED_CNN, loss_function=loss_function,
                                           kernel_regularizer=kernel_regularizer,
                                           kernel_initializer=kernel_initializer, optimizer=optimizer,
                                           metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim, 1)
        trace_input = Input(shape=input_shape, dtype='float32')
        x = Conv1D(6, 5, activation='relu', padding='same', name='block1_conv1', **kwargs)(trace_input)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        x = Conv1D(16, 5, activation='relu', padding='same', name='block2_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(120, activation='relu', name='fc1', **kwargs)(x)
        x = Dense(84, activation='relu', name='fc2', **kwargs)(x)
        scores = Dense(self.num_classes, activation=None, name='scores', kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('softmax', name='predictions')(scores)
        model = Model(inputs=trace_input, outputs=predictions, name='lenet5')
        scoring_model = Model(inputs=trace_input, outputs=scores, name='lenet5_scorer')
        return model, scoring_model


class CustomLeNet5Rectangle(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(CustomLeNet5Rectangle, self).__init__(model_name=model_name, num_classes=num_classes,
                                                    input_dim=input_dim, model_type=TWOD_CNN_RECT,
                                                    loss_function=loss_function,
                                                    kernel_regularizer=kernel_regularizer,
                                                    kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                    metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim[0], self.input_dim[1], 1)
        trace_input = Input(shape=input_shape, dtype='float32')
        x = Conv2D(6, (5, 5), activation='relu', padding='same', name='block1_conv1', **kwargs)(trace_input)
        x = AveragePooling2D((2, 2), strides=2, name='block1_pool')(x)
        x = Conv2D(16, (5, 5), activation='relu', padding='same', name='block2_conv1', **kwargs)(x)
        x = AveragePooling2D((2, 2), strides=2, name='block2_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(120, activation='relu', name='fc1', **kwargs)(x)
        x = Dense(84, activation='relu', name='fc2', **kwargs)(x)
        scores = Dense(self.num_classes, activation=None, name='scores', kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('softmax', name='predictions')(scores)
        model = Model(inputs=trace_input, outputs=predictions, name='lenet5_rect')
        scoring_model = Model(inputs=trace_input, outputs=scores, name='lenet5_rect_scorer')
        return model, scoring_model


class CustomLeNet5Square(CustomLeNet5Rectangle):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(CustomLeNet5Square, self).__init__(model_name=model_name, num_classes=num_classes,
                                                  input_dim=input_dim, loss_function=loss_function,
                                                  kernel_regularizer=kernel_regularizer,
                                                  kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                  metrics=metrics, weight_averaging=weight_averaging, **kwargs)
        self.model_type = TWOD_CNN_SQR
