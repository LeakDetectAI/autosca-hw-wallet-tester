from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation
from keras.models import Model
from keras.optimizers import RMSprop

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import ONED_CNN


class BehavioralCNNBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="glorot_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        # input_dim is expected to be a tuple (timesteps, channels)
        super(BehavioralCNNBaseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                                   model_type=ONED_CNN, loss_function=loss_function,
                                                   kernel_regularizer=kernel_regularizer,
                                                   kernel_initializer=kernel_initializer, optimizer=optimizer,
                                                   metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def reshape_inputs(self, X, y):
        # Overridden to skip flattening of traces, preserving shape (timesteps, channels)
        if y is not None:
            from keras.utils import to_categorical
            y = to_categorical(y, num_classes=self.num_classes)
        return X, y

    def _construct_model_(self, **kwargs):
        # Minimal 1D CNN for shape (timesteps, channels)
        trace_input = Input(shape=self.input_dim, dtype='float32')
        # Block 1
        x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1', **kwargs)(trace_input)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        # Block 2
        x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu', name='fc1', **kwargs)(x)
        scores = Dense(self.num_classes, activation=None, name='scores', kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('softmax', name='predictions')(scores)

        model = Model(inputs=trace_input, outputs=predictions, name='behavioral_cnn_baseline')
        scoring_model = Model(inputs=trace_input, outputs=scores, name='behavioral_cnn_baseline_scorer')
        return model, scoring_model
