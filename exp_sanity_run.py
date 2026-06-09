import os
import h5py
import numpy as np
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, Activation
from keras.models import Model
from keras.optimizers import RMSprop

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import ONED_CNN, ASCAD
from deepscapy.attacks.ascad_attack import ASCADAttack

# Model Code from ASCAD Paper Github Repository (https://github.com/ANSSI-FR/ASCAD)
class ASCADCNNBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="glorot_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):
        super(ASCADCNNBaseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                               model_type=ONED_CNN, loss_function=loss_function,
                                               kernel_regularizer=kernel_regularizer,
                                               kernel_initializer=kernel_initializer, optimizer=optimizer,
                                               metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        # From VGG16 design
        input_shape = (self.input_dim, 1)
        trace_input = Input(shape=input_shape, dtype='float32')
        # Block 1
        x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1', **kwargs)(trace_input)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        # Block 2
        x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
        # Block 3
        x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
        # Block 4
        x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
        # Block 5
        x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1', **kwargs)(x)
        x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1', **kwargs)(x)
        x = Dense(4096, activation='relu', name='fc2', **kwargs)(x)
        scores = Dense(self.num_classes, activation=None, name='scores', kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('softmax', name='predictions')(scores)

        # Create model.
        model = Model(inputs=trace_input, outputs=predictions, name='cnn_baseline')
        # model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics)
        scoring_model = Model(inputs=trace_input, outputs=scores, name='cnn_baseline_scorer')
        return model, scoring_model

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        return super().fit(X=X, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super().predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)

def load_ascad_subset(dataset_path, num_profiling_traces=200, num_attack_traces=100):
    with h5py.File(dataset_path, "r") as in_file:
        x_profiling = np.array(in_file['Profiling_traces/traces'][:num_profiling_traces], dtype=np.float32)
        y_profiling = np.array(in_file['Profiling_traces/labels'][:num_profiling_traces], dtype=int)
        
        x_attack = np.array(in_file['Attack_traces/traces'][:num_attack_traces], dtype=np.float32)
        y_attack = np.array(in_file['Attack_traces/labels'][:num_attack_traces], dtype=int)
        
        # Load metadata to get plaintext and key for the attack
        attack_metadata = in_file['Attack_traces/metadata'][:num_attack_traces]
        plaintext_attack = attack_metadata['plaintext']
        key_attack = attack_metadata['key'][0] # Assuming fixed key for ASCAD_f
        
    return (x_profiling, y_profiling), (x_attack, y_attack), plaintext_attack, key_attack

def run_sanity_check():
    print("======  Initializing Canonical Sanity Profile Run ======")
    
    dataset_path = os.path.join("datasets", "ASCAD.h5")
    if not os.path.exists(dataset_path):
        print(f" Error: Dataset not found at {dataset_path}")
        return

    print(" Slicing dataset traces for quick environment verification...")
    (x_profiling, y_profiling), (x_attack, y_attack), plaintext_attack, key_attack = load_ascad_subset(
        dataset_path, 
        num_profiling_traces=200, 
        num_attack_traces=100
    )

    print("Compiling baseline CNN attack architecture...")
    input_dim = x_profiling.shape[1]
    num_classes = 256
    model = ASCADCNNBaseline(
        model_name="sanity_ascad_cnn",
        num_classes=num_classes,
        input_dim=input_dim,
        loss_function="categorical_crossentropy"
    )

    print(" Executing a single training pass (Epoch = 1)...")
    model.fit(
        x_profiling, 
        y_profiling, 
        epochs=1, 
        batch_size=50, 
        verbose=1
    )

    weight_output = "sanity_model.weights.h5"
    model.model.save_weights(weight_output)
    print(f"Sanity model weights successfully stored to: {weight_output}")

    print(" Evaluating physical information leakage matrices...")
    predictions = model.predict_scores(x_attack)
    
    attack = ASCADAttack(
        model_name="sanity_ascad_cnn",
        model_class=ASCADCNNBaseline,
        loss_name="categorical_crossentropy",
        num_attacks=100,
        total_attack_traces=100,
        dataset_type=ASCAD,
        real_key=key_attack,
        byte=2, # ASCAD usually attacks byte 2
        plaintext_ciphertext=plaintext_attack,
        n_folds=1
    )
    
    attack_results = attack._perform_attacks_(predictions, plaintext_attack, None)
    print(f" Sanity pipeline completed. Attack Results (Mean Rank): {attack_results[-1]}")

if __name__ == "__main__":
    run_sanity_check()
