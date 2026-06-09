import os
import h5py
import numpy as np
import torch
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras.optimizers import RMSprop

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import MLP, ASCAD
from deepscapy.attacks.ascad_attack import ASCADAttack

class TimesFMBaseline(SCANNModel):
    def __init__(self, model_name, num_classes, input_dim, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="glorot_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, checkpoint_path='google/timesfm-2.5-200m-pytorch', **kwargs):
        self.embedding_dim = 1280
        self.checkpoint_path = checkpoint_path
        
        # Load timesfm
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tfm = TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint_path, torch_compile=False)
        self.tfm.model.to(self.device)
        self.tfm.model.eval()
        
        # Call SCANNModel constructor
        super(TimesFMBaseline, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=input_dim,
                                             model_type=MLP, loss_function=loss_function,
                                             kernel_regularizer=kernel_regularizer,
                                             kernel_initializer=kernel_initializer, optimizer=optimizer,
                                             metrics=metrics, weight_averaging=weight_averaging, **kwargs)

    def _construct_model_(self, **kwargs):
        inputs = Input(shape=(self.embedding_dim,))
        x = Dense(512, activation='relu', **kwargs)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', **kwargs)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        scores = Dense(self.num_classes, activation=None, name='scores', **kwargs)(x)
        predictions = Activation('softmax', name='predictions')(scores)
        
        model = Model(inputs=inputs, outputs=predictions, name='timesfm_mlp_head')
        scoring_model = Model(inputs=inputs, outputs=scores, name='timesfm_mlp_head_scorer')
        return model, scoring_model

    def _extract_features(self, X, batch_size=256):
        num_traces = X.shape[0]
        pad_len = 0
        if self.input_dim % 32 != 0:
            pad_len = 32 - (self.input_dim % 32)
        
        if pad_len > 0:
            X_padded = np.pad(X, ((0, 0), (0, pad_len)), mode='edge')
        else:
            X_padded = X
            
        num_patches = X_padded.shape[1] // 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, num_traces, batch_size):
                batch_X = X_padded[i : i + batch_size]
                b_size = batch_X.shape[0]
                
                inputs = torch.tensor(batch_X, dtype=torch.float32, device=self.device).reshape(b_size, num_patches, 32)
                masks = torch.zeros((b_size, num_patches, 32), dtype=torch.bool, device=self.device)
                
                outputs, _ = self.tfm.model(inputs, masks)
                _, output_emb, _, _ = outputs
                # Mean pool over patch dimension
                pooled = torch.mean(output_emb, dim=1).cpu().numpy()
                all_embeddings.append(pooled)
                
        return np.concatenate(all_embeddings, axis=0)

    def reshape_inputs(self, X, y):
        # Extract TimesFM embeddings
        embeddings = self._extract_features(X)
        
        from keras.utils import to_categorical
        if y is not None:
            if len(y.shape) == 1:
                y = to_categorical(y, num_classes=self.num_classes)
        return embeddings, y

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        return super(TimesFMBaseline, self).fit(X=X, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def predict_scores(self, X, verbose=0, **kwargs):
        return super(TimesFMBaseline, self).predict_scores(X, verbose, **kwargs)

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super(TimesFMBaseline, self).evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super(TimesFMBaseline, self).summary(**kwargs)

    def load_head(self, filepath):
        self.model = load_model(filepath)
        self.scoring_model = self.model

    @classmethod
    def load_custom_model(cls, filepath):
        if not os.path.exists(filepath):
            base, ext = os.path.splitext(filepath)
            for new_ext in ['.keras', '.tf', '.weights.h5']:
                if os.path.exists(base + new_ext):
                    filepath = base + new_ext
                    break
        mlp_model = load_model(filepath)
        instance = cls(
            model_name="loaded_timesfm",
            num_classes=mlp_model.output_shape[-1],
            input_dim=700
        )
        instance.model = mlp_model
        instance.scoring_model = mlp_model
        return instance

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
    print("======  Initializing TimesFM Sanity Profile Run ======")
    
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

    print(" Compiling baseline TimesFM attack architecture...")
    input_dim = x_profiling.shape[1]
    num_classes = 256
    model = TimesFMBaseline(
        model_name="sanity_timesfm_baseline",
        num_classes=num_classes,
        input_dim=input_dim,
        loss_function="categorical_crossentropy"
    )

    print(" Executing a single training pass (Epoch = 1) on MLP head...")
    model.fit(
        x_profiling, 
        y_profiling, 
        epochs=1, 
        batch_size=50, 
        verbose=1
    )

    weight_output = "sanity_timesfm_model.weights.h5"
    model.model.save_weights(weight_output)
    print(f" Sanity model weights successfully stored to: {weight_output}")

    print(" Evaluating physical information leakage matrices...")
    # Predict
    predictions = model.predict_scores(x_attack)
    
    attack = ASCADAttack(
        model_name="sanity_timesfm_baseline",
        model_class=TimesFMBaseline,
        loss_name="categorical_crossentropy",
        num_attacks=100,
        total_attack_traces=100,
        dataset_type=ASCAD,
        real_key=key_attack,
        byte=2, # ASCAD usually attacks byte 2
        plaintext_ciphertext=plaintext_attack,
        n_folds=1,
        extension="keras"
    )
    
    attack_results = attack._perform_attacks_(predictions, plaintext_attack, None)
    print(f" TimesFM Sanity pipeline completed. Attack Results (Mean Rank): {attack_results[-1]}")

if __name__ == "__main__":
    run_sanity_check()
