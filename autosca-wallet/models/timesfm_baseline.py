import os
import numpy as np
import torch
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras.optimizers import RMSprop

from deepscapy.core.sca_nn_model import SCANNModel
from deepscapy.constants import MLP

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
