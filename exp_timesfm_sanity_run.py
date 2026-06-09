import os
import h5py
import numpy as np
from deepscapy.constants import ASCAD
from deepscapy.attacks.ascad_attack import ASCADAttack
from deepscapy.models import TimesFMBaseline

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

    print("Loading TimesFM model and initializing MLP head...")
    input_dim = x_profiling.shape[1]
    num_classes = 256
    model = TimesFMBaseline(
        model_name="sanity_timesfm_baseline",
        num_classes=num_classes,
        input_dim=input_dim,
        loss_function="categorical_crossentropy"
    )

    print("Training MLP head for 1 epoch...")
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

    print("Running attack evaluation (Guessing Entropy / Mean Rank)...")
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
