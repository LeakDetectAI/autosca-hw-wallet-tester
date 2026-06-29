import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.saving import load_model
from datasets import behavioral_datasets, wifi_csi_datasets, eeg_datasets
from deepscapy.utils import setup_random_seed

def compute_empirical_guessing_entropy_stats(X_test, y_test, model, num_classes, max_traces=30, num_simulations=100, seed=1234):
    """Empirically simulates joint multi-trace attacks and computes Guessing Entropy mean and std dev.
    
    For each trace count N from 1 to max_traces:
      - Simulates num_simulations independent attack runs.
      - In each run, for each class c, randomly samples N test traces of class c,
        aggregates their log-probabilities, and finds the rank of class c.
      - Returns the mean and standard deviation of the ranks for each N.
    """
    np.random.seed(seed)
    
    # 1. Get model predictions (probabilities) on the entire test set
    predictions = model.predict(X_test, verbose=0)
    # Add small epsilon to avoid log(0)
    log_predictions = np.log(predictions + 1e-15)
    
    # 2. Group test indices by their true class label
    class_indices = {}
    for idx, label in enumerate(y_test):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
        
    # Filter classes that actually exist in the test set
    active_classes = list(class_indices.keys())
    
    ge_means = []
    ge_stds = []
    
    # 3. Simulate Guessing Entropy evolution over trace count N
    for N in range(1, max_traces + 1):
        ranks_across_sims = []
        
        for sim in range(num_simulations):
            for c in active_classes:
                indices = class_indices[c]
                # Sample N traces with replacement if N exceeds class size
                sampled_indices = np.random.choice(indices, size=N, replace=(len(indices) < N))
                
                # Aggregate log-probabilities across the N traces
                joint_log_probs = np.sum(log_predictions[sampled_indices], axis=0)
                
                # Rank candidates (descending order of joint log-probability)
                # High probability gets rank 0
                sorted_classes = np.argsort(joint_log_probs)[::-1]
                rank_of_true_class = np.where(sorted_classes == c)[0][0]
                
                ranks_across_sims.append(rank_of_true_class)
                
        ge_means.append(float(np.mean(ranks_across_sims)))
        ge_stds.append(float(np.std(ranks_across_sims)))
        
    return ge_means, ge_stds

def main():
    seed = 1234
    try:
        setup_random_seed(seed=seed)
    except Exception:
        np.random.seed(seed)
        
    results = {}
    
    # We will evaluate all 4 domains
    datasets_to_eval = ['KEYSTROKE', 'PIN_MOTION', 'WIFI_PIN', 'EEG_PIN']
    
    # Maximum number of traces to simulate
    max_traces = 150
    num_simulations = 100
    
    print("="*70)
    print("EMPIRICAL BEHAVIORAL GUESSING ENTROPY SIMULATOR (WITH CONFIDENCE INTERVALS)")
    print("="*70)
    
    # Create output directory
    output_dir = "results/behavioral"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 12))
    
    for idx, dataset_name in enumerate(datasets_to_eval):
        print(f"\n--- Loading and preprocessing dataset: {dataset_name} ---")
        if dataset_name in behavioral_datasets:
            dataset_class = behavioral_datasets[dataset_name]
        elif dataset_name in wifi_csi_datasets:
            dataset_class = wifi_csi_datasets[dataset_name]
        elif dataset_name in eeg_datasets:
            dataset_class = eeg_datasets[dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset = dataset_class()
        dataset.load()
        dataset.split()
        dataset.preprocess()
        
        X_test, y_test = dataset.X_test, dataset.y_test
        num_classes = dataset.get_num_classes()
        
        print(f"Test Set Size: {len(X_test)} samples | Classes: {num_classes}")
        
        # Path to saved models
        baseline_path = f"trained_models/non_tuned_models/{dataset_name.lower()}_baseline.keras"
        nas_path = f"trained_models/nas_models_new/{dataset_name.lower()}_nas_random.keras"
        
        if not os.path.exists(baseline_path) or not os.path.exists(nas_path):
            print(f"Warning: Could not find trained models for {dataset_name}. Skipping.")
            continue
            
        print("Loading baseline model...")
        baseline_model = load_model(baseline_path)
        print("Loading NAS model...")
        nas_model = load_model(nas_path)
        
        # Run empirical simulations with statistics
        print("Running empirical Guessing Entropy simulation for Baseline Model...")
        baseline_ge_mean, baseline_ge_std = compute_empirical_guessing_entropy_stats(
            X_test, y_test, baseline_model, num_classes, max_traces=max_traces, num_simulations=num_simulations, seed=seed
        )
        
        print("Running empirical Guessing Entropy simulation for NAS Model...")
        nas_ge_mean, nas_ge_std = compute_empirical_guessing_entropy_stats(
            X_test, y_test, nas_model, num_classes, max_traces=max_traces, num_simulations=num_simulations, seed=seed
        )
        
        results[dataset_name] = {
            "baseline_ge_mean": baseline_ge_mean,
            "baseline_ge_std": baseline_ge_std,
            "nas_ge_mean": nas_ge_mean,
            "nas_ge_std": nas_ge_std
        }
        
        # Find exact empirical N_t (number of traces to reach Guessing Entropy < 0.05)
        baseline_nt = next((i + 1 for i, ge in enumerate(baseline_ge_mean) if ge < 0.05), ">150")
        nas_nt = next((i + 1 for i, ge in enumerate(nas_ge_mean) if ge < 0.05), ">150")
        
        print("\n" + "-"*50)
        print(f"EMPIRICAL RESULTS FOR {dataset_name}:")
        print(f"  Baseline Single-Trace Accuracy: {baseline_ge_mean[0]:.4f} Rank")
        print(f"  NAS Single-Trace Accuracy: {nas_ge_mean[0]:.4f} Rank")
        print(f"  Baseline Traces to Disclose (Nt): {baseline_nt}")
        print(f"  NAS Traces to Disclose (Nt): {nas_nt}")
        print("-"*50)
        
        # Convert to numpy arrays for calculation
        base_mean = np.array(baseline_ge_mean)
        base_std = np.array(baseline_ge_std)
        nas_mean = np.array(nas_ge_mean)
        nas_std = np.array(nas_ge_std)
        
        traces_range = range(1, max_traces + 1)
        
        # Plot curves
        plt.subplot(2, 2, idx + 1)
        
        # Plot Baseline with shaded standard deviation
        plt.plot(traces_range, base_mean, 'r-o', label=f'Baseline (Nt={baseline_nt})', linewidth=2)
        plt.fill_between(traces_range, 
                         np.maximum(0, base_mean - base_std), 
                         np.minimum(num_classes - 1, base_mean + base_std), 
                         color='red', alpha=0.15, label='Baseline ±1 SD')
        
        # Plot NAS with shaded standard deviation
        plt.plot(traces_range, nas_mean, 'g-s', label=f'NAS Model (Nt={nas_nt})', linewidth=2)
        plt.fill_between(traces_range, 
                         np.maximum(0, nas_mean - nas_std), 
                         np.minimum(num_classes - 1, nas_mean + nas_std), 
                         color='green', alpha=0.15, label='NAS Model ±1 SD')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f'{dataset_name} Guessing Entropy Evolution', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Traces (Entries / Sessions)', fontsize=10)
        plt.ylabel('Guessing Entropy (Mean Rank +/- 1 SD)', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=9, loc='upper right')
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "empirical_guessing_entropy.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nEmpirical Guessing Entropy plot saved to: {plot_path}")
    
    # Save raw results to JSON
    json_path = os.path.join(output_dir, "empirical_guessing_entropy.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Empirical Guessing Entropy raw results saved to: {json_path}")
    print("="*70)

if __name__ == '__main__':
    main()
