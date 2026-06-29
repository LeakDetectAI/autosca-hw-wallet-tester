import argparse
import os
import time
import csv
import sys
import json
import types
from datetime import timedelta
import numpy as np
from keras.saving import load_model

# Import core deepscapy and dataset files
from deepscapy.constants import ONED_CNN
from deepscapy.utils import setup_random_seed, create_dir_recursively
from datasets import behavioral_datasets, wifi_csi_datasets, eeg_datasets
from datasets.behavioral.base import DatasetMissingError
from deepscapy.models.behavioral_cnn_baseline import BehavioralCNNBaseline
from deepscapy.models.nas_models.nas_advanced import NASAdvanced
from classification_metrics import evaluate_classification_performance
from exp_behavioral_entropy import compute_empirical_guessing_entropy_stats

def behavioral_reshape_inputs(self, X, y):
    """Overridden reshape_inputs for behavioral traces to keep 3D shape (samples, timesteps, channels)
    and one-hot encode target labels.
    """
    if y is not None:
        from keras.utils import to_categorical
        y = to_categorical(y, num_classes=self.num_classes)
    return X, y

def main():
        parser = argparse.ArgumentParser(description='Behavioral Trace Classification Experiment Runner')
        parser.add_argument('--domain', type=str, required=False, default='BEHAVIORAL', choices=['BEHAVIORAL', 'POWER_EM', 'MICROARCHITECTURAL', 'NETWORK_TRAFFIC', 'WIFI_CSI', 'EEG'],
                            help='Domain of the side-channel attack.')
        parser.add_argument('--dataset', type=str, required=True,
                            help='Name of dataset to evaluate.')
        parser.add_argument('--search', type=str, default='random', choices=['random', 'greedy', 'hyperband', 'bayesian'],
                            help='Search strategy/tuner type for NAS model.')
        parser.add_argument('--epochs', type=int, default=30,
                            help='Total training epochs (used for both baseline and NAS).')
        parser.add_argument('--batch_size', type=int, default=100,
                            help='Batch size for training.')
        parser.add_argument('--loss_function', type=str, default='categorical_crossentropy',
                            help='Loss function name.')
        parser.add_argument('--output_dir', type=str, default='results/behavioral/',
                            help='Output directory to save metrics.')
        parser.add_argument('--seed', type=int, default=1234,
                            help='Random seed for reproducibility.')
    
        args = parser.parse_args()
        seed = args.seed
        
        # Make results reproducible
        try:
            setup_random_seed(seed=seed)
        except RuntimeError as e:
            print(f"Warning: could not set random seed parallelism configs: {e}")
            # Fallback to setting basic seeds manually
            import random
            import numpy as np
            import tensorflow as tf
            os.environ['PYTHONHASHSEED'] = str(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
        print("="*60)
        # 1. Load dataset with missing check based on domain
        try:
            if args.domain == 'BEHAVIORAL':
                if args.dataset not in behavioral_datasets:
                    raise DatasetMissingError(f"Dataset {args.dataset} not found in BEHAVIORAL domain.")
                dataset_class = behavioral_datasets[args.dataset]
                dataset = dataset_class()
                dataset.load()
            elif args.domain == 'WIFI_CSI':
                if args.dataset not in wifi_csi_datasets:
                    raise DatasetMissingError(f"Dataset {args.dataset} not found in WIFI_CSI domain.")
                dataset_class = wifi_csi_datasets[args.dataset]
                dataset = dataset_class()
                dataset.load()
            elif args.domain == 'EEG':
                if args.dataset not in eeg_datasets:
                    raise DatasetMissingError(f"Dataset {args.dataset} not found in EEG domain.")
                dataset_class = eeg_datasets[args.dataset]
                dataset = dataset_class()
                dataset.load()
            else:
                print(f"Domain {args.domain} loaders are currently mocked/not implemented.")
                print(f"Please implement {args.domain} dataset loaders to proceed.")
                sys.exit(0)
        except DatasetMissingError as e:
            print(e)
            print("\nSkipping execution because dataset is not present.")
            sys.exit(0)
    
        dataset.split()
        dataset.preprocess()
    
        X_train, X_val, X_test = dataset.X_train, dataset.X_val, dataset.X_test
        y_train, y_val, y_test = dataset.y_train, dataset.y_val, dataset.y_test
    
        input_dim = dataset.get_input_shape() # This will be (timesteps, channels)
        num_classes = dataset.get_num_classes()
    
        print(f"Train Shape: {X_train.shape} | Val Shape: {X_val.shape} | Test Shape: {X_test.shape}")
        print(f"Input dimension (timesteps, channels): {input_dim} | Number of classes: {num_classes}")
    
        # 2. Train baseline 1D CNN model (BehavioralCNNBaseline)
        print("\n--- Training Behavioral 1D CNN Baseline ---")
        baseline_model_name = f"{args.dataset.lower()}_baseline"
        baseline = BehavioralCNNBaseline(
            model_name=baseline_model_name,
            num_classes=num_classes,
            input_dim=input_dim,
            loss_function=args.loss_function
        )
        
        start_time = time.time()
        baseline.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
        baseline_train_time = time.time() - start_time
        print(f"Baseline train time: {timedelta(seconds=int(baseline_train_time))}")
    
        # Evaluate Baseline
        print("Evaluating Baseline Predictions (Guessing Entropy takes precedence over Accuracy)...")
        y_pred_scores_base = baseline.predict_scores(X_test)
        base_metrics = evaluate_classification_performance(
            y_test, y_pred_scores_base, args.dataset, "Baseline", output_dir=args.output_dir
        )
    
        # 3. Train NAS model
        print("\n--- Training AutoKeras NAS Model ---")
        nas_model_name = f"{args.dataset.lower()}_nas_{args.search}"
        
        # Configure epochs for AutoKeras tuners
        if args.search in ['random', 'bayesian']:
            nas_search_epochs = min(20, max(1, args.epochs // 2))
            nas_final_epochs = max(1, args.epochs - nas_search_epochs)
        elif args.search == 'greedy':
            nas_search_epochs = min(50, max(1, args.epochs // 2))
            nas_final_epochs = max(1, args.epochs - nas_search_epochs)
        else: # hyperband
            nas_search_epochs = args.epochs
            nas_final_epochs = args.epochs
    
        # Input dim for NAS constructor: since it checks input_dim dynamically, we pass the total number of features
        # but we monkeypatch reshape_inputs so it preserves 3D traces (timesteps, channels)
        nas = NASAdvanced(
            model_name=nas_model_name,
            num_classes=num_classes,
            input_dim=input_dim[0] * input_dim[1], # dummy flat features for constructor
            dataset=args.dataset,
            reshape_type=ONED_CNN,
            loss_function=args.loss_function,
            loss_function_name=args.loss_function,
            max_trials=10,
            objective='val_accuracy',
            overwrite=True,
            metrics=['accuracy'],
            tuner=args.search,
            seed=seed
        )
    
        # Override the input reshaping dynamically to keep (samples, timesteps, channels)
        nas.reshape_inputs = types.MethodType(behavioral_reshape_inputs, nas)
    
        start_time = time.time()
        nas.fit(
            X=X_train,
            y=y_train,
            epochs=nas_search_epochs,
            final_model_epochs=nas_final_epochs,
            batch_size=args.batch_size,
            verbose=1
        )
        nas_train_time = time.time() - start_time
        print(f"NAS train time: {timedelta(seconds=int(nas_train_time))}")
    
        # Evaluate NAS Model
        print("Evaluating NAS Model Predictions (Accuracy ignored for security evaluation)...")
        y_pred_scores_nas = nas.predict_scores(X_test)
        nas_metrics = evaluate_classification_performance(
            y_test, y_pred_scores_nas, args.dataset, "NAS", output_dir=args.output_dir
        )
    
        # 4. Save best model and extract architecture config and hyperparameters
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        model_to_use = None
        if hasattr(nas, 'best_model') and nas.best_model is not None:
            model_to_use = nas.best_model
        elif os.path.exists(nas.best_model_file_path):
            model_to_use = load_model(nas.best_model_file_path)
    
        if model_to_use is not None:
            # Save results/best_model.keras
            best_model_path = os.path.join(results_dir, "best_model.keras")
            model_to_use.save(best_model_path)
            print(f"Best NAS model saved to {best_model_path}")
            
            # Save results/best_architecture.json
            best_arch_path = os.path.join(results_dir, "best_architecture.json")
            try:
                arch_config = model_to_use.get_config()
                with open(best_arch_path, 'w') as f:
                    json.dump(arch_config, f, indent=4)
                print(f"Best NAS architecture config saved to {best_arch_path}")
            except Exception as e:
                print(f"Could not save best architecture: {e}")
    
        # Save results/best_hyperparameters.json
        best_hp_path = os.path.join(results_dir, "best_hyperparameters.json")
        try:
            best_hps = nas.auto_model.tuner.get_best_hyperparameters()[0]
            with open(best_hp_path, 'w') as f:
                json.dump(best_hps.values, f, indent=4)
            print(f"Best NAS hyperparameters saved to {best_hp_path}")
        except Exception as e:
            print(f"Could not save best hyperparameters: {e}")
    
        # 5. Calculate Guessing Entropy and append results to CSV
        print("\n--- Calculating Guessing Entropy Traces (Nt) ---")
        baseline_ge_mean, _ = compute_empirical_guessing_entropy_stats(
            X_test, y_test, baseline.model, num_classes, max_traces=150, num_simulations=100, seed=seed
        )
        baseline_nt = next((i + 1 for i, ge in enumerate(baseline_ge_mean) if ge < 0.05), ">150")

        nas_nt = "N/A"
        best_arch_str = "N/A"
        if model_to_use is not None:
            nas_ge_mean, _ = compute_empirical_guessing_entropy_stats(
                X_test, y_test, model_to_use, num_classes, max_traces=150, num_simulations=100, seed=seed
            )
            nas_nt = next((i + 1 for i, ge in enumerate(nas_ge_mean) if ge < 0.05), ">150")
            try:
                best_arch_str = " -> ".join([layer.name for layer in model_to_use.layers])
            except Exception:
                pass
                
        # Evaluate Attackability
        is_attackable = False
        if nas_nt != ">150" and nas_nt != "N/A":
            if int(nas_nt) < 3:
                is_attackable = True
                
        csv_file = os.path.join(results_dir, "sca_domains_evaluation.csv")
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['domain', 'dataset', 'seed', 'train_time', 'baseline_ge_nt', 'nas_ge_nt', 'is_attackable', 'best_architecture'])
            writer.writerow([
                args.domain,
                args.dataset,
                seed,
                f"{timedelta(seconds=int(nas_train_time))}",
                baseline_nt,
                nas_nt,
                is_attackable,
                best_arch_str
            ])
            
        print("="*60)
        print(f"Experiment finished successfully. Metrics logged to {csv_file}")
        print(f"Domain: {args.domain} | Dataset: {args.dataset}")
        print(f"Baseline Traces (Nt): {baseline_nt} | NAS Traces (Nt): {nas_nt}")
        print(f"Target is Attackable: {is_attackable}")
        print("="*60)

if __name__ == "__main__":
    main()
