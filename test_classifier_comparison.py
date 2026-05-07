"""
Comprehensive classifier comparison script using SpectralPowerClassifier.
Tests various pipeline configurations and compares their performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from ProcessingApp.EEGCleaningPipeline import EEGCleaningPipeline
from ProcessingApp.classifiers import SpectralPowerClassifier
from ProcessingApp.Signals import TimeDomainSignal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def load_eeg_data(data_paths: list[str], labels_paths: list[str]):
    """Load EEG data and labels from .npy files."""
    X_list = [np.load(path) for path in data_paths]
    y_list = [np.load(path) for path in labels_paths]
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)


    X = np.transpose(X, (0, 2, 1)) 

    print(f"Loaded data shape: {X.shape}, labels shape: {y.shape}")
    return X, y


def preprocess_signals(X: np.ndarray, fs: int = 250) -> list[TimeDomainSignal]:
    """Convert numpy arrays to TimeDomainSignal objects."""
    signals = []
    for i in range(X.shape[0]):
        # Ensure 2D shape (channels x samples)
        if X[i].ndim == 1:
            signal_data = X[i].reshape(1, -1)
        else:
            signal_data = X[i]
        signals.append(TimeDomainSignal(signal_data, fs=fs))
    return signals


def clean_signals(signals: list[TimeDomainSignal], 
                  lowcut: float = 0.1, 
                  highcut: float = 40.0,
                  notch_freq: float = 50.0) -> list[TimeDomainSignal]:
    """Clean signals using EEGCleaningPipeline."""
    print("Cleaning signals with EEGCleaningPipeline...")
    pipeline = EEGCleaningPipeline(
        fs=250,
        lowcut=lowcut,
        highcut=highcut,
        notch_freq=notch_freq,
        quality_factor=30.0
    )
    cleaned_signals = pipeline.clean_dataset(signals)
    print(f"Cleaned {len(cleaned_signals)} signals")
    return cleaned_signals


def create_pipelines():
    """Create a collection of different pipeline configurations."""
    pipelines = {
        'PCA_SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
        ]),
        'PCA_SVM_LINEAR': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('classifier', SVC(kernel='linear', class_weight='balanced', random_state=42))
        ]),
        'PCA_LOGISTIC': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'PCA_RANDOM_FOREST': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'NO_PCA_SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
        ]),
        'NO_PCA_LOGISTIC': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'NO_PCA_RANDOM_FOREST': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'PCA_50_SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)),
            ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
        ]),
        'PCA_POLY_SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('classifier', SVC(kernel='poly', degree=3, class_weight='balanced', random_state=42))
        ]),
    }
    return pipelines


def test_pipelines(X_cleaned: list[TimeDomainSignal], 
                   y: np.ndarray,
                   freq_bands: list[tuple[float, float]],
                   pipelines: dict) -> dict:
    """Test all pipelines and collect results."""
    results = {}
    
    for pipeline_name, pipeline in pipelines.items():
        print(f"\nTesting pipeline: {pipeline_name}")
        try:
            classifier = SpectralPowerClassifier(
                fs=250,
                freq_bands=freq_bands,
                pipeline=pipeline,
                K_fold=5,
                scoring_metrics=['accuracy', 'roc_auc', 'f1_macro', 'precision_macro', 'recall_macro']
            )
            
            scores = classifier.train(X_cleaned, y)
            results[pipeline_name] = scores
            
            # Print individual scores
            print(f"  Accuracy: {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std():.4f})")
            print(f"  ROC-AUC: {scores['test_roc_auc'].mean():.4f} (+/- {scores['test_roc_auc'].std():.4f})")
            print(f"  F1-Macro: {scores['test_f1_macro'].mean():.4f} (+/- {scores['test_f1_macro'].std():.4f})")
            
        except Exception as e:
            print(f"  Error testing {pipeline_name}: {str(e)}")
            results[pipeline_name] = None
    
    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create a summary table of all results."""
    summary_data = []
    
    for pipeline_name, scores in results.items():
        if scores is not None:
            row = {
                'Pipeline': pipeline_name,
                'Accuracy (mean)': f"{scores['test_accuracy'].mean():.4f}",
                'Accuracy (std)': f"{scores['test_accuracy'].std():.4f}",
                'ROC-AUC (mean)': f"{scores['test_roc_auc'].mean():.4f}",
                'ROC-AUC (std)': f"{scores['test_roc_auc'].std():.4f}",
                'F1-Macro (mean)': f"{scores['test_f1_macro'].mean():.4f}",
                'F1-Macro (std)': f"{scores['test_f1_macro'].std():.4f}",
                'Precision (mean)': f"{scores['test_precision_macro'].mean():.4f}",
                'Recall (mean)': f"{scores['test_recall_macro'].mean():.4f}",
            }
        else:
            row = {'Pipeline': pipeline_name, 'Status': 'Failed'}
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df


def plot_comparison(results: dict, metric: str = 'test_accuracy'):
    """Plot comparison of metric across different pipelines."""
    pipeline_names = []
    means = []
    stds = []
    
    for pipeline_name, scores in results.items():
        if scores is not None:
            pipeline_names.append(pipeline_name)
            means.append(scores[metric].mean())
            stds.append(scores[metric].std())
    
    # Create figure with better size
    plt.figure(figsize=(14, 6))
    x_pos = np.arange(len(pipeline_names))
    plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Pipeline Configuration', fontsize=12)
    plt.ylabel(f'{metric.replace("test_", "").upper()} Score', fontsize=12)
    plt.title(f'Classifier Comparison: {metric.replace("test_", "").upper()}', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, pipeline_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('classifier_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to classifier_comparison.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 80)
    print("EEG CLASSIFIER COMPARISON TEST")
    print("=" * 80)
    
    # Configuration
    # Using ERN dataset (Error-Related Negativity) as example
    data_file = ["ERN_X1.npy", "ERN_X2.npy"]  
    labels_file = ["ERN_y1.npy", "ERN_y2.npy"] 
    fs = 250  # Sampling frequency in Hz
    
    # Frequency bands for spectral power features
    freq_bands = [
        (0.5, 4),      # Delta
        (4, 8),        # Theta
        (8, 12),       # Alpha
        (12, 30),      # Beta
        (30, 45),      # Gamma
    ]
    

    
    # Load data
    print(f"\nLoading data from {data_file} and {labels_file}...")
    X, y = load_eeg_data(data_file, labels_file)
    
    # Convert to Signal objects
    print("\nConverting to TimeDomainSignal objects...")
    signals = preprocess_signals(X, fs=fs)
    
    # Clean signals
    cleaned_signals = clean_signals(signals, lowcut=0.1, highcut=40.0, notch_freq=50.0)
    
    # Create pipelines
    print("\nCreating pipeline configurations...")
    pipelines = create_pipelines()
    print(f"Created {len(pipelines)} pipeline configurations")
    
    # Test all pipelines
    print("\n" + "=" * 80)
    print("TESTING PIPELINES")
    print("=" * 80)


    results = test_pipelines(cleaned_signals, y, freq_bands, pipelines)
    
    # Generate comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    summary_df = create_comparison_table(results)
    print(summary_df.to_string(index=False))
    
    # Save results to CSV
    summary_df.to_csv('classifier_comparison_results.csv', index=False)
    print("\nResults saved to classifier_comparison_results.csv")
    
    # Plot comparison
    print("\nGenerating visualizations...")
    plot_comparison(results, metric='test_accuracy')
    plot_comparison(results, metric='test_roc_auc')
    plot_comparison(results, metric='test_f1_macro')
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
