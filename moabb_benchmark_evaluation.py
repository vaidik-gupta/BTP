import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from moabb.datasets import EPFLP300, ErpCore2021_ERN
from moabb.paradigms import P300
from moabb.evaluations import CrossSessionEvaluation
from mne.decoding import CSP
from tensorflow.keras.utils import to_categorical
from eeg_net import EEGNetClassifier
import warnings
warnings.filterwarnings('ignore')


def prepare_moabb_data(X, y):
    """Prepare MOABB data for EEGNet"""
    # Convert to numpy arrays if needed
    X = np.array(X)
    y = np.array(y)
    
    # Ensure correct shape: (n_samples, n_channels, n_timepoints)
    if len(X.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {X.shape}")
    
    X = X[:, :, :, np.newaxis]  # Add channel dimension
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    return X.astype(np.float32), y_categorical, y_encoded, le


def evaluate_on_dataset(dataset_name, paradigm, dataset, X_train, y_train, X_test, y_test, 
                        nb_classes, epochs=50, batch_size=32):
    """Train and evaluate EEGNetClassifier on a single dataset"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating on {dataset_name} Dataset")
    print(f"{'='*70}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {nb_classes}")
    
    # Prepare data
    X_train_prep, y_train_prep, y_train_encoded, _ = prepare_moabb_data(X_train, y_train)
    X_test_prep, y_test_prep, y_test_encoded, le = prepare_moabb_data(X_test, y_test)
    
    # Get actual dimensions
    n_channels = X_train_prep.shape[1]
    n_samples = X_train_prep.shape[2]
    
    # Create and train classifier
    classifier = EEGNetClassifier(
        nb_classes=nb_classes,
        Chans=n_channels,
        Samples=n_samples,
        dropoutRate=0.5,
        F1=8,
        D=2,
        F2=16
    )

    print(f"\nTraining EEGNetClassifier...")
    classifier.eegnet.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = classifier.eegnet.fit(
        X_train_prep, y_train_prep,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Predictions
    print(f"\nGenerating predictions...")
    y_pred_probs = classifier.eegnet.predict(X_test_prep, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC for binary classification
    roc_auc = None
    if nb_classes == 2:
        try:
            roc_auc = roc_auc_score(y_test_encoded, y_pred_probs[:, 1])
        except:
            pass
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Results dictionary
    results = {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'class_labels': le.classes_,
        'y_pred': y_pred,
        'y_true': y_test_encoded,
        'history': history
    }
    
    return results, classifier


def print_results_table(results_list):
    """Print results in a formatted table"""
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*100}")
    
    df_results = []
    for results in results_list:
        row = {
            'Dataset': results['dataset'],
            'Accuracy': f"{results['accuracy']:.4f}",
            'F1 (Macro)': f"{results['f1_macro']:.4f}",
            'F1 (Weighted)': f"{results['f1_weighted']:.4f}",
            'Precision (Macro)': f"{results['precision_macro']:.4f}",
            'Precision (Weighted)': f"{results['precision_weighted']:.4f}",
            'Recall (Macro)': f"{results['recall_macro']:.4f}",
            'Recall (Weighted)': f"{results['recall_weighted']:.4f}",
        }
        if results['roc_auc'] is not None:
            row['ROC-AUC'] = f"{results['roc_auc']:.4f}"
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    print(df.to_string(index=False))
    print(f"{'='*100}\n")
    
    return df


def plot_confusion_matrices(results_list, save_path='confusion_matrices.png'):
    """Plot confusion matrices for all datasets"""
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    
    if len(results_list) == 1:
        axes = [axes]
    
    for idx, results in enumerate(results_list):
        cm = results['confusion_matrix']
        class_labels = results['class_labels']
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=axes[idx],
            cbar_kws={'label': 'Count'}
        )
        
        axes[idx].set_title(f"{results['dataset']} Confusion Matrix\n(Accuracy: {results['accuracy']:.4f})")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {save_path}")
    plt.close()


def plot_training_history(results_list, save_path='training_history.png'):
    """Plot training history for all datasets"""
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 4))
    
    if len(results_list) == 1:
        axes = [axes]
    
    for idx, results in enumerate(results_list):
        history = results['history']
        
        axes[idx].plot(history.history['accuracy'], label='Train Accuracy')
        axes[idx].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[idx].set_title(f"{results['dataset']} Training History")
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Accuracy')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    """Main evaluation function"""
    print("Loading MOABB Benchmark Datasets...")
    
    results_all = []
    
    # ==================== P300 Dataset ====================
    try:
        print("\n" + "="*70)
        print("LOADING P300 DATASET")
        print("="*70)
        
        p300_dataset = EPFLP300()
        p300_paradigm = P300()
        
        # Get a single subject's data for demonstration
        subject_id = p300_dataset.subject_list[0]
        print(f"Using subject: {subject_id}")
        
        X, y, metadata = p300_paradigm.get_data(
            dataset=p300_dataset,
            subjects=[subject_id]
        )
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Full data shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Unique classes: {np.unique(y)}")
        
        # Train-test split
        n_samples = int(X.shape[0])  # Use shape[0] instead of len(X)
        train_size = int(0.7 * n_samples)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train_p300, X_test_p300 = X[train_idx], X[test_idx]
        y_train_p300, y_test_p300 = y[train_idx], y[test_idx]
        
        print(f"Train set size: {len(X_train_p300)}")
        print(f"Test set size: {len(X_test_p300)}")
        
        results_p300, classifier_p300 = evaluate_on_dataset(
            'P300',
            p300_paradigm,
            p300_dataset,
            X_train_p300, y_train_p300,
            X_test_p300, y_test_p300,
            nb_classes=len(np.unique(y)),
            epochs=20,
            batch_size=32
        )
        
        results_all.append(results_p300)
        
        print(f"\nP300 Results:")
        print(f"  Accuracy: {results_p300['accuracy']:.4f}")
        print(f"  F1 (Macro): {results_p300['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {results_p300['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {results_p300['precision_macro']:.4f}")
        print(f"  Recall (Macro): {results_p300['recall_macro']:.4f}")
        
    except Exception as e:
        print(f"Error with P300 dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== ERN Dataset ====================
    try:
        print("\n" + "="*70)
        print("LOADING ERN DATASET")
        print("="*70)
        
        ern_dataset = ErpCore2021_ERN()
        ern_paradigm = P300()
        
        # Get a single subject's data for demonstration
        subject_id = ern_dataset.subject_list[0]
        print(f"Using subject: {subject_id}")
        
        X, y, metadata = ern_paradigm.get_data(
            dataset=ern_dataset,
            subjects=[subject_id]
        )
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Full data shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Unique classes: {np.unique(y)}")
        
        # Train-test split
        n_samples = int(X.shape[0])  # Use shape[0] instead of len(X)
        train_size = int(0.7 * n_samples)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        X_train_ern, X_test_ern = X[train_idx], X[test_idx]
        y_train_ern, y_test_ern = y[train_idx], y[test_idx]
        
        print(f"Train set size: {len(X_train_ern)}")
        print(f"Test set size: {len(X_test_ern)}")
        
        results_ern, classifier_ern = evaluate_on_dataset(
            'ERN',
            ern_paradigm,
            ern_dataset,
            X_train_ern, y_train_ern,
            X_test_ern, y_test_ern,
            nb_classes=len(np.unique(y)),
            epochs=20,
            batch_size=32
        )
        
        results_all.append(results_ern)
        
        print(f"\nERN Results:")
        print(f"  Accuracy: {results_ern['accuracy']:.4f}")
        print(f"  F1 (Macro): {results_ern['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {results_ern['f1_weighted']:.4f}")
        print(f"  Precision (Macro): {results_ern['precision_macro']:.4f}")
        print(f"  Recall (Macro): {results_ern['recall_macro']:.4f}")
        
    except Exception as e:
        print(f"Error with ERN dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ==================== Summary ====================
    if results_all:
        print("\n" + "="*70)
        print("GENERATING SUMMARY REPORTS")
        print("="*70)
        
        # Print results table
        df = print_results_table(results_all)
        df.to_csv('benchmark_results.csv', index=False)
        print("Results saved to benchmark_results.csv")
        
        # Plot confusion matrices
        plot_confusion_matrices(results_all)
        
        # Plot training history
        plot_training_history(results_all)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
    else:
        print("No datasets were successfully evaluated.")


if __name__ == '__main__':
    main()
