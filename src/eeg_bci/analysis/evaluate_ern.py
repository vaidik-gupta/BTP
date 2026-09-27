import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# NOTE: `ProcessingApp` is an external EEG cleaning package that is NOT vendored in
# this repo. Install/provide it before running this script, or comment out the
# cleaning step in evaluate_ern_dataset() to run on raw signals.
from ProcessingApp.EEGCleaningPipeline import EEGCleaningPipeline
from ProcessingApp.Signals import TimeDomainSignal

from eeg_bci.paths import DATA_DIR

def evaluate_ern_dataset(X_paths, y_paths):
    # 1. Load the data
    print("Loading data...")
    try:
        ## Append all the datasets together if multiple paths are provided
        if isinstance(X_paths, list) and isinstance(y_paths, list):
            X_list = [np.load(X_path) for X_path in X_paths]
            y_list = [np.load(y_path) for y_path in y_paths]
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
        else:
            X = np.load(X_paths)
            y = np.load(y_paths)

        X = X.transpose(0, 2, 1)  # Ensure shape is (trials, channels, time)
    except Exception as e:
        return f"Error loading files: {e}"
    
    ## Use EEG cleaning pipeline to preprocess the data (optional, can be commented out if not needed)
    cleaner = EEGCleaningPipeline(fs=250)  # Assuming 250 Hz sampling rate; adjust as needed

    signals = [TimeDomainSignal(trial, fs=250) for trial in X]
    cleaned_signals = cleaner.clean_dataset(signals)


    X = np.array([signal.data for signal in cleaned_signals])

    ## replace -1 with zeros in y
    y = np.where(y == -1, 0, y)

    print(f"Original X shape: {X.shape}")
    print(f"Label y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # 2. Reshape for standard ML 
    # Transforms (trials, channels, time) -> (trials, channels * time)
    n_trials = X.shape[0]
    X_flat = X.reshape(n_trials, -1)
    print(f"Flattened X shape: {X_flat.shape}")

    # 3. Define the classifiers
    # class_weight='balanced' is critical for ERN since errors are usually rare
    classifiers = {
        "Logistic Regression (L2)": LogisticRegression(max_iter=3000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=24),
        "SVM (RBF Kernel)": SVC(kernel='rbf', class_weight='balanced', random_state=1)
    }

    # 4. Set up Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'roc_auc', 'f1_macro']

    # 5. Train and Evaluate
    print("\n--- Model Evaluation (5-Fold CV) ---")
    for name, clf in classifiers.items():
        # Using a pipeline ensures data is scaled within the CV loop, preventing data leakage
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.90)),
            ('classifier', clf)
        ])
        
        scores = cross_validate(pipeline, X_flat, y, cv=cv, scoring=scoring_metrics)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std():.4f})")
        print(f"  ROC-AUC:   {scores['test_roc_auc'].mean():.4f} (+/- {scores['test_roc_auc'].std():.4f})")
        print(f"  F1 (Macro):{scores['test_f1_macro'].mean():.4f} (+/- {scores['test_f1_macro'].std():.4f})")

if __name__ == "__main__":
    # Example usage: point at collected ERN datasets under data/raw/
    evaluate_ern_dataset([str(DATA_DIR / 'raw' / 'ERN_X1.npy')],
                         [str(DATA_DIR / 'raw' / 'ERN_y1.npy')])