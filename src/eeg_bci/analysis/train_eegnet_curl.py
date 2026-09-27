from eeg_bci.models.eeg_net import EEGNet
from eeg_bci.paths import CURL_DATA_DIR, ARTIFACTS_DIR
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns





def train_eegnet(X, y, epochs = 10, batch_size = 32):
    Chans = X.shape[1]
    Samples = X.shape[2]
    nb_classes = len(np.unique(y))
    model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return model


def main1(past = "past"):

    ## read X_past_left and y_past_left from .npy files
    X_left = np.load(CURL_DATA_DIR / f'X_{past}_left.npy')
    y_left = np.load(CURL_DATA_DIR / f'Y_{past}_left.npy')
    X_right = np.load(CURL_DATA_DIR / f'X_{past}_right.npy')
    y_right = np.load(CURL_DATA_DIR / f'Y_{past}_right.npy')


    # Combine left and right data
    X = np.concatenate((X_left, X_right), axis=0)
    y = np.concatenate((y_left, y_right), axis=0)

    # Train test split


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_eegnet(X_train, y_train, epochs=100, batch_size=32)

    # Evaluate the model F1 score, accuracy, precision, recall
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))  

    # Make confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    ## save the matrix as a png file
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(ARTIFACTS_DIR / f'confusion_matrix_{past}.png')





if __name__ == "__main__":
    main1(past = "past")
    






