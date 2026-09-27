import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from eeg_bci.paths import DATA_DIR


def _find_dataset_paths():
    x_candidates = sorted(glob.glob(str(DATA_DIR / '**' / 'eeg_dataset_X*.npy'), recursive=True))
    y_candidates = sorted(glob.glob(str(DATA_DIR / '**' / 'eeg_dataset_y*.npy'), recursive=True))

    if not x_candidates or not y_candidates:
        raise FileNotFoundError(f'No eeg_dataset_X*.npy or eeg_dataset_y*.npy files found under {DATA_DIR}.')

    x_path = x_candidates[0]
    y_path = y_candidates[0]

    return x_path, y_path


def visualize_wavepackets(label='wink', n_visualizations=5, x_path=None, y_path=None):
    """Visualize up to n_visualizations wavepackets for the given label."""
    if x_path is None or y_path is None:
        x_path, y_path = _find_dataset_paths()

    print('Loading', x_path, 'and', y_path)
    X = np.load(x_path)
    y = np.load(y_path)
    print(f'Loaded X shape {X.shape}, y shape {y.shape}')

    if y.dtype.kind == 'S':
        y = np.array([v.decode('utf-8') for v in y])

    unique_labels = np.unique(y)
    label_idx = np.where(y == label)[0]

    if label_idx.size == 0:
        raise ValueError(f"No '{label}' labels found in dataset. Available labels: {', '.join(unique_labels)}")

    n_show = min(n_visualizations, label_idx.size)
    selected_idx = label_idx[:n_show]

    print(f'Found {label_idx.size} {label} wavepacket(s); showing {n_show}.')

    for i, idx in enumerate(selected_idx, start=1):
        wavepacket = X[idx]
        if wavepacket.ndim != 2:
            raise ValueError(f'Unexpected wavepacket shape {wavepacket.shape}; expected 2D array.')

        n_samples, n_channels = wavepacket.shape
        fig, axs = plt.subplots(n_channels, 1, figsize=(12, 2.2 * n_channels), sharex=True)
        fig.suptitle(f'{str(label).capitalize()} Wavepacket #{i} (trial index {idx})', fontsize=14)

        for ch in range(n_channels):
            ax = axs[ch] if n_channels > 1 else axs
            ax.plot(np.arange(n_samples), wavepacket[:, ch], lw=0.8)
            ax.set_ylabel(f'Ch {ch+1}')
            ax.grid(True, alpha=0.3)

        axs[-1].set_xlabel('Sample index')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    print('Visualization complete.')


if __name__ == '__main__':
    visualize_wavepackets(
        label="Left",
        n_visualizations=5,
        x_path=str(DATA_DIR / 'raw' / 'direction_waves_X1.npy'),
        y_path=str(DATA_DIR / 'raw' / 'direction_waves_y1.npy'),
    )
