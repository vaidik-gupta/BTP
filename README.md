# EEG Brain-Computer Interface — PPO Game Control

B.Tech Project (BTP). An **8-channel EEG** headset streams brain waves over
[LSL](https://labstreaminglayer.org/); a **PPO reinforcement-learning agent**
reads short windows of those signals and uses **motor imagery** (imagining a
left- vs right-hand curl) to steer a player in a lane-based "cone avoidance"
game rendered on the laptop screen.

The same convolutional network used by the RL agent ([EEGNet](https://arxiv.org/abs/1611.08024))
is also used offline to sanity-check that the recorded brain waves are
classifiable at all.

> 📄 Full write-up: [`docs/BTP_Report.pdf`](docs/BTP_Report.pdf)

---

## How it fits together

```
   EEG headset ──LSL stream──▶ data_collection/   ──▶  data/  (.npy / .csv)
                                                          │
                                                          ▼
                                      analysis/  (offline: is the signal learnable?)
                                                          │
   ┌──────────────────────────────────────────────────────┘
   ▼
 game/  ──▶  cone_game (pygame env)  ◀──drives──  train_ppo  ◀── models/ (EEGNet + PPO)
                                                     │
                                                     ▼
                                          models_saved/*.keras  ──▶  run_policy
```

- **models/** — the neural nets. `eeg_net.py` is the EEGNet CNN (acts as both the
  policy and value network, and as a plain classifier). `ppo.py` is the PPO
  agent (clipped surrogate objective, TensorFlow).
- **game/** — `cone_game.py` is the pygame environment (`reset` / `step` /
  `render`, 3 lanes, actions = left / stay / right). `train_ppo.py` runs the live
  PPO training loop off the EEG stream. `run_policy.py` replays a saved policy.
- **data_collection/** — scripts that record from the LSL stream into `data/`.
- **analysis/** — offline evaluation: train EEGNet on the curl data, classic-ML
  baselines on ERN data, a MOABB public-dataset benchmark, and signal plotting.

---

## Layout

```
src/eeg_bci/
├── paths.py                 # central data / artifact locations
├── models/
│   ├── eeg_net.py           # EEGNet CNN + EEGNetClassifier
│   └── ppo.py               # PPOAgentTF
├── game/
│   ├── cone_game.py         # pygame environment
│   ├── train_ppo.py         # live PPO training from EEG
│   └── run_policy.py        # run a saved policy (CLI)
├── data_collection/
│   ├── collect_curl.py          # cued left/right curl windows -> .npy
│   ├── collect_biosignal_csv.py # timed REST/CURL protocol -> .csv
│   ├── collect_xon_csv.py       # raw Xon-headset dump -> .csv
│   └── ern_experiment.py        # direction + error-feedback (ERN) -> .npy
└── analysis/
    ├── train_eegnet_curl.py     # train EEGNet on curl datasets
    ├── evaluate_ern.py          # classic ML baselines on ERN data
    ├── moabb_benchmark.py       # EEGNet on public P300 / ERN datasets
    └── visualize_wavepackets.py # per-channel wavepacket plots

data/         # (git-ignored) curl/ = recorded curl datasets, raw/ = new recordings
artifacts/    # (git-ignored) generated figures, benchmark tables
models_saved/ # (git-ignored) trained PPO policy / value networks
docs/         # BTP_Report.pdf
tests/        # ppo_smoke_test.py — PPO sanity check on a toy env
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core deps + importable `eeg_bci` package
pip install -e ".[benchmark]"   # also install moabb + mne for moabb_benchmark
```

All scripts import the package, so run them as modules from the repo root, e.g.
`python -m eeg_bci.game.run_policy ...`.

## Data

Datasets and trained models are **not** tracked in git. `paths.py` creates the
`data/`, `artifacts/`, and `models_saved/` folders on first import.

The curl motor-imagery datasets (place under `data/curl/`) have shape
`(trials, 10 channels, samples)` with int labels:

| file                | shape            | window   |
|---------------------|------------------|----------|
| `X_past_*.npy`      | `(~240, 10, 125)`| 0.5 s    |
| `X_future_*.npy`    | `(~240, 10, 250)`| 1.0 s    |
| `Y_*.npy`           | `(~240,)`        | 0/1 label|

## Running things

Live pieces need the EEG device streaming over LSL first.

```bash
# 1. Record cued left/right curl windows
python -m eeg_bci.data_collection.collect_curl

# 2. Check the curl signal is classifiable (offline, no device)
python -m eeg_bci.analysis.train_eegnet_curl      # writes artifacts/confusion_matrix_past.png

# 3. Train the PPO agent live on the game
python -m eeg_bci.game.train_ppo                  # saves models to models_saved/

# 4. Replay a trained policy
python -m eeg_bci.game.run_policy models_saved/VAIDIK_trial2_episode10_policy.keras

# PPO sanity check (no EEG, no game)
python -m tests.ppo_smoke_test
```

## Known caveats

- **`analysis/evaluate_ern.py`** imports an external `ProcessingApp` EEG-cleaning
  package that is **not** included in this repo. Provide it, or comment out the
  cleaning step, before running.
- **`data_collection/ern_experiment.py`** expects direction cue images at
  `ERN_images/{Up,Down,Left,Right}.png` (not included); it degrades gracefully in
  headless environments.
- `game/train_ppo.py` runs a keyboard-free dummy loop (`main2`) by default; switch
  the `__main__` block to `main()` for real EEG-driven training.
