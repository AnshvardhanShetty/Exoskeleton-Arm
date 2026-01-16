# ExoHand  
**An End-to-End EMG-Controlled Exoskeleton Hand System**  
*signal processing · machine learning · embedded control*

---

## Overview
**ExoHand** is an in-progress project to build a **real-time EMG-controlled exoskeleton hand**, integrating signal acquisition, robust preprocessing, intent inference, and hardware actuation.

The project is designed as a **complete control pipeline**, not a standalone model:

raw EMG → preprocessing → windowed representations → intent decoding → actuation

To prioritise robustness and deployability, the current system predicts **high-level hand intent**:

- **Rest**
- **Open / Extend**
- **Close / Flex**

This abstraction mirrors how practical assistive devices are controlled and provides a stable foundation for finer-grained finger-level control.

---

## Data Sources
ExoHand intentionally combines **heterogeneous EMG data sources** to stress-test robustness beyond idealised lab conditions.

### GrabMyo Dataset (Previous Study)
We use EMG data from a previous study recorded with the **GrabMyo system** (32-channel surface EMG, WFDB format).

- Raw `.dat / .hea` trials ingested directly  
- Subset of **4 informative channels** selected for modelling  
- Sliding-window segmentation  
- Raw gesture labels mapped to functional intent classes  
- Supports multi-participant, session-level evaluation  

### Custom Forearm EMG Recordings
We also recorded our own EMG data during **hand flex, rest, and extend** using a **2-channel forearm setup**.

This pipeline is explicitly designed to handle real-world acquisition issues:
- irregular or noisy timestamps  
- spikes and motion artefacts  
- inconsistent units (ADC counts vs volts)  
- unknown or drifting sampling rate  

---

## Signal Processing & Preprocessing

### Robust TXT-Based Ingestion (Custom Data)
Custom EMG logs are parsed using a fault-tolerant loader that:
- ignores header and metadata lines  
- extracts numeric values from arbitrarily formatted rows  
- infers time from the first value and channel signals from trailing values  

### Sampling Rate Estimation
Rather than assuming a fixed sampling frequency:
- sampling rate is estimated directly from timestamp deltas  
- median-based estimation with span-based fallback  
- clamped to physiologically reasonable EMG ranges  

### Channel-Level Cleaning
Each channel undergoes the following preprocessing steps:
- heuristic ADC → volt conversion  
- spike detection via sliding median / MAD z-score  
- spike replacement via linear interpolation  
- band-pass filtering (20–450 Hz, Nyquist-safe)  
- full-wave rectification  
- moving-average envelope extraction  

Diagnostic plots are generated to verify signal quality at each stage.

---

## Windowed Feature Extraction
Signals are segmented into **overlapping windows** (e.g. 100 ms windows with 25 ms stride).

### Per-Channel Features
- RMS, MAV, variance  
- waveform length  
- max amplitude  
- zero crossings  
- slope sign changes  
- Willison amplitude  
- integrated EMG  
- mean and median frequency (FFT-based)  
- envelope mean and max  

### Cross-Channel Features
- envelope activity ratio  
- inter-channel correlation  

This produces compact, information-dense representations suitable for low-latency control.

---

## Intent Abstraction
Raw gestures are mapped to functional intent classes:

0 → Rest
1 → Close / Flex
2 → Open / Extend

This reduces label noise and aligns model outputs with actuator-level behaviour.

---

## Models

### Feature-Based Models
A **Histogram Gradient Boosting Classifier** is trained on extracted features.

- class-balanced training  
- participant-level train / validation / test splits  
- EMG-specific data augmentation (gain variation, bias shifts, channel dropout, noise injection)  

This provides a strong, interpretable baseline suitable for embedded deployment.

### Temporal CNN Models
A lightweight **1D convolutional neural network** is trained directly on downsampled EMG envelopes.

- multi-channel time-series input  
- temporal convolution and pooling  
- adaptive temporal aggregation  
- noise-based augmentation during training  

Evaluation uses participant-held-out splits to test generalisation.

---

## Hardware Integration (In Progress)
Hardware is a core component of ExoHand and is being developed in parallel with the software stack.

Current work includes:
- mechanical exoskeleton hand design (3D-printed)  
- actuator selection and integration  
- microcontroller interface  
- intent-to-action mapping (state machine → proportional control)  

The goal is **closed-loop, real-time EMG-driven actuation**.

---
## Repository Structure
```
preprocessing.py              # GrabMyo WFDB preprocessing + feature extraction
build_intent_dataset.py       # Gesture → intent dataset construction
model_building.py             # Feature-based ML training and evaluation
cnn_preprocess.py             # Envelope extraction for CNN input
cnn_train.py                  # CNN training and evaluation
preprocess_2ch_txt.py         # Robust preprocessing for custom EMG recordings
```
---

## Technologies
- **Python**
- **NumPy, SciPy** (signal processing)
- **scikit-learn** (classical ML)
- **PyTorch** (CNNs)
- **WFDB** (biomedical signal ingestion)
- **Matplotlib** (diagnostics)

---

## Project Status
**Active development.**  
Software pipeline largely complete; hardware integration and real-time deployment ongoing.
