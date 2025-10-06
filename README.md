# Exoskeleton-Arm
# EMG-Controlled Exoskeleton Arm

## Overview
This project aims to develop an **electromyography (EMG)-controlled exoskeleton arm** for upper-limb rehabilitation.  
The system detects muscle activation signals and translates them into proportional motor control using signal processing and machine-learning methods.  
Our long-term goal is to create an assistive device that adapts to user-specific movement intent in real time.

---

## Data Collection
All EMG data used in this project was **self-recorded** using surface electrodes placed over the biceps and triceps.

We recorded across a range of conditions, including:
- Different contraction intensities  
- Static holds and dynamic movements  
- Supination and pronation variations  
- Isometric, isotonic, and fatigue conditions  

This custom dataset allowed us to test model robustness against realistic noise and variability.

---

## Preprocessing Pipeline
We built a **custom data preprocessing and filtering pipeline** in Python to clean and prepare the raw EMG signals for analysis.

**Steps include:**
1. **Band-pass filtering** to isolate the EMG frequency band (typically 20–450 Hz)  
2. **Rectification and smoothing** to obtain the signal envelope  
3. **Segmentation and feature extraction** (RMS, mean absolute value, zero-crossings, slope sign changes)  
4. **Normalization** across trials for model compatibility  

This ensures that all training data is consistent and ready for classification.

---

## Machine Learning Models
We’ve tested two models so far:
- **Random Forest Classifier**  
- **Logistic Regression Classifier**

Each was trained to classify muscle activity patterns into distinct movement intentions (e.g., flexion vs. relaxation).  
Both models achieved around **70 % accuracy**, with ongoing work focused on feature engineering, noise reduction, and testing additional algorithms such as **SVMs**, **CNNs**, and **LSTMs** for temporal pattern recognition.

---

## Hardware Development
We are currently building the **first exoskeleton prototype**, integrating:
- A lightweight 3D printed mechanical arm assembly  
- Servo or linear actuator–based motion control  
- A microcontroller interface for EMG-to-motion translation  

Future iterations will improve ergonomics, signal stability, and responsiveness.

---

## Next Steps
- Refine preprocessing for real-time performance  
- Collect more EMG data across multiple users  
- Integrate trained models with embedded hardware for live testing  
- Explore adaptive and reinforcement-learning control strategies  

---

## Technologies Used
- **Languages:** Python  
- **Libraries:** NumPy, SciPy, scikit-learn, matplotlib, pandas  
- **Hardware:** Surface EMG sensors, Arduino/microcontroller, servo motors, 3D Printed partts  
- **Tools:** VS Code, GitHub, Jupyter Notebook  

---

## Team
Developed by **Ansh Shetty** as the lead on the software development and **Adhi Sasikumar** as the lead on the hardware development as an independent biomedical engineering project exploring the intersection of **signal processing, machine learning, and human–computer interfaces**.
