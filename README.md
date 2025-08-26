# 🏎️ AI Racing Driver for TORCS

An **intelligent self-driving agent** that learns to drive in the TORCS racing simulator using supervised machine learning.
The system leverages a **multi-output neural network** trained on real driving data to control steering, acceleration, and braking in real time.

---

## 📌 Overview

This project implements a **machine-learning pipeline** for building an AI-controlled racecar in the TORCS simulator.

The pipeline consists of:

* **Data Collection & Processing** – Human driving data aggregated from multiple tracks.
* **Model Training** – Neural network trained on key sensory features.
* **Real-Time Control** – Python socket-based integration with TORCS for live racing.

---

## ✨ Key Features

* **Neural Network-Based Driving**: Predicts continuous steering, binary acceleration, and braking.
* **Real-Time Integration**: Uses a Python socket client to control TORCS in live simulation.
* **Modular Pipeline**: Separate stages for data processing, training, and inference for flexibility and reusability.

---

## 🧠 Model Training Architecture

### 🔹 Input Features (37 total)

* **Track sensors**: Track\_1 to Track\_19 (distance to track edges)
* **Velocities**: SpeedX, SpeedY, SpeedZ
* **Car state**: Angle, TrackPosition, RPM, Damage
* **Wheel spin & opponent proximity**

These features provide **spatial awareness** and **kinematic context** for the AI.

### 🔹 Output Targets

* **Steering** → Continuous value ∈ \[-1, 1] (scaled with `MinMaxScaler`)
* **Acceleration** → Binary (on/off)
* **Braking** → Binary (on/off)

### 🔹 Network Architecture

Implemented with **Scikit-learn’s `MLPRegressor`**

| Layer    | Configuration                               |
| -------- | ------------------------------------------- |
| Input    | 37 features                                 |
| Hidden 1 | 64 neurons, ReLU                            |
| Hidden 2 | 128 neurons, ReLU                           |
| Hidden 3 | 64 neurons, ReLU                            |
| Output   | 3 neurons (Steering, Acceleration, Braking) |

**Hyperparameters**:

```python
learning_rate='adaptive'
early_stopping=True
max_iter=1000
batch_size=64
tol=1e-5
n_iter_no_change=15
```

---

## 🔄 Data Handling & Training Strategy

1. **CSV Aggregation** → Combine multiple driving sessions.
2. **Preprocessing** →

   * Missing values filled with feature-wise means
   * Normalization: `StandardScaler` (inputs), `MinMaxScaler` (steering)
   * Thresholding applied to acceleration & braking
3. **Splitting** → 60% training, 20% validation, 20% testing.
4. **Training** → Validation-driven early stopping & adaptive learning rate scheduling.

---

## ⚙️ Requirements

* **Python 3.6+**
* Libraries:

  ```bash
  pip install scikit-learn pandas numpy matplotlib joblib
  ```
* **TORCS** with **SCRC patch** (socket-based control)

---

## 👥 Team Contributions

* **Muhammad Bilal** → Collected driving data on *E-Track 3*, tested TensorFlow NN.
* **Rana Bilal** → Collected data on *Dirt Track*, implemented Scikit-learn MLP models, experimented with TD3 RL.
* **Mehboob** → Collected data on *G-Speedway*, tested Random Forest model.

Final model: **Scikit-learn `MLPRegressor`**, selected for efficiency and real-time compatibility.

---

## 🏁 Results & Conclusion

* Built a **neural network driver** for TORCS that runs efficiently in real time.
* Model **learns human-like driving** from data instead of relying on hard-coded rules.
* Key learnings: importance of **data preprocessing**, **feature scaling**, and **model tuning** for performance.

This project demonstrates how **supervised learning** can effectively replicate human driving behavior in a racing simulator.

---

## 🚀 Future Work

* Experiment with **reinforcement learning (TD3, DDPG)** for adaptive racing strategies.
* Expand training data with **more diverse tracks & driving styles**.
* Implement **multi-agent racing** with AI vs AI competitions.

---
