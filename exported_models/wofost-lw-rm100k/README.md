# Exported WOFOST Crop Reward Model Ensemble (wofost-lw-rm100k)

This repository contains the exported reward model ensemble for the WOFOST wheat crop growth simulation environment (`wofost-lw-v0`). It is trained using the **PEBBLE** (Preference-based Reinforcement Learning) framework on pairwise segment comparisons.

Collaborators can use this model to evaluate crop management policies or guide reinforcement learning agents (e.g., SAC or PPO) on the same crop growth task.

---

## Directory Contents

The export package includes the following files:
*   `reward_model_100000_0.pt`: Trained PyTorch weights for ensemble member 0.
*   `reward_model_100000_1.pt`: Trained PyTorch weights for ensemble member 1.
*   `reward_model_100000_2.pt`: Trained PyTorch weights for ensemble member 2.
*   `lw_diagRM_100k_20260420_175419.log`: Diagnostic training log file documenting training progress, batch statistics, and evaluation metrics.

---

## Specifications & Model Architecture

### 1. Model Configuration
*   **Ensemble Size**: 3 independent neural networks.
*   **Model Type**: Multi-Layer Perceptron (MLP) mapping state-action inputs to scalar rewards.
*   **Layer Architecture**:
    *   **Input Layer**: 16 units (15 observation features + 1 continuous action).
    *   **Hidden Layers**: 3 linear layers of width 256.
    *   **Activation Functions**: `LeakyReLU` on all hidden layer activations, and `tanh` output activation.
    *   **Output Layer**: 1 linear unit mapping to a scalar reward prediction.

### 2. Training Metrics
*   **Training Step**: Step 100,000 of the diagnostic run.
*   **Feedback Budget**: Trained on 1,568 active preference queries labeled by a synthetic teacher.
*   **Pairwise Accuracy**: Reached **~79.6% accuracy** on the pairwise segment validation dataset.

---

## State and Action Spaces

### 1. Observation Space (15 Dimensions)
The underlying environment exposes a 15-dimensional state representation of the crop and soil. All observations are scaled and normalized to the range `[-1.0, 1.0]` by the environment wrapper before inputting to the model:
*   **Development Stage (DVS)**: Normalized at index 1 of the state vector.
    *   `DVS <= -1.0`: Emerging phase.
    *   `-1.0 < DVS < 0.0`: Vegetative phase.
    *   `0.0 < DVS < 1.0`: Reproductive phase.
    *   `DVS >= 1.0`: Mature phase.
*   Other observation indices represent dynamic variables of the soil calendar, crop canopy, transpirational demand, and weather.

### 2. Action Space (1 Dimension)
*   Exposed as a continuous range in `[-1.0, 1.0]`.
*   Represents continuous agromanagement decisions (e.g., fertilization and irrigation scheduling), which are mapped internally to discrete schedules.

---

## Normalization & Inference Rules

To replicate the training conditions, input observations and actions must be processed as follows:

1.  **Observation Preprocessing**: Observations must be clipped and normalized to `[-1.0, 1.0]` based on the environment's boundary limits before forming the state-action vector.
2.  **State-Action Concatenation**: Concatenate the 15-dimensional normalized state vector and 1-dimensional action vector into a 16-dimensional input vector.
3.  **Active Standardization (Optional)**: During online RL training, raw predictions from the ensemble can be standardized using running mean and standard deviation statistics of past predicted rewards, and subsequently clipped to `[-5.0, 5.0]` to guarantee numeric stability in policy gradients.

---

## Quickstart Usage Guide

Below is a complete, copy-pasteable example of how to load the ensemble in PyTorch and perform forward inference:

```python
import os
import numpy as np
import torch
import torch.nn as nn

def make_mlp_member(in_size=16, out_size=1, hidden_dim=256, n_layers=3, activation='tanh'):
    """Reconstructs the network architecture matching training configs."""
    net = []
    for _ in range(n_layers):
        net.append(nn.Linear(in_size, hidden_dim))
        net.append(nn.LeakyReLU())
        in_size = hidden_dim
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    return nn.Sequential(*net)

class WofostRewardEnsemble:
    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.device = device
        self.ensemble = []
        
        # Load all 3 ensemble members
        for idx in range(3):
            model = make_mlp_member()
            filepath = os.path.join(model_dir, f"reward_model_100000_{idx}.pt")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Weight file not found: {filepath}")
            
            state_dict = torch.load(filepath, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.ensemble.append(model)
            print(f"Loaded ensemble member {idx} from {filepath}")

    def predict_ensemble(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        """
        Predict step-wise reward from the ensemble.
        Args:
            obs: np.ndarray of shape (15,) or (B, 15), normalized to [-1.0, 1.0]
            act: np.ndarray of shape (1,) or (B, 1) in range [-1.0, 1.0]
        Returns:
            mean_reward: np.ndarray of shape (B, 1) representing mean ensemble prediction.
        """
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        
        if obs.ndim == 1:
            obs = obs[None, :]
        if act.ndim == 1:
            act = act[None, :]
            
        # Concatenate state and action inputs to form the 16-D feature vector
        x = np.concatenate([obs, act], axis=-1)
        x_t = torch.from_numpy(x).to(self.device)
        
        with torch.no_grad():
            predictions = [model(x_t).cpu().numpy() for model in self.ensemble]
            
        # Average predictions across the ensemble
        mean_reward = np.mean(predictions, axis=0)
        return mean_reward

# Example instantiation and inference
if __name__ == "__main__":
    # Path containing the .pt weight files
    model_directory = "/home/sohams/BPref/exported_models/wofost-lw-rm100k"
    
    ensemble = WofostRewardEnsemble(model_directory, device='cpu')
    
    # Generate dummy input arrays (15-D obs, 1-D action) in range [-1.0, 1.0]
    dummy_obs = np.random.uniform(-1.0, 1.0, size=(15,))
    dummy_act = np.random.uniform(-1.0, 1.0, size=(1,))
    
    reward = ensemble.predict_ensemble(dummy_obs, dummy_act)
    print(f"Predicted step-wise reward: {reward[0][0]:.4f}")
```
