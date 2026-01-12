# Training Lucy
**From Quadrupedal to Bipedal Locomotion with PPO**

Please see the related medium article at:
https://medium.com/p/bdad7811ef69/edit

Further projectory and article updates are planned in February and March 2026, as my academic workload is projected to decrease.

# intro

This repository contains the codebase for **Lucy**, a MuJoCo-based reinforcement learning project exploring how **Proximal Policy Optimization (PPO)** can learn stable standing and locomotion behaviors in a biologically inspired articulated agent.

Lucy is a simplified, shrew-like model with 19 controllable degrees of freedom and ~200 sensory observations. Using PPO (via Stable-Baselines3), the project investigates how reward design, morphology, and policy constraints shape the emergence of coordinated motion, with the long-term goal of transitioning from quadrupedal to bipedal locomotion.

---

## Key Features
- **Custom MuJoCo environments** for standing and walking
- **PPO (actor–critic)** training with constrained policy updates
- C**reward shaping** for posture and balance.
- **Noise-robust training** via randomized initial conditions
- Structured **experiment logging and reproducibility**
- Designed for **CPU training** and HPC batch execution

---

## Repository Structure
- `src/` – Environments, wrappers, training scripts, and utilities  
- `animals/` – MuJoCo XML models (Lucy morphology variants)  
- `cluster_scripts/` – HPC-friendly training and logging scripts  
- `trained_models/` – Saved checkpoints (paths expected, files excluded)  
- `logs/` – Training logs and monitoring outputs  

*(Generated images, videos, and large model files are excluded to keep the repo lightweight.)*

---

## Environment
``` Python
pip install uv
uv lock
uv sync
```


