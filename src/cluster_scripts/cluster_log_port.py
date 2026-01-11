
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from src import helpers
from src.definitions import enforce_absolute_path
import src.lucy_classes_v1 as lucy


output_prefix = "cluster_walking_v0"


# -----------------------------
# Paths / run directory
# -----------------------------
def run_dir_for(run_name: str, dt_str: str) -> str:
    return os.path.join("cluster_copy", f"{run_name}_{dt_str}")


def within_run_dir(run_name: str, dt_str: str, name: str) -> str:
    return os.path.join(run_dir_for(run_name, dt_str), name)


# -----------------------------
# Defaults tuned for HPC CPU
# -----------------------------
DEFAULT_MODEL_PARAMS = {
    "policy": "MlpPolicy",
    "verbose": 0,  # reduce stdout overhead; use callback for sparse prints
    "device": "cpu",
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "learning_rate": 2e-4,
    "target_kl": 0.03,
    "policy_kwargs": {"net_arch": {"pi": [512, 512], "vf": [512, 512]}},
}

DEFAULT_ENV_PARAMS = {
    "env_kwargs": {
        "xml_file": enforce_absolute_path("animals/lucy_v3.xml"),
        "render_mode": "None",
        "max_episode_seconds": 30,
    },
    "wrapper_kwargs": {
        "stillness_weight": -2.0,
        "forward_weight": 3.0,
    },
}

DEFAULT_RUN_PARAMS = {
    "env_number": 40,          # match your 40-core goal
    "timesteps": 3_000_000,
    # Logging cadence (low overhead)
    "print_every_s": 20,       # sparse console output
    "flush_every_episodes": 50 # write episode CSV in chunks
}


def create_env(env_params: dict):
    env_kwargs = env_params.get("env_kwargs", {})
    wrapper_kwargs = env_params.get("wrapper_kwargs", {})
    return lucy.LucyWalkingWrapper(lucy.LucyEnv(**env_kwargs), **wrapper_kwargs)


# -----------------------------
# Lightweight episode logger callback
# -----------------------------
@dataclass
class EpisodeRow:
    t_steps: int
    wall_s: float
    ep_rew: float
    ep_len: int


class EpisodeCSVLogger(BaseCallback):
    """
    Captures episode returns from VecMonitor info['episode'] and writes:
      - episode_log.csv (append)
      - final_summary.json
      - training_curve.png
    Also prints sparse progress lines (fps + mean reward).
    """

    def __init__(
        self,
        run_dir: str,
        print_every_s: int = 20,
        flush_every_episodes: int = 50,
        window: int = 100,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.print_every_s = int(print_every_s)
        self.flush_every_episodes = int(flush_every_episodes)
        self.window = int(window)

        self.csv_path = os.path.join(run_dir, "episode_log.csv")
        self.summary_path = os.path.join(run_dir, "final_summary.json")
        self.plot_path = os.path.join(run_dir, "training_curve.png")

        self._t0 = None
        self._last_print_t = 0.0
        self._rows_buf: List[EpisodeRow] = []
        self._ep_rewards: List[float] = []
        self._ep_lengths: List[int] = []

    def _init_callback(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        self._t0 = time.time()
        self._last_print_t = self._t0

        # CSV header (overwrite if exists; each run has its own dir)
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("timesteps,wall_s,ep_rew,ep_len\n")

    def _extract_episodes(self) -> int:
        """
        VecMonitor injects `episode` info dicts when an episode ends:
          info['episode'] = {'r': ep_reward, 'l': ep_len, 't': elapsed}
        In VecEnv, `infos` is a list per env.
        """
        infos = self.locals.get("infos")
        if not infos:
            return 0

        new_eps = 0
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            r = float(ep.get("r", np.nan))
            l = int(ep.get("l", -1))

            wall_s = time.time() - self._t0
            self._rows_buf.append(EpisodeRow(self.num_timesteps, wall_s, r, l))

            self._ep_rewards.append(r)
            self._ep_lengths.append(l)
            new_eps += 1
        return new_eps

    def _flush_csv(self) -> None:
        if not self._rows_buf:
            return
        with open(self.csv_path, "a", encoding="utf-8") as f:
            for row in self._rows_buf:
                f.write(f"{row.t_steps},{row.wall_s:.3f},{row.ep_rew:.6f},{row.ep_len}\n")
        self._rows_buf.clear()

    def _maybe_print(self) -> None:
        now = time.time()
        if (now - self._last_print_t) < self.print_every_s:
            return
        self._last_print_t = now

        # compute recent stats
        n = len(self._ep_rewards)
        if n > 0:
            w = min(self.window, n)
            mean_r = float(np.mean(self._ep_rewards[-w:]))
            mean_l = float(np.mean(self._ep_lengths[-w:]))
        else:
            mean_r, mean_l = float("nan"), float("nan")

        wall_s = now - self._t0
        fps = self.num_timesteps / max(wall_s, 1e-6)
        # single short line (minimal stdout overhead)
        print(f"[t={self.num_timesteps:,}] fps={fps:,.0f}  ep_rew_mean({min(self.window, n)})={mean_r:.2f}  ep_len_mean={mean_l:.1f}")

    def _on_step(self) -> bool:
        new_eps = self._extract_episodes()

        # Flush occasionally to keep memory bounded, but avoid constant I/O
        if new_eps > 0 and (len(self._ep_rewards) % self.flush_every_episodes == 0):
            self._flush_csv()

        self._maybe_print()
        return True

    def _on_training_end(self) -> None:
        # final flush
        self._flush_csv()

        # final summary
        total_eps = len(self._ep_rewards)
        wall_s = time.time() - self._t0
        fps = self.num_timesteps / max(wall_s, 1e-6)

        if total_eps > 0:
            mean_r_100 = float(np.mean(self._ep_rewards[-min(100, total_eps):]))
            mean_l_100 = float(np.mean(self._ep_lengths[-min(100, total_eps):]))
            best_r = float(np.max(self._ep_rewards))
        else:
            mean_r_100 = mean_l_100 = best_r = float("nan")

        summary = {
            "timesteps": int(self.num_timesteps),
            "wall_seconds": float(wall_s),
            "fps": float(fps),
            "episodes": int(total_eps),
            "ep_rew_mean_last_100": mean_r_100,
            "ep_len_mean_last_100": mean_l_100,
            "best_episode_reward": best_r,
            "episode_csv": os.path.basename(self.csv_path),
            "training_curve_png": os.path.basename(self.plot_path),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # plot (end-only; negligible overhead)
        self._write_plot()

        # final one-liner
        print(f"[DONE] steps={self.num_timesteps:,} eps={total_eps} fps={fps:,.0f} last100_rew={mean_r_100:.2f}")

    def _write_plot(self) -> None:
        if len(self._ep_rewards) == 0:
            return

        y = np.array(self._ep_rewards, dtype=float)
        x = np.arange(1, len(y) + 1)

        # rolling mean for a “nice” curve
        w = min(self.window, len(y))
        if w >= 2:
            kernel = np.ones(w) / w
            y_smooth = np.convolve(y, kernel, mode="valid")
            x_smooth = x[w - 1 :]
        else:
            y_smooth = y
            x_smooth = x

        plt.figure()
        plt.plot(x, y, alpha=0.25)
        plt.plot(x_smooth, y_smooth)
        plt.xlabel("Episode")
        plt.ylabel("Episode reward")
        plt.title("Training reward (raw + rolling mean)")
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()


# -----------------------------
# Run logging (single JSON)
# -----------------------------
class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_info_path = os.path.join(self.run_dir, "run_info.json")
        self.info: Dict[str, Any] = {}

    def log_params(self, model_params, env_params, run_params):
        self.info.update(
            {
                "model_params": model_params,
                "env_params": env_params,
                "run_params": run_params,
            }
        )
        self._save()

    def log_artifacts(self, artifacts: Dict[str, str]):
        self.info.update({"artifacts": artifacts})
        self._save()

    def _save(self):
        with open(self.run_info_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main(
    model_params: dict = DEFAULT_MODEL_PARAMS,
    env_params: dict = DEFAULT_ENV_PARAMS,
    run_params: dict = DEFAULT_RUN_PARAMS,
    output_prefix: str = "cluster_walking_v0",
):
    # Threading hygiene (prevents thread-explosion across many processes)
    # Best is to set env vars in PBS too; this helps even if you forgot.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import datetime
    dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = output_prefix
    run_dir = run_dir_for(run_name, dt_str)
    os.makedirs(run_dir, exist_ok=True)

    # Model path: keep within run_dir to avoid clashes
    # (ignore monitor_path from helpers; we are not writing Monitor CSVs)
    _, model_path = helpers.generate_paths_monitor_model(output_prefix)
    model_path = os.path.join(run_dir, os.path.basename(model_path))

    # TensorBoard logs: central, low overhead
    tb_log_dir = os.path.join(run_dir, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    # Lightweight run metadata
    run_logger = RunLogger(run_dir)
    run_logger.log_params(model_params, env_params, run_params)

    # Vectorized envs (no per-env Monitor files)
    # Use forkserver/spawn to avoid fork-related issues on some clusters; SB3 supports this via vec_env_kwargs.
    vec_env = make_vec_env(
        lambda: create_env(env_params),
        n_envs=int(run_params["env_number"]),
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver" if os.name != "nt" else "spawn"},
    )

    # Keeps episode stats in info dicts; no file output
    vec_env = VecMonitor(vec_env)

    # PPO with TensorBoard logging; reduce stdout by verbose=0 above
    model = PPO(
        env=vec_env,
        **model_params,
        tensorboard_log=tb_log_dir,
    )

    # Callback: sparse prints + end-of-run CSV/PNG/summary
    callback = EpisodeCSVLogger(
        run_dir=run_dir,
        print_every_s=int(run_params.get("print_every_s", 20)),
        flush_every_episodes=int(run_params.get("flush_every_episodes", 50)),
        window=100,
    )

    # Train
    model.learn(
        total_timesteps=int(run_params["timesteps"]),
        tb_log_name=output_prefix,
        callback=callback,
        progress_bar=False,
        log_interval=50,  # how often SB3 dumps to TB; not stdout (verbose=0)
    )

    # Save model
    model.save(model_path)

    # Record artifact locations (relative)
    run_logger.log_artifacts(
        {
            "run_dir": run_dir,
            "model_path": model_path,
            "tensorboard_log_dir": tb_log_dir,
            "episode_log_csv": os.path.join(run_dir, "episode_log.csv"),
            "final_summary_json": os.path.join(run_dir, "final_summary.json"),
            "training_curve_png": os.path.join(run_dir, "training_curve.png"),
        }
    )


if __name__ == "__main__":
    main(DEFAULT_MODEL_PARAMS, DEFAULT_ENV_PARAMS, DEFAULT_RUN_PARAMS, output_prefix)
