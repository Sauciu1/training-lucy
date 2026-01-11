import numpy as np
import matplotlib.pyplot as plt


def _plot_metric(
    df,
    values,
    timesteps,
    window,
    figsize,
    col,
    time_col,
    title,
    ylabel,
    color,
    avg_color,
    ax,
    show,
):
    """Generic plotting helper."""
    if values is None and df is not None:
        values = df[col].values
    if timesteps is None:
        timesteps = (
            df[time_col].cumsum().values
            if df is not None and time_col in df.columns
            else np.arange(len(values))
        )
    if window is None:
        window = min(100, len(values) // 10 + 1)

    fig, ax = (ax.get_figure(), ax) if ax else plt.subplots(figsize=figsize)
    ax.plot(timesteps, values, alpha=0.3, color=color, label=ylabel)

    if len(values) >= window:
        avg = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(
            timesteps[window - 1 :],
            avg,
            color=avg_color,
            linewidth=2,
            label=f"Rolling Avg ({window} eps)",
        )

    ax.set(xlabel="Timesteps", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax


def plot_rewards(
    df=None,
    rewards=None,
    timesteps=None,
    window=None,
    figsize=(7, 5),
    ax=None,
    show=True,
):
    return _plot_metric(
        df,
        rewards,
        timesteps,
        window,
        figsize,
        "r",
        "t",
        "Training Reward Over Time",
        "Episode Reward",
        "blue",
        "red",
        ax,
        show,
    )


def plot_lengths(
    df=None,
    lengths=None,
    timesteps=None,
    window=None,
    figsize=(7, 5),
    ax=None,
    show=True,
):
    return _plot_metric(
        df,
        lengths,
        timesteps,
        window,
        figsize,
        "l",
        "t",
        "Episode Length Over Time",
        "Episode Length",
        "green",
        "orange",
        ax,
        show,
    )


def iqr_filter(rewards, timesteps=None, multiplier=20):
    """
    Returns a boolean mask for rewards within [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    If timesteps is provided, ensures the mask matches the length of timesteps.
    """
    rewards = np.asarray(rewards)
    q1 = np.percentile(rewards, 25)
    q3 = np.percentile(rewards, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (rewards >= lower) & (rewards <= upper)
    if timesteps is not None:
        timesteps = np.asarray(timesteps)
        if len(mask) != len(timesteps):
            raise ValueError("rewards and timesteps must have the same length")
    return mask

def plot_training_progress(
    df,
    reward_col="r",
    length_col="l",
    timestep_col="t",
    window=50,
    figsize=(14, 5),
):
    """
    Plots training progress using a DataFrame and specified column names.
    Requires df and the column names for rewards, lengths, and timesteps.
    """
    if df is None:
        raise ValueError("df must be provided and not None.")
    for col in [reward_col, length_col, timestep_col]:
        if col not in df:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    rewards = np.asarray(df[reward_col])
    lengths = np.asarray(df[length_col])
    timesteps = np.asarray(df[timestep_col])

    # Filter out NaNs
    valid_mask = ~(
        np.isnan(rewards) | np.isnan(lengths) | np.isnan(timesteps)
    )
    rewards = rewards[valid_mask]
    lengths = lengths[valid_mask]
    timesteps = timesteps[valid_mask]

    # Optionally filter out extreme outliers in rewards
    mask = iqr_filter(rewards)
    rewards = rewards[mask]
    lengths = lengths[mask]
    timesteps = timesteps[mask]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _plot_metric(None, rewards, timesteps, window, figsize, reward_col, timestep_col, "Training Reward Over Time", "Episode Reward", "blue", "red", axes[0], show=False)
    _plot_metric(None, lengths, timesteps, window, figsize, length_col, timestep_col, "Episode Length Over Time", "Episode Length", "green", "orange", axes[1], show=False)
    plt.tight_layout()
    plt.show()
    return fig, axes




def print_training_summary(df=None, rewards=None, lengths=None, last_n=100):
    r = rewards if rewards is not None else df["r"].values
    l = lengths if lengths is not None else df["l"].values if df is not None else None
    print(f"Training Summary:")
    print(f"Total episodes: {len(r)}")
    print(f"Final avg reward (last {last_n} eps): {np.mean(r[-last_n:]):.2f}")
    print(f"Max reward: {np.max(r):.2f}")

    if l is not None:
        print(f"  Final avg length (last {last_n} eps): {np.mean(l[-last_n:]):.1f}")


from datetime import datetime
import os
from src import enforce_absolute_path


def generate_paths_monitor_model(prefix: str) -> tuple[str, str]:
    time_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M")

    monitor_dir = enforce_absolute_path(
        os.path.join("logs", f"{prefix}_{time_suffix}")
    )

    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir, exist_ok=True)

    model_path = enforce_absolute_path(
        os.path.join(
            "trained_models", f"{prefix}_{time_suffix}"
        )
    )
    return monitor_dir, model_path




if __name__ == "__main__":
    print("This is a helper module and is not meant to be run directly.")


