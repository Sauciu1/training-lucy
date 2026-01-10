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


def plot_training_progress(
    df=None, rewards=None, lengths=None, timesteps=None, window=None, figsize=(14, 5)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_rewards(df, rewards, timesteps, window, ax=axes[0], show=False)
    plot_lengths(df, lengths, timesteps, window, ax=axes[1], show=False)
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
