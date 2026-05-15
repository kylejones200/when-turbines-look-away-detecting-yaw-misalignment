#!/usr/bin/env python3
import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

np.random.seed(42)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
FPS, N_FRAMES = 10, 100
n_samples = 400

# Generate yaw data
true_wind_dir = 180 + 30 * np.sin(2 * np.pi * np.arange(n_samples) / 150)
yaw_angle = true_wind_dir + np.random.normal(0, 5, n_samples)
# Add misalignment periods
yaw_angle[100:150] += 15
yaw_angle[250:300] -= 20

misalignment = yaw_angle - true_wind_dir
power_loss = 100 * (1 - np.cos(np.radians(misalignment)))

fig = plt.figure(figsize=(14, 8), facecolor="white")
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1, ax2, ax3 = (
    fig.add_subplot(gs[0, :]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
)

for ax in [ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def update(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    end_idx = int((frame / N_FRAMES) * n_samples)

    ax1.plot(
        true_wind_dir[:end_idx], "black", linewidth=2, label="Wind Direction", alpha=0.6
    )
    ax1.plot(yaw_angle[:end_idx], "gray", linewidth=2, label="Yaw Angle")
    ax1.set_xlabel("Time", fontsize=10)
    ax1.set_ylabel("Angle (degrees)", fontsize=10)
    ax1.set_title(
        f"Yaw vs Wind Direction - Frame {frame + 1}/{N_FRAMES}",
        fontsize=11,
        fontweight="normal",
    )
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    if end_idx > 20:
        ax2.plot(misalignment[:end_idx], "gray", linewidth=1.5)
        ax2.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax2.axhline(10, color="red", linestyle="--", alpha=0.3)
        ax2.axhline(-10, color="red", linestyle="--", alpha=0.3)
        ax2.set_xlabel("Time", fontsize=10)
        ax2.set_ylabel("Misalignment (degrees)", fontsize=10)
        ax2.set_title("Yaw Misalignment", fontsize=11, fontweight="normal")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    current_misalign = misalignment[min(end_idx - 1, n_samples - 1)]
    status = "MISALIGNED!" if abs(current_misalign) > 10 else "Aligned"
    color = "red" if abs(current_misalign) > 10 else "green"
    ax3.text(
        0.5,
        0.7,
        status,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=color,
        transform=ax3.transAxes,
    )
    ax3.text(
        0.5,
        0.4,
        f"Error: {current_misalign:.1f}°",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax3.transAxes,
    )
    ax3.text(
        0.5,
        0.2,
        f"Power Loss: {power_loss[min(end_idx - 1, n_samples - 1)]:.1f}%",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax3.transAxes,
    )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("Status", fontsize=11, fontweight="normal")
    return []



def main():
    logger.info("Creating animation for Article 38...")
    anim = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True
    )
    anim.save("38_yaw_misalignment_animation.gif", writer="pillow", fps=FPS, dpi=100)
    logger.info("✓ Animation saved: 38_yaw_misalignment_animation.gif")
    plt.close()


if __name__ == "__main__":
    main()
