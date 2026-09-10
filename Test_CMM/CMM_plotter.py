import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_results(time_array, y_hom, y_gr, y_label, title):
    plt.figure(figsize=(8, 5))
    plt.plot(time_array, y_hom, label="Homeostatic", linestyle="--")
    plt.plot(time_array, y_gr, label="Growth & Remodeling", linestyle="-")
    plt.xlabel("Time (days)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
