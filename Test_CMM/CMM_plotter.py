import matplotlib.pyplot as plt

# FCMM (full) vs HCMM (homogenized) get a fixed style each, so the same model
# reads the same way across every panel of a figure.
MODEL_STYLE = {
    "FCMM": dict(color="tab:blue", linestyle="-", marker=None),
    "HCMM": dict(color="tab:red", linestyle="--", marker=None),
}


def plot_model_comparison(time_array, series, y_label, title, ax=None,
                          save_path=None, show=None):
    """One panel comparing constrained-mixture models on the same quantity.

    series: dict {model_label: 1-D sequence over time}, e.g.
            {"FCMM": out_f["sigma_driven"], "HCMM": out_h["sigma_driven"]}.
    ax:     pass an existing Axes to build multi-panel figures; when omitted a
            new figure is made and shown.
    show:   defaults to True only when this call created the figure.
    """
    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(8, 5))
    if show is None:
        show = owns_figure

    for label, y in series.items():
        ax.plot(time_array, y, label=label,
                **MODEL_STYLE.get(label, dict(linestyle="-")))

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        ax.get_figure().savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return ax


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
