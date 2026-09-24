"""Generic line and bar plots; series are {label: y} dicts, styles are {label: kwargs}."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINESTYLES = {"FCMM": "-", "HCMM": "--", "HCMM dep": "--", "HCMM init": ":"}
COLORS = {"A": "tab:green", "B": "tab:purple", "C": "tab:orange", "D": "tab:red", "E": "tab:blue",
          "FCMM": "tab:blue", "HCMM": "tab:red", "HCMM dep": "tab:red", "HCMM init": "tab:orange"}


def style_for(label):
    """Color from the first known token, linestyle from the last."""
    tokens = label.split(" ", 1)
    color = COLORS.get(tokens[0], COLORS.get(label))
    ls = LINESTYLES.get(tokens[-1], LINESTYLES.get(label, "-"))
    return dict(color=color, linestyle=ls)


def plot_series(ax, t, series, title="", ylabel="", xlabel="Time (days)", legend=True):
    for label, y in series.items():
        ax.plot(t, y, label=label, **style_for(label))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if legend:
        ax.legend(fontsize=7, ncol=2)
    return ax


def panels(specs, path, ncols=2, size=(6.5, 5)):
    """specs: list of dict(t, series, title, ylabel) -> one figure, saved to path."""
    nrows = -(-len(specs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    for ax, s in zip(axes.flat, specs):
        plot_series(ax, **s)
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bars(groups, path, ylabel="", title=""):
    """groups: {group: [(tick_label, value), ...]} -> grouped bar chart."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x, ticks, labels = 0, [], []
    for i, (group, items) in enumerate(groups.items()):
        for tick, v in items:
            ax.bar(x, v, color=f"C{i}", label=group if tick == items[0][0] else None)
            ticks.append(x)
            labels.append(tick)
            x += 1
        x += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
