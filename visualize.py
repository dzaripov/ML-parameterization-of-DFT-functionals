import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc


dir_path = os.path.dirname(os.path.realpath(__file__))


path = dir_path + '\\' + sys.argv[1]

size = 11.5
lw = 0.7

ticks = size/16
tick_length = size/4.5


def strip_name(filename):
    functional, num_layers, h_dim, dropout, omega = filename.split("_")[:5]
    return f"{functional} {num_layers}x{h_dim}, dropout = {dropout}, omega = {omega}"


def set_style():
    sns.set_context("paper")

    sns.set_style("ticks", {
        "font.family": "serif",
        "font.serif": ["Helvetica", "sans-serif"]
    })


def extract_history(name, function="Full"):
    with open(name, "r") as file:
        counter = 0
        test_loss = []
        train_loss = []
        train_local_loss = []
        test_local_loss = []
        for line in file:
            if line.strip() == "Epoch 1":
                counter += 1
            if line.startswith("test") and counter == 2 and "Local" not in line:
                loss = float(line.split()[-1])
                test_loss.append(loss)
            elif line.startswith("train") and counter == 2 and "Local" not in line:
                loss = float(line.split()[-1])
                train_loss.append(loss)
            elif line.startswith("train Local"):
                train_local_loss.append(float(line.strip().split(" = ")[1]))
            elif line.startswith("test Local"):
                test_local_loss.append(float(line.strip().split(" = ")[1]))
        test_loss = np.array(test_loss)
        train_loss = np.array(train_loss)
        train_local_loss = np.array(train_local_loss)
        test_local_loss = np.array(test_local_loss)
        functional, num_layers, h_dim, dropout, omega = name.split("_")[:5]
        omega = float(omega)
        full_train_loss = (1 - omega) / 5 * train_loss + omega * 100 * train_local_loss
        full_test_loss = (1 - omega) / 5 * test_loss + omega * 100 * test_local_loss
    size = min(len(test_loss), len(train_loss))
    result_dict = {
        "Reaction": {"train": train_loss[:size], "test": test_loss[:size]},
        "Local": {"train": train_local_loss[:size], "test": test_local_loss[:size]},
        "Full": {"train": full_train_loss[:size], "test": full_test_loss[:size]},
    }
    return result_dict[function]


NAME_DICT = {
    "Reaction": "Reaction MAE loss",
    "Local": "Local loss",
    "Full": "Full loss",
}


def plot_results(ax, name, function):
    result_dict = extract_history(f"{sys.argv[1]}/{name}", function)
    train_loss, test_loss = result_dict["train"], result_dict["test"]
    n_epochs = len(train_loss)
    if "PBE" in name:
        ax.plot(
            range(1, n_epochs + 1), train_loss, label="Train Loss", lw=lw, color="blue"
        )
        ax.plot(
            range(1, n_epochs + 1),
            test_loss,
            label="Validation Loss",
            alpha=0.7,
            lw=lw,
            color="green",
        )
        major_tick_range = 100
        minor_tick_range = 50
    else:
        ax.plot(
            range(1, n_epochs + 1), train_loss, label="Train Loss", lw=lw/2, color="blue"
        )
        ax.plot(
            range(1, n_epochs + 1),
            test_loss,
            label="Validation Loss",
            alpha=0.7,
            lw=lw/2,
            color="green",
        )
        major_tick_range = 300
        minor_tick_range = 100

    major_ticks = np.arange(0, n_epochs, major_tick_range)
    minor_ticks = np.arange(0, n_epochs, minor_tick_range)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_tick_params(width=ticks, length=tick_length)
    ax.yaxis.set_tick_params(width=ticks, length=tick_length)
    ax.xaxis.set_tick_params(width=ticks*2/3, length=tick_length*2/3, which='minor')
    ax.yaxis.set_tick_params(width=ticks*2/3, length=tick_length*2/3, which='minor')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


    ax.set_xlim(0, n_epochs + 1)
    if "8x32" in name and "XALPHA" in name:
        ax.set_ylim([5, 70])
    if "16x32" in name and "XALPHA" in name:
        ax.set_ylim([5, 50])
    if "32x32" in name and "XALPHA" in name:
        ax.set_ylim([5, 55])
    if "4x512" in name and "XALPHA" in name:
        ax.set_ylim([5, 40])
    if "PBE" in name and function == "Reaction":
        ax.plot(
            [0, n_epochs + 1],
            [7.85833597, 7.85833597],
            label="PBE Train Loss",
            color="blue",
            linestyle="--",
            lw=lw,
        )
        ax.plot(
            [0, n_epochs + 1],
            [7.64929252, 7.64929252],
            label="PBE Validation Loss",
            color="green",
            linestyle="--",
            lw=lw,
        )
    elif function == "Reaction":
        ax.plot(
            [0, n_epochs + 1],
            [16.477398, 16.477398],
            label="Xα Train Loss",
            color="blue",
            linestyle="--",
            lw=lw,
        )
        ax.plot(
            [0, n_epochs + 1],
            [17.32020, 17.32020],
            label="Xα Validation Loss",
            color="green",
            linestyle="--",
            lw=lw,
        )
    return min(min(train_loss), min(test_loss)), max(max(train_loss), max(test_loss))

filenames = list(os.walk(path))[0][2]

offset = len(filenames)
set_style()
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('axes', linewidth=ticks)
filenames.sort(key=lambda x: ("_".join(x.split("_")[:3]), float(x.split("_")[4])))
for i in range(0, len(filenames), offset):
    mins_reaction = []
    maxs_reaction = []
    mins_local = []
    maxs_local = []
    names = filenames[i: i + offset]
    fig, ax = plt.subplots(2, offset, figsize=[7, 4], dpi=1200, sharey='row', sharex='col')
    title = " ".join(strip_name(filenames[i]).split()[:5])

    for j in range(offset):
        mn, mx = plot_results(ax[0][j], names[j], "Reaction")
        mins_reaction.append(mn)
        maxs_reaction.append(mx)
        mn, mx = plot_results(ax[1][j], names[j], "Local")
        mins_local.append(mn)
        maxs_local.append(mx)
    min_local = min(mins_local) * 0.9
    max_local = (
        min(max(maxs_local) * 1.1, 0.2)
        if "PBE" in title
        else min(max(maxs_local) * 1.1, 0.5)
    )
    min_reaction = min(mins_reaction) * 0.9
    max_reaction = (
        min(max(maxs_reaction) * 1.1, 55)
        if "XALPHA" in title
        else min(max(maxs_reaction) * 1.1, 15)
    )
    for j in range(offset):
        title = "".join(strip_name(filenames[j]).split()[7])
        ax[0][j].set_title(f"Ω={title}", fontsize=size, y=1.05)
        ax[0][j].set_ylim([min_reaction, max_reaction])
        ax[1][j].set_ylim([min_local, max_local])
        ax[0, j].tick_params(axis="both", labelsize=size**2/20, pad=size/5)
        ax[1, j].tick_params(axis="both", labelsize=size**2/20, pad=size/5)


fig.text(0.5, 0.04, 'Epoch', ha='center', size=size)
fig.text(0.04, 0.95, 'Reaction MAE Loss, kcal/mol', va='top', rotation='vertical', size=size*0.9)
fig.text(0.04, 0.12, 'Local Loss, kcal/mol', va='bottom', rotation='vertical', size=size*0.9)

blue = Line2D([0], [0], color='blue', lw=lw*2, label='Train')
green = Line2D([0], [0], color='green', lw=lw*2, label='Validation')

straight = Line2D([0], [0], color='black', lw=lw*2, label='NN')
dashed = Line2D([0], [0], color='black', lw=lw*2, label='Non-NN', linestyle='dashed')

train_test = plt.legend(handles=[blue, green], loc='lower right', fontsize=size*5/6, fancybox=False, title="Dataset", title_fontsize=size, bbox_to_anchor=(0.92, -0.06), bbox_transform=fig.transFigure, ncol=2, frameon=False)
NN_non_NN = plt.legend(handles=[straight, dashed], loc='lower left', fontsize=size*5/6, fancybox=False, title="Functional", title_fontsize=size, bbox_to_anchor=(0.06, -0.06), bbox_transform=fig.transFigure, ncol=2, frameon=False)
fig.add_artist(train_test)
fig.add_artist(NN_non_NN)

functional = filenames[0].split("_")[0]
N = len(filenames)

plt.savefig(f"Figure_plot_{functional}-{N}.png", dpi=1200, bbox_inches='tight')
