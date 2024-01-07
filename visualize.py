import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path = " ".join(sys.argv[1:])


def strip_name(filename):
    functional, num_layers, h_dim, dropout, omega = filename.split("_")[:5]
    return f"{functional} {num_layers}x{h_dim}, dropout = {dropout}, omega = {omega}"


def set_style():
    sns.set_context("paper")

    sns.set(font="serif")

    sns.set_style(
        "whitegrid",
        {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )


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
    result_dict = extract_history(f"{path}/{name}", function)
    train_loss, test_loss = result_dict["train"], result_dict["test"]
    n_epochs = len(train_loss)
    if "PBE" in name:
        ax.plot(
            range(1, n_epochs + 1), train_loss, label="Train Loss", lw=4, color="blue"
        )
        ax.plot(
            range(1, n_epochs + 1),
            test_loss,
            label="Validation Loss",
            alpha=0.7,
            lw=4,
            color="green",
        )
    else:
        ax.plot(
            range(1, n_epochs + 1), train_loss, label="Train Loss", lw=2, color="blue"
        )
        ax.plot(
            range(1, n_epochs + 1),
            test_loss,
            label="Validation Loss",
            alpha=0.7,
            lw=2,
            color="green",
        )
    ax.grid()
    ax.set_xlabel("Epoch", fontsize=30)

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
            lw=4,
        )
        ax.plot(
            [0, n_epochs + 1],
            [7.64929252, 7.64929252],
            label="PBE Validation Loss",
            color="green",
            linestyle="--",
            lw=4,
        )
    elif function == "Reaction":
        ax.plot(
            [0, n_epochs + 1],
            [16.477398, 16.477398],
            label="Xα Train Loss",
            color="blue",
            linestyle="--",
            lw=4,
        )
        ax.plot(
            [0, n_epochs + 1],
            [17.32020, 17.32020],
            label="Xα Validation Loss",
            color="green",
            linestyle="--",
            lw=4,
        )
    return min(min(train_loss), min(test_loss)), max(max(train_loss), max(test_loss))


filenames = list(os.walk(path))[0][2]

offset = len(filenames)
set_style()
filenames.sort(key=lambda x: ("_".join(x.split("_")[:3]), float(x.split("_")[4])))
for i in range(0, len(filenames), offset):
    mins_reaction = []
    maxs_reaction = []
    mins_local = []
    maxs_local = []
    names = filenames[i : i + offset]
    fig, ax = plt.subplots(2, offset, figsize=[60, 30])
    title = " ".join(strip_name(filenames[i]).split()[:5])

    for j in range(offset):
        mn, mx = plot_results(ax[0][j], names[j], "Reaction")
        mins_reaction.append(mn)
        maxs_reaction.append(mx)
        mn, mx = plot_results(ax[1][j], names[j], "Local")
        mins_local.append(mn)
        maxs_local.append(mx)
        ax[0, j].grid()
        ax[1, j].grid()
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
        title = " ".join(strip_name(filenames[j]).split()[7])
        ax[0][j].set_title(f"Ω = {title}", fontsize=50, y=1.05)
        ax[0][j].set_ylim([min_reaction, max_reaction])
        ax[1][j].set_ylim([min_local, max_local])
        ax[0, j].tick_params(axis="x", labelsize=20)
        ax[1, j].tick_params(axis="x", labelsize=20)
        ax[0, j].tick_params(axis="y", labelsize=20)
        ax[1, j].tick_params(axis="y", labelsize=20)

    ax[0][0].set_ylabel("Reaction MAE Loss, kcal/mol", fontsize=40)
    ax[1][0].set_ylabel("Local Loss, kcal/mol", fontsize=40)
plt.savefig("Figure.jpg", dpi=600, format="jpg")
