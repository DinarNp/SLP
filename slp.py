"""
Single Layer Perceptron (SLP) - Binary classification of Iris data
(Iris-setosa = 0, Iris-versicolor = 1)

The algorithm reproduces the Google Sheets template (PMM-TemplateSLP) exactly:
  * initial weights: bias = theta1..theta4 = 0.5
  * learning rate  : 0.1
  * epochs         : 5
  * z      = bias + sum(theta_i * x_i)
  * g(z)   = 1 / (1 + exp(-z))                 (sigmoid)
  * pred   = 1 if g(z) > 0.5 else 0
  * error  = g(z) - target,   SSE = error^2
  * dtheta_i = 2 * (g - y) * g * (1 - g) * x_i  (x_0 = 1 for the bias)
  * theta_i <- theta_i - lr * dtheta_i          (update after EVERY sample)

Data split (same order as the spreadsheet, no shuffling):
  * training   : setosa rows 1-40  + versicolor rows 51-90   (80 samples)
  * validation : setosa rows 41-50 + versicolor rows 91-100  (20 samples)

Validation for each epoch uses the weights obtained at the end of that
training epoch (no weight update during validation).

Per-epoch metrics:
  * loss     = mean of SSE over the samples in the epoch
  * accuracy = (#correct predictions) / (#samples)

Author : Dinar Nugroho Pratomo (26/591683/SPA/01279)
"""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "iris.csv")
OUT_DIR = os.path.join(HERE, "outputs")

LEARNING_RATE = 0.1
EPOCHS = 5
INITIAL_WEIGHT = 0.5
LABELS = {"Iris-setosa": 0, "Iris-versicolor": 1}


def load_data(path):
    """Read the 100 setosa/versicolor rows and split them like the spreadsheet."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            x = [float(r["sepal_length"]), float(r["sepal_width"]),
                 float(r["petal_length"]), float(r["petal_width"])]
            rows.append((x, LABELS[r["species"]]))
    train = rows[0:40] + rows[50:90]
    valid = rows[40:50] + rows[90:100]
    return train, valid


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def forward(w, x):
    """w = [bias, theta1, theta2, theta3, theta4]"""
    z = w[0] + sum(wi * xi for wi, xi in zip(w[1:], x))
    return sigmoid(z)


def train_one_epoch(w, data, lr):
    """Online (per-sample) gradient descent, exactly like the spreadsheet rows."""
    total_sse, correct = 0.0, 0
    for x, y in data:
        g = forward(w, x)
        pred = 1 if g > 0.5 else 0
        err = g - y
        total_sse += err ** 2
        correct += int(pred == y)
        common = 2 * (g - y) * (1 - g) * g
        grads = [common] + [common * xi for xi in x]
        w = [wi - lr * gi for wi, gi in zip(w, grads)]
    return w, total_sse / len(data), correct / len(data)


def evaluate(w, data):
    total_sse, correct = 0.0, 0
    for x, y in data:
        g = forward(w, x)
        pred = 1 if g > 0.5 else 0
        total_sse += (g - y) ** 2
        correct += int(pred == y)
    return total_sse / len(data), correct / len(data)


def plot(history, key_train, key_val, ylabel, title, filename, ylim, percent=False):
    """Line chart styled like the spreadsheet chart (same axis range & colours)."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(epochs, [h[key_train] for h in history], marker="o", color="#5B9BD5",
            label="Train " + ylabel.split()[0])
    ax.plot(epochs, [h[key_val] for h in history], marker="o", color="#ED7D31",
            label="Val " + ylabel.split()[0])
    for h in history:
        t, v = h[key_train], h[key_val]
        for val, color, up in ((t, "#2E75B6", t >= v), (v, "#C55A11", v > t)):
            if not up and val - ylim[0] < 0.08 * (ylim[1] - ylim[0]):
                up = True  # too close to the x-axis: put the label above the point
            txt = f"{val:.2%}" if percent else f"{val:.4f}"
            ax.annotate(txt, (h["epoch"], val), textcoords="offset points",
                        xytext=(0, 8 if up else -14), ha="center", fontsize=8, color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(epochs)
    ax.set_ylim(*ylim)
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train, valid = load_data(DATA_FILE)
    w = [INITIAL_WEIGHT] * 5

    history = []
    header = (f"{'Epoch':>5} | {'bias':>9} {'theta1':>9} {'theta2':>9} {'theta3':>9} {'theta4':>9}"
              f" | {'TrainLoss':>9} {'TrainAcc':>8} | {'ValLoss':>9} {'ValAcc':>8}")
    print(header)
    print("-" * len(header))
    for epoch in range(1, EPOCHS + 1):
        w, train_loss, train_acc = train_one_epoch(w, train, LEARNING_RATE)
        val_loss, val_acc = evaluate(w, valid)
        history.append(dict(epoch=epoch, bias=w[0], theta1=w[1], theta2=w[2],
                            theta3=w[3], theta4=w[4], train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))
        print(f"{epoch:>5} | " + " ".join(f"{v:9.5f}" for v in w) +
              f" | {train_loss:9.6f} {train_acc:8.4f} | {val_loss:9.6f} {val_acc:8.4f}")

    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plot(history, "train_acc", "val_acc", "Accuracy",
         "Accuracy - Training vs Validation (Python)", "accuracy_python.png",
         ylim=(0, 1.2), percent=True)
    plot(history, "train_loss", "val_loss", "Loss (mean SSE)",
         "Loss - Training vs Validation (Python)", "loss_python.png",
         ylim=(0, 0.5))
    print(f"\nSaved metrics.csv, accuracy_python.png, loss_python.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
