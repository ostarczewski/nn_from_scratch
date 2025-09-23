import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise


def show_grid(X, y, class_names, grid_size=None, cmap='grey'):
    idx = range(len(X))
    images, labels = X[idx], y[idx]

    if not grid_size:
        grid_size = int(len(X)**(1/2))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i]

        if img.shape[0] == 1:   # grayscale (C=1)
            img = img[0]        # (H, W)
            ax.imshow(img, cmap=cmap)
        else:                   # RGB or multi-channel
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
            ax.imshow(img)

        ax.set_title(class_names[np.argmax(labels[i])])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_class_balance(y_list, labels, tilt=0):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    titles = ["Train data",
              "Validation data",
              "Test data"]

    for y, ax, title in zip(y_list, axes, titles):
        ax.bar(labels, np.bincount(y) / len(y))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=tilt)
        ax.set_xlabel('Class')
        ax.set_ylabel('Density')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_hist(history):
    fig, ax = plt.subplots(1,2, figsize=(14,6))
    epochs = range(1, len(history['loss'])+1)

    # plot loss
    ax[0].plot(epochs, history['loss'], label='Training loss')
    if history['val_loss']:
        ax[0].plot(epochs, history['val_loss'], label='Validation loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    # plot acc
    ax[1].plot(epochs, history['acc'], label='Training accuracy')
    if history['val_acc']:
        ax[1].plot(epochs, history['val_acc'], label='Validation accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    plt.show()


def evaluate_classification(y_true, y_pred, labels = [str(i) for i in range(10)]):
    y_true, y_pred = np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)
    
    # classification report
    print(classification_report(y_true, y_pred, digits=4, target_names=labels))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # row-normalized
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()    


def transform_mnist(img):
    # random small rotation (-15° to +15°)
    angle = np.random.uniform(-15, 15)
    img = rotate(img, angle, mode="edge")

    # random shift & scale
    tform = AffineTransform(
        # shift up to 2 pixels
        translation=(np.random.uniform(-2, 2), np.random.uniform(-2, 2)),
        # +/- 10% zoom
        scale=(np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1))
    )
    img = warp(img, tform.inverse, mode="edge")

    # add random noise (40% chance)
    if np.random.random() < 0.4:
        img = random_noise(img, mode="gaussian", rng=42, var=0.01)

    return img.astype(np.float32)


def transform_mnist_dataset(X):
    batch = X.shape[0]
    X_aug = np.empty_like(X, dtype=np.float32)  # preallocate output

    for i in range(batch):
        X_aug[i, 0] = transform_mnist(X[i, 0])

    return X_aug


def transform_f_mnist(img):
    # random shift & scale
    tform = AffineTransform(
        scale=(np.random.uniform(0.9, 1), np.random.uniform(0.9, 1)),
        translation=(np.random.uniform(-3, 3), np.random.uniform(-1, 1))
    )
    img = warp(img, tform.inverse, mode="constant")

    # add random noise
    if np.random.random() < 0.3:
        img = random_noise(img, mode="gaussian", rng=42, var=0.0009)

    if np.random.random() < 0.5:
        img = np.fliplr(img)

    return img.astype(np.float32)


def transform_f_mnist_dataset(X):
    batch = X.shape[0]
    X_aug = np.empty_like(X, dtype=np.float32)  # preallocate output

    for i in range(batch):
        X_aug[i, 0] = transform_f_mnist(X[i, 0])

    return X_aug


def transform_cifar(img):
    # random shift & scale
    tform = AffineTransform(
        translation=(np.random.uniform(-4, 4), np.random.uniform(-4, 4))
    )
    img = warp(img, tform.inverse, mode="constant")

    if np.random.random() < 0.5:
        img = np.fliplr(img)

    return img.astype(np.float32)


def transform_cifar_dataset(X: np.ndarray):
    batch = X.shape[0]
    X = X.transpose(0, 2, 3, 1)  # C last
    X_aug = np.empty_like(X, dtype=np.float32)  # preallocate output

    for i in range(batch):
        X_aug[i] = transform_cifar(X[i])

    return X_aug.transpose(0, 3, 1, 2)  # back to C first
