import numpy as np
import matplotlib.pyplot as plt

def plot_loss(epochs, train_loss, test_loss, title):
    plt.figure(figsize=(8 ,8))
    x = np.arange(1 ,epochs +1)
    plt.plot(x, train_loss, label = 'Training Loss')
    plt.plot(x, test_loss, label = 'Validation Loss')
    plt.xlabel('Epochs', fontsize =16)
    plt.ylabel('Loss', fontsize =16)
    plt.title(title ,fontsize =16)
    plt.legend(fontsize=16)


def plot_acc(epochs ,test_acc):
    plt.figure(figsize=(8 ,8))
    x = np.arange(1 ,epochs +1)
    plt.plot(x, test_acc)
    plt.xlabel('Epochs', fontsize =16)
    plt.ylabel('Test Accuracy', fontsize =16)
    plt.title('Test Accuracy v/s Epochs' ,fontsize =16)

