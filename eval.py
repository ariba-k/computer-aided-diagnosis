import matplotlib.pyplot as plt
from utils import config
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import load_model
import pickle
import os
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

output = {"Combined": "model_combined",
          "3D CNN": "model_3D_cnn_only",
          "CSV": "model_csv_only",
          "3D CNN Diff": "model_3D_cnn_and_diff",
          "3D CNN CSV": "model_3D_cnn_and_csv",
          "Baseline": "baseline"}

output_path = "results"


def plot_eval_gen(path, test_generator, model=None):
    if model:
        model.load_weights(path)
    else:
        model = load_model(path)

    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    save_path = os.path.join(output_path, filename)
    os.makedirs(save_path, exist_ok=True)
    class_labels = ["Converted", "Demented", "Nondemented"]

    test_pred = model.predict(test_generator)
    y_pred = np.argmax(test_pred, axis=1)

    # Collect all the true labels from test_generator
    y_true = []
    for i in range(len(test_generator)):
        _, labels_batch = test_generator[i]
        y_true.extend(np.argmax(labels_batch, axis=1))
    y_true = np.array(y_true)
    evaluation_results = model.evaluate(test_generator)
    print(f"{filename}:", evaluation_results)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

    # Calculate AUC
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_binary = lb.transform(y_true)
    y_pred_binary = lb.transform(y_pred)
    auc_score = roc_auc_score(y_true_binary, y_pred_binary, average='weighted', multi_class='ovr')
    print(f"AUC: {auc_score}")

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Create the confusion matrix plot using seaborn
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', cbar=True, ax=ax)

    # Set labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    # Set x-axis and y-axis tick labels
    ax.xaxis.set_ticklabels(class_labels, fontsize=6)
    ax.yaxis.set_ticklabels(class_labels, fontsize=6)

    # Rotate the tick labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Show the plot
    plt.savefig(os.path.join(save_path, f'{filename}_mat_rev.png'))
    plt.show()


def plot_eval(path, test, labels=None, test_indices=None):
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    save_path = os.path.join(output_path, filename)
    os.makedirs(save_path, exist_ok=True)

    model = load_model(path)
    class_labels = ["Converted", "Demented", "Nondemented"]
    if isinstance(test, tuple):
        X_test, y_test = test
        evaluation_results = model.evaluate(X_test, y_test)
        y_pred = [class_labels[np.argmax(x)] for x in model.predict(X_test, batch_size=config['batch_size'])]
        y_true = [class_labels[np.argmax(x)] for x in y_test]

    else:
        evaluation_results = model.evaluate(test)
        y_pred = [class_labels[np.argmax(x)] for x in model.predict(test, batch_size=config['batch_size'])]
        y_true = [class_labels[labels[idx]] for idx in test_indices]

    print(f"{filename}:", evaluation_results)
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

    # Calculate AUC
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_binary = lb.transform(y_true)
    y_pred_binary = lb.transform(y_pred)
    auc_score = roc_auc_score(y_true_binary, y_pred_binary, average='weighted', multi_class='ovr')
    print(f"AUC: {auc_score}")

    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Create the confusion matrix plot using seaborn
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', cbar=True, ax=ax)

    # Set labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    # Set x-axis and y-axis tick labels
    ax.xaxis.set_ticklabels(class_labels, fontsize=6)
    ax.yaxis.set_ticklabels(class_labels, fontsize=6)

    # # Rotate the tick labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Show the plot
    plt.savefig(os.path.join(save_path, f'{filename}_mat_rev.png'))
    plt.show()


def plot_history(path):
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    save_path = os.path.join(output_path, filename)
    os.makedirs(save_path, exist_ok=True)

    with open(path, 'rb') as f:
        history = pickle.load(f)
    if 'acc' in history.keys():
        acc_metric = "acc"
    else:
        acc_metric = "accuracy"

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate([acc_metric, 'loss']):
        ax[i].plot(history[metric])
        ax[i].plot(history[f'val_{metric}'])
        ax[i].set_title(f'Model {metric}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(metric)
        ax[i].legend(['train', 'val'])
    plt.savefig(os.path.join(save_path, f'{filename}_hist_rev.png'))
    plt.show()
