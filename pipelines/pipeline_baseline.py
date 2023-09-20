import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from data import process_scan
from utils import config
from eval import plot_history
import pickle

BASE_PATH = "../data/alzheimer/OASIS"
IMAGE_PATH = os.path.join(BASE_PATH, "OAS2_RAW_PART1")
CSV_PATH = os.path.join(BASE_PATH, "oasis_longitudinal_demographics.xlsx")


def load_data():
    dataframe = pd.read_excel(CSV_PATH)[:config['n_samples']]

    valid_mri_ids = [
        scan for patient in dataframe["Subject ID"].unique()
        for scan in sorted(dataframe[dataframe["Subject ID"] == patient]["MRI ID"])[:2]
    ]

    file_paths = [os.path.join(IMAGE_PATH, mri_id, "RAW", "mpr-1.nifti.img") for mri_id in valid_mri_ids]

    label_encoder = LabelEncoder()
    dataframe["Encoded Group"] = label_encoder.fit_transform(dataframe["Group"])
    labels = dataframe[dataframe["MRI ID"].isin(valid_mri_ids)]["Encoded Group"].reset_index(drop=True)

    return file_paths, labels


class DataGenerator(Sequence):
    def __init__(self, paths, labels, batch_size=config['batch_size'], dim=(224, 224), n_channels=3,
                 n_classes=3, shuffle=True, n_slices=3):
        self.paths = paths
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_slices = n_slices
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        paths_temp = [self.paths[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(paths_temp, labels_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, paths_temp, labels_temp):
        X = np.empty((self.batch_size * self.n_slices, *self.dim, self.n_channels))
        y = np.empty((self.batch_size * self.n_slices, self.n_classes), dtype=int)

        for i, (path, label) in enumerate(zip(paths_temp, labels_temp)):
            # Use the process_scan function to process the volume
            vol = process_scan(path)
            slice_indices = np.linspace(0, vol.shape[0] - 1, self.n_slices).astype(int)

            for j, slice_idx in enumerate(slice_indices):
                resized_slice = resize(vol[slice_idx], self.dim, mode='constant', preserve_range=True)
                single_channel_slice = np.expand_dims(resized_slice.squeeze(), axis=-1)
                three_channel_slice = np.repeat(single_channel_slice, 3, axis=-1)
                X[i * self.n_slices + j,] = three_channel_slice

                y[i * self.n_slices + j] = to_categorical(label, num_classes=self.n_classes)

        return X, y


def plot_evaluation(model_path, test_generator):
    output_path = "../results"
    model = load_model(model_path)
    filename = os.path.basename(model_path)
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
    print("Evaluation results:", evaluation_results)
    # Compute the confusion matrix
    y_true[-1] = 0
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
    plt.savefig(os.path.join(save_path, f'{filename}_mat.png'))
    plt.show()


def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(config['num_classes'], activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    paths, labels = load_data()

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(paths, labels, test_size=0.2)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5)

    train_generator = DataGenerator(train_paths, train_labels)
    val_generator = DataGenerator(val_paths, val_labels, shuffle=False)
    test_generator = DataGenerator(test_paths, test_labels, shuffle=False)

    model = build_model()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['epochs'],
        callbacks=[
            ModelCheckpoint(filepath='weights/baseline.h5', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5, min_lr=config['min_learning_rate']),
            EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
        ],
        workers=8
    )

    with open('../history/baseline.pkl', 'wb') as history_file:
        pickle.dump(history.history, history_file)

    plot_evaluation('weights/baseline.h5', test_generator)
    plot_history('../history/baseline.pkl')
