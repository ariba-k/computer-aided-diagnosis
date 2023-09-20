import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
from data import process_scan, random_rotate, create_paired_indices
from model import get_combined_model
from utils import config
from eval import plot_eval_gen, plot_history, plot_eval

base_path = "../data/alzheimer/OASIS"
base_image_path = os.path.join(base_path, "OAS2_RAW_PART1")
base_csv_path = os.path.join(base_path, "oasis_longitudinal_demographics.xlsx")

seed = 27
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

data = pd.read_excel(base_csv_path)[:config['n_samples']]
le = LabelEncoder()
data["Group Encoded"] = le.fit_transform(data["Group"])


def create_data(df):
    df = df.copy()
    valid_mri_ids = []

    for patient_id in df["Subject ID"].unique():
        # Get all MRI scans for the current patient
        patient_scans = sorted([mri_id for mri_id in df[df["Subject ID"] == patient_id]["MRI ID"]])
        # Add the first two scans to the list
        valid_mri_ids += patient_scans[:2]

    paths = [os.path.join(*[base_image_path, mri_id, "RAW", "mpr-1.nifti.img"]) for mri_id in
             valid_mri_ids]

    labels = df[df["MRI ID"].isin(valid_mri_ids)]["Group Encoded"]
    labels = labels.reset_index(drop=True)

    return paths, labels


paths, labels = create_data(data)
train_indices, val_indices, test_indices = create_paired_indices(paths, labels)


class PairCSVDataGenerator(tf.keras.utils.Sequence):
    """Generates pairs of MRI scans (MR1, MR2) for the same subject and corresponding CSV data."""

    def __init__(self, indices, paths, labels, csv_data, batch_size=config['batch_size'], dim=config['dim'],
                 n_classes=config['n_classes'], shuffle=True, transform=None):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.paths = paths
        self.labels = labels
        self.indices = indices
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()
        self.subject_index_map = self._create_subject_index_map()
        self.csv_data = csv_data

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        indices_temp = [self.indices[k] for k in indexes]

        # Generate data
        X1, X2, y = self.__data_generation(indices_temp)
        csv_batch = self.csv_data[index * self.batch_size:(index + 1) * self.batch_size]

        return [X1, X2, csv_batch], y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _create_subject_index_map(self):
        subject_index_map = {}
        for index in self.indices:
            subject_id = self._get_subject_id(self.paths[index])
            subject_index_map[subject_id] = subject_index_map.get(subject_id, []) + [index]
        return subject_index_map

    @staticmethod
    def _get_subject_id(path):
        # Split the path by '/' and get the subject ID (e.g., OAS2_0001) from the corresponding part of the path
        return path.split('/')[4][:-4]

    def __data_generation(self, indices_temp):
        """Generates data containing batch_size pairs of MRI scans and corresponding CSV data"""
        # Initialization
        if config['weights']:
            X1 = np.empty((self.batch_size, *self.dim, 3))
            X2 = np.empty((self.batch_size, *self.dim, 3))
        else:
            X1 = np.empty((self.batch_size, *self.dim, 1))
            X2 = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty(self.batch_size, dtype=int)

        for i, ID in enumerate(indices_temp):
            # Store sample
            subject_id = self._get_subject_id(self.paths[ID])
            scan_indices = self.subject_index_map[subject_id]
            volume1 = process_scan(self.paths[scan_indices[0]])
            volume2 = process_scan(self.paths[scan_indices[1]])

            if self.transform is not None:
                volume1 = self.transform(volume1)
                volume2 = self.transform(volume2)

            if config['weights']:
                X1[i,] = np.repeat(volume1[..., np.newaxis], 3, axis=-1) if volume1.ndim == 3 else volume1
                X2[i,] = np.repeat(volume2[..., np.newaxis], 3, axis=-1) if volume2.ndim == 3 else volume2
            else:
                X1[i,] = volume1[..., np.newaxis] if volume1.ndim == 3 else volume1
                X2[i,] = volume2[..., np.newaxis] if volume2.ndim == 3 else volume2

            y[i] = self.labels[ID]

        return X1, X2, to_categorical(y, num_classes=self.n_classes)


def create_data_arrays(df):
    features = df.copy()
    labels = features.pop('Group Encoded')
    labels = to_categorical(labels, num_classes=config['n_classes'])

    features = features.drop(['Subject ID', 'MRI ID', 'Group', 'Visit'], axis=1)
    features = pd.get_dummies(features, drop_first=True)
    features.fillna(features.median(), inplace=True)

    features = np.array(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_generators():
    train, val, test = create_data_arrays(data)

    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    train_indices, val_indices, test_indices = create_paired_indices(paths, labels)

    print(
        'Number of samples:\n'
        f'train: {len(train_indices)}\n'
        f'validation: {len(val_indices)}\n'
        f'test: {len(test_indices)}'
    )

    train_generator = PairCSVDataGenerator(train_indices, paths, labels, X_train, batch_size=config['batch_size'],
                                           shuffle=True, transform=random_rotate)
    valid_generator = PairCSVDataGenerator(val_indices, paths, labels, X_val, batch_size=config['batch_size'],
                                           shuffle=True)
    test_generator = PairCSVDataGenerator(test_indices, paths, labels, X_test, batch_size=config['test_batch_size'],
                                          shuffle=False)

    return train_generator, valid_generator, test_generator


train_generator, valid_generator, test_generator = create_data_generators()


input_shape_3D = (config['dim']) + (3 if config['weights'] else 1,)
input_shape_csv = config['csv_dim']
model = get_combined_model(input_shape_3D, input_shape_csv)

model.load_weights('weights/model_combined.h5')
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=config['learning_rate']),
    metrics=['acc'],
)


plot_eval_gen('weights/model_combined.h5', test_generator, model)
plot_history('../history/model_combined.pkl')

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=config['epochs'],
    verbose=1,
    callbacks=[
        ModelCheckpoint('weights/model_combined.h5', save_best_only=True, monitor='val_loss', mode='min'),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            patience=7,
            factor=0.5,
            min_lr=config['min_learning_rate'],
        ),
        EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True),
    ],
    workers=8
)

with open('../history/model_combined.pkl', 'wb') as file:
    pickle.dump(history.history, file)
