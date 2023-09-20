import os
import random
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from scipy.ndimage import zoom
from utils import config
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

base_path = "data/alzheimer/OASIS"
base_image_path = os.path.join(base_path, "OAS2_RAW_PART1")
base_csv_path = os.path.join(base_path, "oasis_longitudinal_demographics.xlsx")

seed = 27
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume.astype('float32')


def resize_volume(img, desired_width, desired_height, desired_depth):
    """Resize the volume"""
    # Compute zoom factors
    width_factor = desired_width / img.shape[0]
    height_factor = desired_height / img.shape[1]
    depth_factor = desired_depth / img.shape[2]
    # print(f'Zoom factors: {width_factor}, {height_factor}, {depth_factor}')
    # Rotate volume by 90 degrees
    img = ndimage.rotate(img, 90, axes=(0, 1), reshape=False)
    # Resize the volume using spline interpolated zoom (SIZ)
    img = zoom(img, (width_factor, height_factor, depth_factor, 1), order=1)  # Add 1 to the zoom factors

    return img


def random_rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # print(f'Input volume shape: {volume.shape}')
    # Resize width, height and depth
    volume = resize_volume(
        volume, config['dim'][0], config['dim'][1], config['dim'][2]
    )
    # print(f'Resized volume shape: {volume.shape}')
    return volume


def get_data(path):
    df = pd.read_excel(path)[:config['n_samples']]
    valid_mri_ids = []

    for patient_id in df["Subject ID"].unique():
        # Get all MRI scans for the current patient
        patient_scans = sorted([mri_id for mri_id in df[df["Subject ID"] == patient_id]["MRI ID"]])
        # Add the first two scans to the list
        valid_mri_ids += patient_scans[:2]

    paths = [os.path.join(*[base_image_path, mri_id, "RAW", "mpr-1.nifti.img"]) for mri_id in
             valid_mri_ids]

    le = LabelEncoder()
    df["Group Encoded"] = le.fit_transform(df["Group"])
    label_to_encoding = {label: encoding for label, encoding in zip(le.classes_, range(len(le.classes_)))}

    print(label_to_encoding)

    labels = df[df["MRI ID"].isin(valid_mri_ids)]["Group Encoded"]

    labels = labels.reset_index(drop=True)

    return paths, labels, df


paths, labels, df = get_data(base_csv_path)


def create_paired_indices(paths, labels):
    # Create a list of pairs of MRI scans (MR1, MR2) for each subject
    pair_paths = [(paths[idx], paths[idx + 1]) for idx in range(0, len(paths), 2)]
    pair_labels = [labels[idx] for idx in range(0, len(labels), 2)]

    # Modify the stratification process to use these pairs instead of single MRI scans
    train_indices, test_indices = next(
        StratifiedShuffleSplit(1, train_size=0.8, random_state=seed).split(
            pair_paths, pair_labels
        )
    )

    tmp_labels = [pair_labels[idx] for idx in train_indices]
    tmp_paths = [pair_paths[idx] for idx in train_indices]

    tmp_train_indices, tmp_val_indices = next(
        StratifiedShuffleSplit(1, train_size=0.8, random_state=seed).split(
            tmp_paths, tmp_labels
        )
    )

    train_pairs = [pair_paths[index] for index in train_indices]
    val_pairs = [pair_paths[index] for index in tmp_val_indices]
    test_pairs = [pair_paths[index] for index in test_indices]

    # Convert pairs back to individual MRI scans
    train_indices = [paths.index(path) for pair in train_pairs for path in pair]
    val_indices = [paths.index(path) for pair in val_pairs for path in pair]
    test_indices = [paths.index(path) for pair in test_pairs for path in pair]

    return train_indices, val_indices, test_indices


def create_indices(paths, labels):
    train_indices, test_indices = next(
        StratifiedShuffleSplit(1, train_size=0.8, random_state=seed).split(
            paths, labels
        )
    )

    tmp_labels = [labels[idx] for idx in train_indices]
    tmp_paths = [paths[idx] for idx in train_indices]

    tmp_train_indices, tmp_val_indices = next(
        StratifiedShuffleSplit(1, train_size=0.8, random_state=seed).split(
            tmp_paths, tmp_labels
        )
    )

    tmp_train_paths = [tmp_paths[idx] for idx in tmp_train_indices]
    tmp_val_paths = [tmp_paths[idx] for idx in tmp_val_indices]

    train_indices = [paths.index(path) for path in tmp_train_paths]
    val_indices = [paths.index(path) for path in tmp_val_paths]

    return train_indices, val_indices, test_indices


def create_scaled_csv_data():
    features = df.copy()
    features = features.drop(['Subject ID', 'MRI ID', 'Group', 'Visit'], axis=1)
    features = pd.get_dummies(features, drop_first=True)
    features.fillna(features.median(), inplace=True)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return np.array(scaled_features)


class ImageGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, indices, paths, labels, batch_size=config['batch_size'], dim=config['dim'],
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
        X, y = self.__data_generation(indices_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indices_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        if config['weights']:
            X = np.empty((self.batch_size, *self.dim, 3))
        else:
            X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(indices_temp):
            # Store sample
            volume = process_scan(self.paths[ID])
            if self.transform is not None:
                volume = self.transform(volume)
            if config['weights']:
                X[i,] = np.repeat(volume[..., np.newaxis], 3, axis=-1) if volume.ndim == 3 else volume
            else:
                X[i,] = volume[..., np.newaxis] if volume.ndim == 3 else volume

            # Store class
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)


class PairImageGenerator(tf.keras.utils.Sequence):
    """Generates pairs of MRI scans (MR1, MR2) for the same subject."""

    def __init__(self, indices, paths, labels, batch_size=config['batch_size'], dim=config['dim'],
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

        return [X1, X2], y

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
        """Generates data containing batch_size pairs of MRI scans"""
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

            # Store class
            y[i] = self.labels[ID]

        return X1, X2, to_categorical(y, num_classes=self.n_classes)


class CSVImageGenerator(ImageGenerator):
    def __init__(self, csv_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_data = csv_data

    def __data_generation(self, indices_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        if config['weights']:
            X_3D = np.empty((self.batch_size, *self.dim, 3))
        else:
            X_3D = np.empty((self.batch_size, *self.dim, 1))
        X_csv = np.empty((self.batch_size, self.csv_data.shape[1]))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(indices_temp):
            # Store 3D sample
            volume = process_scan(self.paths[ID])
            if self.transform is not None:
                volume = self.transform(volume)
            if config['weights']:
                X_3D[i,] = np.repeat(volume[..., np.newaxis], 3, axis=-1) if volume.ndim == 3 else volume
            else:
                X_3D[i,] = volume[..., np.newaxis] if volume.ndim == 3 else volume

            # Store CSV data
            X_csv[i,] = self.csv_data[ID]

            # Store class
            y[i] = self.labels[ID]

        return (X_3D, X_csv), to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        indices_temp = [self.indices[k] for k in indexes]

        # Generate data
        (X_3D, X_csv), y = self.__data_generation(indices_temp)

        return [X_3D, X_csv], y


def create_image_generators():
    train_indices, val_indices, test_indices = create_indices(paths, labels)

    print(
        'Number of samples:\n'
        f'train: {len(train_indices)}\n'
        f'validation: {len(val_indices)}\n'
        f'test: {len(test_indices)}'
    )

    train_generator = ImageGenerator(
        train_indices,
        paths,
        labels,
        batch_size=config['batch_size'],
        shuffle=True,
        transform=random_rotate,
    )

    valid_generator = ImageGenerator(
        val_indices,
        paths,
        labels,
        batch_size=config['batch_size'],
        shuffle=True,
    )

    test_generator = ImageGenerator(
        test_indices,
        paths,
        labels,
        batch_size=config['test_batch_size'],
        shuffle=False,
    )

    return train_generator, valid_generator, test_generator


def create_paired_image_generators():
    train_indices, val_indices, test_indices = create_paired_indices(paths, labels)

    print(
        'Number of samples:\n'
        f'train: {len(train_indices)}\n'
        f'validation: {len(val_indices)}\n'
        f'test: {len(test_indices)}'
    )

    train_generator = PairImageGenerator(
        train_indices,
        paths,
        labels,
        batch_size=config['batch_size'],
        shuffle=True,
        transform=random_rotate,
    )

    valid_generator = PairImageGenerator(
        val_indices,
        paths,
        labels,
        batch_size=config['batch_size'],
        shuffle=True,
    )

    test_generator = PairImageGenerator(
        test_indices,
        paths,
        labels,
        batch_size=config['test_batch_size'],
        shuffle=False,
    )

    return train_generator, valid_generator, test_generator


def create_csv_image_generators():
    scaled_csv_data = create_scaled_csv_data()
    train_indices, val_indices, test_indices = create_indices(paths, labels)

    train_gen = CSVImageGenerator(
        indices=train_indices,
        paths=paths,
        labels=labels,
        csv_data=scaled_csv_data,
        batch_size=config['batch_size'],
        dim=config['dim'],
        n_classes=config['n_classes'],
        shuffle=True,
        transform=None
    )

    val_gen = CSVImageGenerator(
        indices=val_indices,
        paths=paths,
        labels=labels,
        csv_data=scaled_csv_data,
        batch_size=config['batch_size'],
        dim=config['dim'],
        n_classes=config['n_classes'],
        shuffle=True,
    )

    test_gen = CSVImageGenerator(
        indices=test_indices,
        paths=paths,
        labels=labels,
        csv_data=scaled_csv_data,
        batch_size=config['batch_size'],
        dim=config['dim'],
        n_classes=config['n_classes'],
        shuffle=False,

    )

    return train_gen, val_gen, test_gen


def create_csv_data_split():
    features = create_scaled_csv_data()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
