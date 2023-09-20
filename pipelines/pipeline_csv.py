import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import get_csv_model
from utils import config
from eval import plot_eval, plot_history
import pickle


def set_random_seeds(seed_value=27):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)


def preprocess_data(df):
    le = LabelEncoder()
    df["Group Encoded"] = le.fit_transform(df["Group"])

    df = df.drop(['Subject ID', 'MRI ID', 'Group', 'Visit'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    df.fillna(df.median(), inplace=True)

    X = df.drop('Group Encoded', axis=1)
    y = df['Group Encoded']
    y = to_categorical(y, num_classes=config['n_classes'])

    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def train_and_save_model(X_train, y_train, X_val, y_val):
    model = get_csv_model(config['csv_dim'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('weights/model_csv_only.h5', save_best_only=True, monitor='val_loss', mode='min'),
        ReduceLROnPlateau(monitor='val_loss', mode='min', patience=7, factor=0.5, min_lr=config['min_learning_rate']),
        EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config['epochs'],
                        verbose=1, callbacks=callbacks, workers=8)

    with open('../history/model_csv_only.pkl', 'wb') as file:
        pickle.dump(history.history, file)


if __name__ == '__main__':
    set_random_seeds()

    base_path = "../data/alzheimer/OASIS"
    base_csv_path = os.path.join(base_path, "oasis_longitudinal_demographics.xlsx")
    df = pd.read_excel(base_csv_path)[:config['n_samples']]

    X_temp, X_test, y_temp, y_test = preprocess_data(df)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    print("Number of samples in X_train:", X_train.shape[0])
    print("Number of samples in X_val:", X_val.shape[0])
    print("Number of samples in X_test:", X_test.shape[0])

    plot_eval('../weights/model_csv_only.h5', (X_test, y_test))
    plot_history('../history/model_csv_only.pkl')
    train_and_save_model(X_train, y_train, X_val, y_val)
