from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import get_3D_CNN, get_3D_CNN_and_Diff, get_csv_model, get_3D_CNN_and_CSV_model
from utils import config
from data import create_generators, create_data_arrays, create_other_generators

train_name = "3D CNN and CSV"

if train_name == "3D CNN":
    model = get_3D_CNN((config['dim']) + (3 if config['weights'] else 1,))
    training_generator, validation_generator, test_generator = create_generators(paired=False)
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config['learning_rate']),
        metrics=['acc'],
    )

    # Train the model, doing validation at the end of each epoch
    model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=config['epochs'],
        verbose=1,
        callbacks=[
            ModelCheckpoint('weights/model_3D_cnn_only.h5', save_best_only=True, monitor='val_loss', mode='min'),
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

elif train_name == "3D CNN and Diff":
    model = get_3D_CNN_and_Diff((config['dim']) + (3 if config['weights'] else 1,))
    training_generator, validation_generator, test_generator = create_generators(paired=True)
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config['learning_rate']),
        metrics=['acc'],
    )

    # Train the model, doing validation at the end of each epoch
    model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=config['epochs'],
        verbose=1,
        callbacks=[
            ModelCheckpoint('weights/model_3D_cnn_and_diff.h5', save_best_only=True, monitor='val_loss', mode='min'),
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

elif train_name == "CSV":
    model = get_csv_model(config['csv_dim'])
    train, valid, test = create_data_arrays()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config['learning_rate']),
        metrics=['acc'],
    )

    model.fit(
        train[0],
        train[1],
        validation_data=valid,
        epochs=config['epochs'],
        verbose=1,
        callbacks=[
            ModelCheckpoint('weights/csv_only.h5', save_best_only=True, monitor='val_loss', mode='min'),
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

elif train_name == "3D CNN and CSV":
    model = get_3D_CNN_and_CSV_model((config['dim']) + (3 if config['weights'] else 1,), config['csv_dim'])
    training_generator, validation_generator, test_generator = create_other_generators()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config['learning_rate']),
        metrics=['acc'],
    )

    model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=config['epochs'],
        verbose=1,
        callbacks=[
            ModelCheckpoint('weights/model_3D_cnn_and_csv.h5', save_best_only=True, monitor='val_loss', mode='min'),
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



