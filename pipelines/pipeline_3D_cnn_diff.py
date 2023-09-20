import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import config
from model import get_3D_CNN_and_Diff
from data import create_paired_image_generators
from eval import plot_eval_gen, plot_history

# Constants
MODEL_WEIGHTS_PATH = 'weights/model_3D_cnn_and_diff.h5'
HISTORY_PATH = '../history/model_3D_cnn_and_diff.pkl'

# Model initialization
model_dimensions = config['dim'] + (3 if config['weights'] else 1,)
model = get_3D_CNN_and_Diff(model_dimensions, 180, 215)

# Load data generators
training_generator, validation_generator, test_generator = create_paired_image_generators()

# Display evaluation plots
plot_eval_gen(MODEL_WEIGHTS_PATH, test_generator)
plot_history(HISTORY_PATH)

# Model compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=config['learning_rate']),
    metrics=['acc'],
)

# Callbacks for training
callbacks = [
    ModelCheckpoint(MODEL_WEIGHTS_PATH, save_best_only=True, monitor='val_loss', mode='min'),
    ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        patience=7,
        factor=0.5,
        min_lr=config['min_learning_rate'],
    ),
    EarlyStopping(monitor='loss', mode='min', patience=30, restore_best_weights=True),
]

# Model training
history = model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=config['epochs'],
    verbose=1,
    callbacks=callbacks,
    workers=8
)

# Save training history
with open(HISTORY_PATH, 'wb') as file:
    pickle.dump(history.history, file)
