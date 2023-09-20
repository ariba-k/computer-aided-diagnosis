import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import get_3D_CNN
from utils import config
from data import create_image_generators
from eval import plot_eval_gen, plot_history

# Constants
MODEL_WEIGHTS_PATH = 'weights/model_3D_cnn_only.h5'
HISTORY_PATH = '../history/model_3D_cnn_only.pkl'

# Load model and data generators
model = get_3D_CNN((config['dim']) + (3 if config['weights'] else 1,))
training_generator, validation_generator, test_generator = create_image_generators()

# Display evaluation plots
plot_eval_gen(MODEL_WEIGHTS_PATH, test_generator)
plot_history(HISTORY_PATH)

# Model compilation
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=config['learning_rate']),
    metrics=['acc'],
)

# Model training
callbacks = [
    ModelCheckpoint(MODEL_WEIGHTS_PATH, save_best_only=True, monitor='val_loss', mode='min'),
    ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        patience=7,
        factor=0.5,
        min_lr=config['min_learning_rate'],
    ),
    EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True),
]

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
