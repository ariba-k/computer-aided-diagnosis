import optuna
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import get_3D_CNN_and_Diff
from utils import config
from data import create_paired_image_generators

import optuna.visualization
import plotly.io as pio

# Create paired image generators
training_generator, validation_generator, test_generator = create_paired_image_generators()

def objective(trial):
    # Define hyperparameters range
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    n_units_1 = trial.suggest_int('n_units_1', 64, 256)
    n_units_2 = trial.suggest_int('n_units_2', 64, 256)
    patience = trial.suggest_int('patience', 5, 30)

    # Get and compile the model
    model_dim = (config['dim']) + (3 if config['weights'] else 1,)
    model = get_3D_CNN_and_Diff(model_dim, n_units_1, n_units_2)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['acc'])

    # Set up callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', mode='min', patience=patience, factor=0.5, min_lr=config['min_learning_rate']),
        EarlyStopping(monitor='loss', mode='min', patience=patience, restore_best_weights=True),
        optuna.integration.KerasPruningCallback(trial, 'val_loss'),
    ]

    # Train and evaluate the model
    model.fit(training_generator, validation_data=validation_generator, epochs=config['epochs'], verbose=1, callbacks=callbacks, workers=8)
    score = model.evaluate(validation_generator, verbose=0)
    return score[1]  # Return validation accuracy

# Optuna study setup and optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Generate and save plots
plots = [
    ("Optimization History", optuna.visualization.plot_optimization_history(study), 'optimization_history.png'),
    ("Parallel Coordinate Plot", optuna.visualization.plot_parallel_coordinate(study), 'parallel_coordinate.png'),
    ("Slice Plot", optuna.visualization.plot_slice(study), 'slice.png'),
    ("Parameter Importances", optuna.visualization.plot_param_importances(study), 'param_importances.png'),
    ("Contour Plot", optuna.visualization.plot_contour(study, params=['n_units_1', 'n_units_2']), 'contour.png')
]

for title, fig, filename in plots:
    fig.update_layout(title=title)
    pio.write_image(fig, f'results/hypertune/{filename}')

# Print best trial details
best_trial = study.best_trial
print("Best trial:")
print(f"Value: {best_trial.value}")
print("Params: ")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")
