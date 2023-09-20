config = {
    'learning_rate': 0.009,      # A slightly higher learning rate can speed up training
    'min_learning_rate': 1e-8,  # Keep this value the same
    'epochs': 50,                # Increase the number of epochs for better convergence
    'batch_size': 4,            # Decrease the batch size to fit 3D data into GPU memory
    'test_batch_size': 1,       # Keep this value the same
    'n_classes': 3,             # Number of diseases to predict (keep this the same)
    'n_samples': 100,           # Keep this value the same
    'dim': (128, 128, 64),      # Increase spatial resolution and adjust depth for 3D MRI
    'csv_dim': (10,),           # Keep this value the same
    'weights': 'imagenet'       # Pretrained weights can help, but consider training from scratch if the model doesn't converge
}
