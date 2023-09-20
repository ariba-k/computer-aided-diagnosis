from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate, GlobalAveragePooling3D, Lambda
from tensorflow.keras.models import Model
from utils import config
from classification_models_3D.tfkeras import Classifiers
from tensorflow.keras import backend as K


def get_3D_CNN_base(input_shape):
    modelPoint, _ = Classifiers.get('resnet18')
    net = modelPoint(input_shape=input_shape, include_top=False, weights='imagenet')
    x = net.layers[-1].output
    x = GlobalAveragePooling3D()(x)
    return Model(net.inputs, x, name='3D-CNN-base')


def get_3D_CNN(input_shape):
    """Build a 3D convolutional neural network model."""
    base_model = get_3D_CNN_base(input_shape)

    x = base_model.output
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(config['n_classes'], activation='softmax')(x)

    return Model(base_model.inputs, outputs, name='3D-CNN')


def get_3D_CNN_and_Diff(input_shape, n_units_1=128, n_units_2=64, absolute=False):
    cnn_base = get_3D_CNN_base(input_shape)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    features1 = cnn_base(input1)
    features2 = cnn_base(input2)

    diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]) if absolute else tensors[0] - tensors[1])([features1, features2])
    x = Dense(n_units_1, activation='relu')(diff)
    x = Dense(n_units_2, activation='relu')(x)
    outputs = Dense(config['n_classes'], activation='softmax')(x)

    return Model([input1, input2], outputs, name='3D-CNN-Diff')


def get_csv_model(input_shape):
    return Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(config['n_classes'], activation='softmax')
    ])


def get_3D_CNN_and_CSV(input_shape_cnn, input_shape_csv):
    cnn_base = get_3D_CNN_base(input_shape_cnn)

    csv_model = get_csv_model(input_shape_csv)
    csv_model.pop()

    input_cnn = Input(shape=input_shape_cnn)
    input_csv = Input(shape=input_shape_csv)

    features_cnn = cnn_base(input_cnn)
    features_csv = csv_model(input_csv)

    merged_features = Concatenate()([features_cnn, features_csv])
    x = Dense(128, activation='relu')(merged_features)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(config['n_classes'], activation='softmax')(x)

    return Model(inputs=[input_cnn, input_csv], outputs=outputs, name='3D-CNN-CSV')


def get_combined_model(input_shape_3D, input_shape_csv):
    input1_3D = Input(shape=input_shape_3D)
    input2_3D = Input(shape=input_shape_3D)
    input_csv = Input(shape=input_shape_csv)

    cnn_diff_model = get_3D_CNN_and_Diff(input_shape_3D)
    csv_model = get_csv_model(input_shape_csv)

    cnn_diff_output = cnn_diff_model([input1_3D, input2_3D])
    csv_output = csv_model(input_csv)

    combined_output = Concatenate()([cnn_diff_output, csv_output])
    x = Dense(128, activation='relu')(combined_output)
    x = Dense(64, activation='relu')(x)
    final_output = Dense(config['n_classes'], activation='softmax')(x)

    return Model(inputs=[input1_3D, input2_3D, input_csv], outputs=final_output)
