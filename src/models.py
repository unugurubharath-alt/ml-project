"""
Model Architectures for GDI Prediction Replication
Stanford CS 229: Automated Identification of Gait Abnormalities
Models:
1. ZeroRuleMeanGuesser (Baseline)
2. LinearRegressionBaseline (Baseline)
3. VGG16TransferModel (Spatial Transfer Learning)
4. ANet2DCNN (Custom Spatial 2D CNN)
5. CNN1DConvModel (CNN + 1D-CNN Temporal)
6. GDINet (CNN + LSTM Best Performer)
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

class ZeroRuleMeanGuesser(BaseEstimator, RegressorMixin):
    """Zero-Rule Baseline: Always predicts the training mean GDI."""
    def __init__(self):
        self.mean_val_ = None
        
    def fit(self, X, y):
        self.mean_val_ = float(np.mean(y))
        return self
        
    def predict(self, X):
        return np.full((len(X), 1), self.mean_val_)

class LinearRegressionBaseline(BaseEstimator, RegressorMixin):
    """Linear Regression Baseline on flattened or pooled spatial features."""
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)

def build_anet_frame_model(input_shape=(480, 640, 3), dropout_rate=0.5, l2_reg=0.000316):
    """
    Constructs the custom 2D CNN spatial frame extractor ('ANet')
    described in Gotlin et al. (CS229 Report 31).
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    frame_input = layers.Input(shape=input_shape, name="frame_input")
    
    # Block 1: 2 Conv layers (8 filters, 8x8 kernel)
    x = layers.Conv2D(8, (8, 8), padding='same', activation='relu', name='conv2d_67')(frame_input)
    x = layers.BatchNormalization(name='batch_normalization_67')(x)
    x = layers.Conv2D(8, (8, 8), padding='same', activation='relu', name='conv2d_68')(x)
    x = layers.BatchNormalization(name='batch_normalization_68')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_34')(x)
    x = layers.Dropout(dropout_rate, name='dropout_34')(x)
    
    # Block 2: 2 Conv layers with L2 regularization
    x = layers.Conv2D(8, (8, 8), padding='same', activation='relu', kernel_regularizer=reg, name='conv2d_69')(x)
    x = layers.BatchNormalization(name='batch_normalization_69')(x)
    x = layers.Conv2D(8, (8, 8), padding='same', activation='relu', kernel_regularizer=reg, name='conv2d_70')(x)
    x = layers.BatchNormalization(name='batch_normalization_70')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_35')(x)
    x = layers.Dropout(dropout_rate, name='dropout_35')(x)
    
    # Block 3: 2 Conv layers with L2 regularization
    x = layers.Conv2D(8, (8, 8), padding='same', activation='relu', kernel_regularizer=reg, name='conv2d_71')(x)
    x = layers.BatchNormalization(name='batch_normalization_71')(x)
    x = layers.Conv2D(8, (8, 8), padding='same', activation='relu', kernel_regularizer=reg, name='conv2d_72')(x)
    x = layers.BatchNormalization(name='batch_normalization_72')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), name='max_pooling2d_36')(x)
    x = layers.Dropout(dropout_rate, name='dropout_36')(x)
    
    # Dense Projection: Flatten to 16,960 -> Dense 16
    x = layers.Flatten(name='flatten_12')(x)
    frame_output = layers.Dense(16, activation='relu', name='dense_22')(x)
    
    return models.Model(frame_input, frame_output, name="ANet_Frame_Model")

def build_gdi_net(video_shape=(10, 480, 640, 3), lstm_units=256, dropout_rate=0.5):
    """
    Constructs GDI-Net: Best performing architecture (CNN + LSTM).
    Extracts spatial representations per frame via TimeDistributed(ANet)
    and captures temporal trajectories via LSTM(256).
    """
    frame_shape = video_shape[1:]
    frame_model = build_anet_frame_model(input_shape=frame_shape, dropout_rate=dropout_rate)
    
    video_input = layers.Input(shape=video_shape, name="video_input")
    seq_features = layers.TimeDistributed(frame_model, name="time_distributed_12")(video_input)
    lstm_out = layers.LSTM(lstm_units, name="lstm_12")(seq_features)
    gdi_prediction = layers.Dense(1, activation='linear', name="dense_23")(lstm_out)
    
    video_model = models.Model(video_input, gdi_prediction, name="GDI_Net_CNN_LSTM")
    return video_model, frame_model

def build_cnn_1dconv_model(video_shape=(10, 480, 640, 3)):
    """
    Constructs CNN + 1D-CNN temporal model described in paper Table 1.
    """
    frame_shape = video_shape[1:]
    frame_model = build_anet_frame_model(input_shape=frame_shape)
    
    video_input = layers.Input(shape=video_shape, name="video_input_1dconv")
    seq_features = layers.TimeDistributed(frame_model, name="time_distributed_1d")(video_input)
    
    # 1D Convolution over time dimension
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(seq_features)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    
    return models.Model(video_input, out, name="CNN_1D_Conv_Temporal")

def build_vgg16_spatial_model(input_shape=(224, 224, 3), pretrained=True):
    """
    Constructs VGG-16 spatial transfer learning model on cropped frames.
    """
    weights = 'imagenet' if pretrained else None
    base_vgg = keras.applications.VGG16(include_top=False, weights=weights, input_shape=input_shape)
    if pretrained:
        for layer in base_vgg.layers:
            layer.trainable = False
            
    x = layers.Flatten()(base_vgg.output)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    
    return models.Model(base_vgg.input, out, name="VGG16_Spatial")

def load_pretrained_gdi_net_weights(model, weights_path):
    """
    Loads pretrained weights saved by the authors into the rebuilt Keras 3 model.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
    with h5py.File(weights_path, 'r') as f:
        # Load frame model conv & dense layers
        td_group = f['time_distributed_12']
        frame_model = model.get_layer('time_distributed_12').layer
        
        for layer_name in ['conv2d_67', 'conv2d_68', 'conv2d_69', 'conv2d_70', 'conv2d_71', 'conv2d_72', 'dense_22']:
            l = frame_model.get_layer(layer_name)
            k = td_group[layer_name]['kernel:0'][:]
            b = td_group[layer_name]['bias:0'][:]
            l.set_weights([k, b])
            
        for bn_name in ['batch_normalization_67', 'batch_normalization_68', 'batch_normalization_69', 
                        'batch_normalization_70', 'batch_normalization_71', 'batch_normalization_72']:
            l = frame_model.get_layer(bn_name)
            gamma = td_group[bn_name]['gamma:0'][:]
            beta = td_group[bn_name]['beta:0'][:]
            mean = td_group[bn_name]['moving_mean:0'][:]
            var = td_group[bn_name]['moving_variance:0'][:]
            l.set_weights([gamma, beta, mean, var])
            
        # Load LSTM weights
        lstm_l = model.get_layer('lstm_12')
        lk = f['lstm_12']['lstm_12']['kernel:0'][:]
        lrk = f['lstm_12']['lstm_12']['recurrent_kernel:0'][:]
        lb = f['lstm_12']['lstm_12']['bias:0'][:]
        lstm_l.set_weights([lk, lrk, lb])
        
        # Load output Dense weights
        dense_l = model.get_layer('dense_23')
        dk = f['dense_23']['dense_23']['kernel:0'][:]
        db = f['dense_23']['dense_23']['bias:0'][:]
        dense_l.set_weights([dk, db])
        
    print(f"[SUCCESS] Loaded all authors' pretrained weights from {weights_path}")
    return model
