import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

PLANES = 31
MOVES = 361

def mix_depthwise_conv(x, expansion_filters):
    half = expansion_filters // 2
    x1 = x[:, :, :, :half]
    x2 = x[:, :, :, half:]
    
    d1 = layers.DepthwiseConv2D(3, padding='same', use_bias=False, 
                                depthwise_regularizer=regularizers.l2(1e-4))(x1)
    d2 = layers.DepthwiseConv2D(5, padding='same', use_bias=False, 
                                depthwise_regularizer=regularizers.l2(1e-4))(x2)
    return layers.Concatenate(axis=-1)([d1, d2])

def inverted_residual_mix_swish_block(inputs, filters, expansion, stride=1):
    x = layers.Conv2D(expansion, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x) 
    
    x = mix_depthwise_conv(x, expansion)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    if inputs.shape[-1] == filters and stride == 1:
        x = layers.Add()([inputs, x])
    return x

def se_block(x, filters, reduction=8):
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // reduction, activation='swish', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.Multiply()([x, se])



def get_student_model():
    inputs = keras.Input(shape=(19, 19, PLANES), name='board')
    x = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    for i in range(13): 
        shortcut = x
        x = mix_depthwise_conv(x, expansion_filters=64) 
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(64, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        if i > 9: x = layers.SpatialDropout2D(0.05)(x)
        x = layers.Add()([x, shortcut])
        if i > 9: x = se_block(x, 64)
        x = layers.Activation('swish')(x)

    p = layers.Conv2D(1, 1, padding='same', use_bias=True, kernel_regularizer=regularizers.l2(1e-4))(x)
    p = layers.Flatten()(p)
    policy = layers.Activation('linear', name='policy', dtype='float32')(p)

    v = layers.GlobalAveragePooling2D()(x)
    v = layers.Dense(48, kernel_regularizer=regularizers.l2(1e-4))(v)
    v = layers.Activation('swish')(v)
    v = layers.Dense(16, kernel_regularizer=regularizers.l2(1e-4))(v)
    v = layers.Activation('swish')(v)
    value = layers.Dense(1, activation='sigmoid', name='value', dtype='float32')(v)

    return keras.Model(inputs=inputs, outputs=[policy, value], name="MixConv_Student_Distilled")