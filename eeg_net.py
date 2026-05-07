from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.constraints import max_norm

def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, F1=4, D=2, F2=8, value = False):
    """
    EEGNet-4,2 Model as described in Lawhern et al. (2018)
    """
    input_main = Input((Chans, Samples, 1))
    
    # ================================
    # Block 1: Temporal and Spatial Filtering
    # ================================
    # Temporal Convolution: Learns frequency filters [cite: 152]
    block1 = Conv2D(F1, (1, 64), padding='same', use_bias=False)(input_main)
    block1 = BatchNormalization(axis=-1)(block1)
    
    # Depthwise Convolution: Learns spatial filters [cite: 154]
    # constraint: max_norm(1.) [cite: 180]
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, 
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    # ================================
    # Block 2: Separable Convolution
    # ================================
    # Separable Convolution: Summarizes individual feature maps, then mixes them [cite: 183]
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    
    # ================================
    # Classification Block
    # ================================
    flatten = Flatten()(block2)
    # constraint: max_norm(0.25) [cite: 197, 501]
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.25))(flatten)

    if value:
        return Model(inputs=input_main, outputs=dense)
    
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

