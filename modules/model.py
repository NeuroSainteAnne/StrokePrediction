from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D, Flatten
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow import random_normal_initializer, tile

def conv_block(input_tensor, filters, size, strides=1, batchnorm=True, name="null"):
    'Base convolutional block'
    initializer = random_normal_initializer(0., 0.02)
    x = Conv2D(filters, size, strides=strides, padding='same',
               kernel_initializer=initializer, 
              name=name)(input_tensor)
    if batchnorm:
        x = BatchNormalization(name=name+"_BatchNorm")(x)
    x = Activation("relu", name=name+"_Activation")(x)
    return x

def UNet2D(n_filters=16, dropout=0.5, n_input_sequences=3, n_input_slices=3):
    # Inputs
    input_img = Input((256,256,n_input_slices,n_input_sequences), name='G_InputSequences')
    input_reshaped = Reshape((256,256,n_input_slices*n_input_sequences), 
                             name="G_InputSequencesShaped")(input_img)
        
    # contracting path    
    c0 = conv_block(input_reshaped, n_filters*1, 3, name="Down_1a")
    c1 = conv_block(c0, n_filters*1, 3, 2, name="Down_1b")
    p1 = Dropout(dropout, name="Down_Drop1")(c1)

    c2 = conv_block(p1, n_filters*2, 3, name="Down_2a")
    c2 = conv_block(c2, n_filters*2, 3, 2, name="Down_2b")
    p2 = Dropout(dropout, name="Down_Drop2")(c2)

    c3 = conv_block(p2, n_filters*4, 3, name="Down_3a")
    c3 = conv_block(c3, n_filters*4, 3, 2, name="Down_3b")
    p3 = Dropout(dropout, name="Down_Drop3")(c3)

    c4 = conv_block(p3, n_filters*8, 3, name="G_Down_4a")
    c4 = conv_block(c4, n_filters*8, 3, 2, name="G_Down_4b")
    p4 = Dropout(dropout, name="G_Down_Drop4")(c4)
    
    c5 = conv_block(p4, n_filters*16, 3, name="G_Down_5a")
    c5 = conv_block(c5, n_filters*16, 3, 2, name="G_Down_5b")
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, 3, 2, padding='same', name="Up_1a")(c5)
    u6 = concatenate([u6, c4], name="Up_Skip1")
    u6 = Dropout(dropout, name="Up_Drop1")(u6)
    u6 = conv_block(u6, n_filters*8, 3, name="Up_1b")
    u6 = conv_block(u6, n_filters*8, 3, name="Up_1c")

    u7 = Conv2DTranspose(n_filters*4, 3, 2, padding='same', name="Up_2a")(u6)
    u7 = concatenate([u7, c3], name="Up_Skip2")
    u7 = Dropout(dropout, name="Up_Drop2")(u7)
    u7 = conv_block(u7, n_filters*4, 3, name="Up_2b")
    u7 = conv_block(u7, n_filters*4, 3, name="Up_2c")

    u8 = Conv2DTranspose(n_filters*2, 3, 2, padding='same', name="Up_3a")(u7)
    u8 = concatenate([u8, c2], name="Up_Skip3")
    u8 = Dropout(dropout, name="Up_Drop3")(u8)
    u8 = conv_block(u8, n_filters*2, 3, name="Up_3b")
    u8 = conv_block(u8, n_filters*2, 3, name="Up_3c")

    u9 = Conv2DTranspose(n_filters*1, 3, 2, padding='same', name="Up_4a")(u8)
    u9 = concatenate([u9, c1], axis=3, name="Up_Skip4")
    u9 = Dropout(dropout, name="Up_Drop4")(u9)
    u9 = conv_block(u9, n_filters*1, 3, name="Up_4b")
    u9 = conv_block(u9, n_filters*1, 3, name="Up_4c")

    u10 = Conv2DTranspose(n_filters*1, 3, 2, padding='same', name="Up_5a") (u9)
    u10 = concatenate([u10, c0], axis=3, name="Up_Skip5")
    u10 = Dropout(dropout, name="Up_Drop5")(u10)
    u10 = conv_block(u10, n_filters*1, 3, name="Up_5b")
    u10 = conv_block(u10, n_filters*1, 3, name="Up_5c")
    
    output = Conv2D(1, 1, activation='tanh', dtype='float32', name="End") (u10)
    model = Model(inputs=[input_img, input_quality], outputs=[output], name="UNet")
    return model
