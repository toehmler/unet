from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import initializers
import numpy as np
import keras.backend as K
import tensorflow as tf

def weighted_log_loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # weights are assigned in this order : normal,necrotic,edema,enhancing 
    weights=np.array([1,5,2,4])
    weights = K.variable(weights)
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

def gen_dice_loss(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''

    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights

    return GDL

def dice_coef(y_true, y_pred, smooth=1.0):
    ''' Dice Coefficient

    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    '''

    class_num = 2
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss


def dice_coef_loss(y_true, y_pred):
    ''' Dice Coefficient Loss

    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    '''
    return 1-dice_coef(y_true, y_pred)



def unet(input_size = (240,240,4)):
    ''' U-Net

    '''

    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv5),conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv6),conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv7),conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer=initializers.random_normal(stddev=0.01))(conv8),conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializers.random_normal(stddev=0.01))(conv9)

    conv10 = Conv2D(5, (1, 1), activation='relu',
                    kernel_initializer=initializers.random_normal(stddev=0.01))(conv9)
    conv10 = Activation('softmax')(conv10)
    model = Model(inputs=[inputs], outputs=[conv10])

    lr = 1e-4
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=[dice_coef])

    return model

if __name__ == "__main__":
    model_name = input('Model Name: ')
    unet_model = unet()
    unet_model.save('models/{}.h5'.format(model_name))
