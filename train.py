from glob import glob
import numpy as np
from keras.models import load_model
from model import *
from utils import *
import json 
import sys
from sklearn.utils import class_weight
import keras

with open('config.json') as config_file:
    config = json.load(config_file)

def generate_data(start, end): 
    current = start
    x = []
    y = []
    pbar = tqdm(total = (end - start))
    while current < end:

        path = glob(config['root'] + '/*pat{}*'.format(current))[0]
        scans = load_scans(path)
        scans = norm_scans(scans)
        current_x = []
        labels = []
        for slice in scans:
            slice_label = slice[:,:,4]
            slice_x = slice[:,:,:4]
            current_x.append(slice_x)
            categorical = keras.utils.to_categorical(slice_label, 
                                                    num_classes = 5)
            labels.append(categorical)

        current_x = np.array(current_x)
        labels = np.array(labels)

        # transform data to one hot encoding
        '''
        pbar2 = tqdm(total = labels.shape[0])
        current_y = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2],5))
        for z in range(labels.shape[0]):
            pbar2.update(1)
            for i in range(labels.shape[1]):
                for j in range(labels.shape[2]):
                    current_y[z,i,j,int(labels[z,i,j])] = 1
        pbar2.close()
        '''
        x.extend(current_x)
        y.extend(labels)

        current += 1
        pbar.update(1)

    shuffle = list(zip(x, y))
    np.random.shuffle(shuffle)
    x, y = zip(*shuffle)
    x = np.array(x)
    y = np.array(y)
    pbar.close()
    return x, y
    

if __name__ == "__main__":
    
    model_name = sys.argv[1]                
    start_pat = int(sys.argv[2])
    end_pat = int(sys.argv[3])
    eps = int(sys.argv[4])
    bs = int(sys.argv[5])
    vs = float(sys.argv[6])

    model = load_model("models/{}.h5".format(model_name),
            custom_objects = {"dice_coef" : dice_coef,
                              "dice_coef_loss" : dice_coef_loss})  

    '''
    start_pat = int(input("Start patient: "))
    end_pat = int(input("End patient: "))
    eps = int(input('Epochs: '))
    bs = int(input('Batch size: '))
    vs = float(input('Validation split: '))
    '''


    x, y = generate_data(start_pat, end_pat)
    print("x shape:{}".format(x.shape))
    print("y shape:{}".format(y.shape))

    model.fit(x, y, epochs=eps, batch_size=bs, validation_split=vs, shuffle=True)
    model.save('models/{}.h5'.format(model_name))




