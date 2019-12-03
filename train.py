from glob import glob
import numpy as np
from keras.models import load_model
from model import *
from utils import *
import json 
import sys
from sklearn.utils import class_weight

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
            # exclude slices that are more than 75% background
            if len(np.argwhere(slice_x == 0)) > (240*240*2):
                continue
            current_x.append(slice_x)
            labels.append(slice_label)

        current_x = np.array(current_x)
        labels = np.array(labels)

        # transform data to one hot encoding
        '''
        current_y = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2],5))
        for z in range(labels.shape[0]):
            for i in range(labels.shape[1]):
                for j in range(labels.shape[2]):
                    current_y[z,i,j,int(labels[z,i,j])] = 1
        '''
        
        x.extend(current_x)
#        y.extend(current_y)

        y.extend(labels)
        current += 1
        pbar.update(1)

    shuffle = list(zip(x, y))
    np.random.shuffle(shuffle)
    x, y = zip(*shuffle)
    x = np.array(x)
    y = np.array(y)
    tmp_y = y.reshape(y.shape[0]*y.shape[1]*y.shape[2])
    class_weights = class_weight.compute_class_weight('balanced',
                                             np.unique(tmp_y),tmp_y)

    one_hot_y = np.zeros((y.shape[0],y.shape[1],y.shape[2],5))
    for z in range(y.shape[0]):
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                one_hot_y[z,i,j,int(y[z,i,j])] = 1
    pbar.close()
    return x, one_hot_y, class_weights
    

if __name__ == "__main__":
    model_name = input("Model name: ")
    model = load_model("models/{}.h5".format(model_name),
            custom_objects = {"dice_coef" : dice_coef})  

    start_pat = int(input("Start patient: "))
    end_pat = int(input("End patient: "))
    eps = int(input('Epochs: '))
    bs = int(input('Batch size: '))
    vs = float(input('Validation split: '))


    x, y, class_weights = generate_data(start_pat, end_pat)
    model.fit(x, y, epochs=eps, batch_size=bs, validation_split=0.25, shuffle=True, class_weight=class_weights)
    model.save('models/{}.h5'.format(model_name))




