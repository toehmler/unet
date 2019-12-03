from glob import glob
import numpy as np
from keras.models import load_model
from model import *
from utils import *
import json 
import sys

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
        current_y = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2],5))
        for z in range(labels.shape[0]):
            for i in range(labels.shape[1]):
                for j in range(labels.shape[2]):
                    current_y[z,i,j,int(labels[z,i,j])] = 1
        
        x.extend(current_x)
        y.extend(current_y)
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
    model_name = input("Model name: ")
    start_pat = int(input("Start patient: "))
    end_pat = int(input("End patient: "))
    eps = int(input('Epochs: '))
    bs = int(input('Batch size: '))
    vs = float(input('Validation split: '))

    model = load_model("models/{}.h5".format(model_name),
            custom_objects = {"dice_coef_loss" : dice_coef_loss,
                              "dice_coef" : dice_coef})
    print(model.summary())

    x, y = generate_data(start_pat, end_pat)
    model.fit(x, y, epochs=eps, batch_size=bs, validation_split=0.25, shuffle=True)
    model.save('models/{}.h5'.format(model_name))




