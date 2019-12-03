from keras.models import load_model
import json
from utils import *
from model import *
from tqdm import tqdm


with open('config.json') as config_file:
    config = json.load(config_file)

if __name__ == "__main__":
    model_name = input("Model name: ") 
    model = load_model("models/{}.h5".format(model_name),
            custom_objects = {"dice_coef_loss" : dice_coef_loss,
                              "dice_coef" : dice_coef})
    patient = int(input("Patient no: "))
    path = glob(config['root'] + "/*pat{}*".format(patient))[0]
    scans = load_scans(path)
    scans = norm_scans(scans)

    gt = []
    pred = []

    pbar = tqdm(total = scans.shape[0])
    for slice in scans:
        test_slice = slice[:,:,:4]
        test_label = slice[:,:,4]
        prediction = model.predict(test_slice, batch_size=32)
        prediction = np.around(prediction)
        prediction = np.argmax(prediction, axis=-1)
        gt.extend(truth)
        pred.extend(prediction)
        pbar.update(1)

    pbar.close()

    
    

