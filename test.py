from keras.models import load_model
import matplotlib.pyplot as plt
import imageio
from skimage import io
import skimage

import json
from utils import *
from model import *
from metrics import *
from tqdm import tqdm


with open('config.json') as config_file:
    config = json.load(config_file)

if __name__ == "__main__":
    model_name = input("Model name: ") 
    model = load_model("models/{}.h5".format(model_name),
            custom_objects = {"dice_coef_loss" : dice_coef_loss,
                              "dice_coef" : dice_coef})

    dw = 0
    de = 0
    dc = 0

    patient = 200
    while patient < 216:

        path = glob(config['root'] + "/*pat{}*".format(patient))[0]
        scans = load_scans(path)
        scans = norm_scans(scans)

        gt = []
        pred = []

        for slice_no in range(scans.shape[0]):
            test_slice = scans[slice_no:slice_no+1,:,:,:4]
            test_label = scans[slice_no:slice_no+1,:,:,4]
            prediction = model.predict(test_slice, batch_size=32)
            prediction = prediction[0]
            prediction = np.around(prediction)
            prediction = np.argmax(prediction, axis=-1)
            gt.extend(test_label[0])
            pred.extend(prediction)

            '''
            scan = test_slice[0,:,:,2]
            tmp_label = test_label[0]

            fig = plt.figure(figsize=(15, 10))
            plt.subplot(131)
            plt.title('Input')
            plt.imshow(scan, cmap='gray')
            plt.subplot(132)
            plt.title('Ground Truth')
            plt.imshow(tmp_label,cmap='gray')
            plt.subplot(133)
            plt.title('Prediction')
            plt.imshow(prediction,cmap='gray')
            plt.savefig('outputs/{}_pat{}_slice{}.png'.format(model_name,patient,slice_no), bbox_inches='tight')
            plt.close(fig)
            '''



        gt = np.array(gt)
        pred = np.array(pred)

        dice_whole = DSC_whole(pred, gt)
        dice_en = DSC_en(pred, gt)
        dice_core = DSC_core(pred, gt)

        sen_whole = sensitivity_whole(pred, gt)
        sen_en = sensitivity_en(pred, gt)
        sen_core = sensitivity_core(pred, gt)

        spec_whole = specificity_whole(pred, gt)
        spec_en = specificity_en(pred, gt)
        spec_core = specificity_core(pred, gt)
     
        print("=======================================")
        print("Patient {}".format(patient))
        print("---------------------------------------")
        print("Dice whole tumor score: {:0.4f}".format(dice_whole)) 
        print("Dice enhancing tumor score: {:0.4f}".format(dice_en)) 
        print("Dice core tumor score: {:0.4f}".format(dice_core)) 
        print("---------------------------------------")
        print("Sensitivity whole tumor score: {:0.4f}".format(sen_whole)) 
        print("Sensitivity enhancing tumor score: {:0.4f}".format(sen_en)) 
        print("Sensitivity core tumor score: {:0.4f}".format(sen_core)) 
        print("---------------------------------------")
        print("Specificity whole tumor score: {:0.4f}".format(spec_whole)) 
        print("Specificity enhancing tumor score: {:0.4f}".format(spec_en)) 
        print("Specificity core tumor score: {:0.4f}".format(spec_core)) 
        print("=======================================")

        dw += dice_whole
        de += dice_en
        dc += dice_core

        patient += 1
        
    dw_total = dw / 15
    de_total = de / 15
    dc_total = dc / 15
    print("=======================================")
    print("Summary (Patients 200 - 215)")
    print("---------------------------------------")
    print("Dice whole tumor score: {:0.4f}".format(dw_total)) 
    print("Dice enhancing tumor score: {:0.4f}".format(de_total)) 
    print("Dice core tumor score: {:0.4f}".format(dc_total)) 
    print("=======================================")






       
    

