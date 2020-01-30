from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
from skimage import io
import skimage
import json
from utils import *
from model import *
from metrics import *
from tqdm import tqdm
import sys

with open('config.json') as config_file:
    config = json.load(config_file)


if __name__ == "__main__":
    model_name = input("Model name: ") 
    model = load_model("models/{}.h5".format(model_name),
            custom_objects = {"dice_coef_loss" : dice_coef_loss,
                              "dice_coef" : dice_coef})


    patient_no = input("Patient no: ")
    path = glob(config['root'] + "/*pat{}*".format(patient_no))[0]
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

    pbar = tqdm(total = scans.shape[0])

    ims = []
    fig = plt.figure()

    for slice_no in range(scans.shape[0]):
        test_slice = scans[slice_no:slice_no+1,:,:,:4]
        test_label = scans[slice_no:slice_no+1,:,:,4]
        prediction = model.predict(test_slice, batch_size=32)
        prediction = prediction[0]
        prediction = np.around(prediction)
        prediction = np.argmax(prediction, axis=-1)
        gt.extend(test_label[0])
        pred.extend(prediction)
        pbar.update(1)

        scan = test_slice[0,:,:,2]
        label = test_label[0]

        prediction_img = plt.imshow(prediction, cmap='gray', animated=True)
        ims.append([prediction_img])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save('test_animation.mp4')

    pbar.close()

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
    print("Patient {}".format(patient_no))
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












