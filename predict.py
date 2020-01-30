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
from skimage.exposure import adjust_gamma
from skimage import color, img_as_float

with open('config.json') as config_file:
    config = json.load(config_file)


def gen_prediction_mask(background, mask, model_name, patient, slice):
    ones = np.argwhere(mask == 1)
    twos = np.argwhere(mask == 2)
    threes = np.argwhere(mask == 3)
    fours = np.argwhere(mask == 4)

    fig = plt.figure()

    background = img_as_float(background)
    background = adjust_gamma(color.gray2rgb(background), 0.65)
    bg_copy = background.copy()
    red = [1, 0.2, 0.2]
    yellow = [1, 1, 0.25]
    green = [0.35, 0.75, 0.25]
    blue = [0, 0.25, 0.9]

    print('ones shape: {}'.format(ones.shape))


    for i in xrange(len(ones)):
        bg_copy[ones[i][0]][ones[i][1]] = red
    for i in xrange(len(twos)):
        bg_copy[twos[i][0]][twos[i][1]] = green 
    for i in xrange(len(threes)):
        bg_copy[threes[i][0]][threes[i][1]] = blue 
    for i in xrange(len(fours)):
        bg_copy[fours[i][0]][fours[i][1]] = yellow
    
    plt.imshow(bg_copy)
    plt.savefig('outputs/{}_pat{}_slice{}.png'.format(model_name, patient, slice))
    plt.close(fig)


    








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

    pbar = tqdm(total = scans.shape[0])

    #ims = []

    for slice_no in range(scans.shape[0]):
        
        test_slice = scans[slice_no:slice_no+1,:,:,:4]
        test_label = scans[slice_no:slice_no+1,:,:,4]
        prediction = model.predict(test_slice, batch_size=32)
        prediction = prediction[0]
        prediction = np.around(prediction)
        prediction = np.argmax(prediction, axis=-1)
        gt.extend(test_label[0])
        pred.extend(prediction)

        scan = test_slice[0,:,:,2]
        label = test_label[0]

        gen_prediction_mask(scan, label, model_name, patient_no, slice_no)

        '''
        im = plt.figure(figsize=(15, 10))
        im = plt.subplot(131)
        im = plt.title('Input')
        im = plt.imshow(scan, cmap='gray')
        im = plt.subplot(132)
        im = plt.title('Ground Truth')
        im = plt.imshow(label,cmap='gray')
        im = plt.subplot(133)
        im = plt.title('Prediction')
        im = plt.imshow(prediction,cmap='gray')
        '''

        #plt.imshow(label, cmap='gray', animated=True)
        #plt.imshow(prediction, cmap='jet', alpha=0.5, animated=True)
        #plt.savefig('outputs/{}_pat{}_slice{}.png'.format(model_name, patient_no, slice_no))

        pbar.update(1)




        #ims.append([prediction_img])
        #ims.append([im])
        #plt.close(im)

#    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
#    ani.save('test.gif')


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












