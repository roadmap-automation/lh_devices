import cv2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from capturetools import get_input_devices, normalize_histogram

FMT = '%Y%m%d_%H%M%S'

names = get_input_devices("FriendlyName")
paths = get_input_devices("DevicePath")
scopes = {}
for idx, (name, path) in enumerate(zip(names, paths)):
    if name == 'HD Camera':
        print(path)
        scopes[idx] = path.split('mi_00#a&')[1].split('&0&0')[0]

print(scopes)

#cam = cv2.VideoCapture(1, apiPreference=cv2.CAP_DSHOW)

#cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
#cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
#cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#cam.set(cv2.CAP_PROP_EXPOSURE, -5) # ranges from -1 to -13, can't go above -3 for this camera
#cam.set(cv2.CAP_PROP_APERTURE, -2)
#cam.set(cv2.CAP_PROP_FOCUS, -2)
#cam.set(cv2.CAP_PROP_GAIN, 64)      # no idea what this is really; ranges from 0 to 128
#cam.set(cv2.CAP_PROP_GAMMA, 120)    # default apparently

good_result = False
exposure = -8
while not good_result:
    cam = cv2.VideoCapture(list(scopes.keys())[0], apiPreference=cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    #cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
    #cam.set(cv2.CAP_PROP_GAIN, 128)      # no idea what this is really; ranges from 0 to 128
    #cam.set(cv2.CAP_PROP_GAMMA, 120)    # default apparently
    if cam.isOpened():
        print('reading')
        result, image = cam.read()
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if exposure != cam.get(cv2.CAP_PROP_EXPOSURE):
            print('breaking', exposure, cam.get(cv2.CAP_PROP_EXPOSURE))
            # stuck
            #break
        #exposure = cam.get(cv2.CAP_PROP_EXPOSURE)
        med_pixel = np.median(grayscale)
        print(exposure, med_pixel)
        if med_pixel < 100:
            exposure += 1.0
        elif med_pixel > 156:
            exposure -= 1.0
        else:
            good_result = True
    
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cam.release()

    #print(cam.get(cv2.CAP_PROP_APERTURE))
    #print(cam.get(cv2.CAP_PROP_SATURATION))
    #print(cam.get(cv2.CAP_PROP_EXPOSURE))
    #print(cam.get(cv2.CAP_PROP_FOCUS))
    #print(cam.get(cv2.CAP_PROP_GAIN))
    #print(cam.get(cv2.CAP_PROP_GAMMA))
    #print(result, image)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, cmap='gray')
print(image.shape)
newimage = np.zeros_like(image)
for i in range(3):
    newimage[:,:,i] = normalize_histogram(image[:,:,i])
ax[1].imshow(newimage)
#    ax[1].imshow(normalize_histogram(image), cmap='gray')
plt.show()

#cv2.imwrite(datetime.datetime.now().strftime(FMT) + '.png', image)
