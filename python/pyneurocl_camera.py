# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy, time, sys
#from PIL import Image

sys.path.append("../lib")
import pyneurocl

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return r/4 + g/2 + b/4

print "---> pyneurocl - initialize"
h = pyneurocl.helper(True)
h.init('../nets/mnist/topology-mnist-kaggle.txt','../nets/mnist/weights-mnist-kaggle.bin')

# initialize the camera and grab a reference to the raw camera capture
print "---> picamera - initialize"
camera = PiCamera()
raw = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.1)

# configure camera
#camera.resolution = (128, 112)
#camera.color_effects = (128,128) # turn camera to black and white

# grab an image from the camera
camera.capture(raw, format="rgb")

#print(raw.array.shape)
#image = Image.fromarray(rgb2gray(raw.array), 'L')
#image.save('my.png')
#requires imagemagick to be installed!
#image.show()

print "---> compute ocr output:"
# NOTE : raw.array.dtype is uint8
output = h.digit_recognizer( rgb2gray(raw.array) )
print output
