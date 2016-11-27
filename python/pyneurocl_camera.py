# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from PIL import Image
import numpy, time, sys

sys.path.append("../lib")
import pyneurocl

print "---> pyneurocl - initialize"
h = pyneurocl.helper(False)
h.init('../nets/alpr/topology-alpr-let2.txt','../nets/alpr/weights-alpr-let2.bin')

# initialize the camera and grab a reference to the raw camera capture
print "---> picamera - initialize"
camera = PiCamera()
raw = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.1)

# configure camera
camera.resolution = (50, 100)
camera.color_effects = (128,128) # turn camera to black and white

# grab an image from the camera
camera.capture(raw, format="rgb")
image = Image.fromarray(raw.array, 'RGB')
#image.save('my.png')

output = numpy.zeros(shape=(100,50)) #rows,cols 

print "---> compute network output:"
h.compute(raw.array,output)
print output

#requires imagemagick to be installed!
#image.show()
