# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from microdotphat import write_string, clear, show
import numpy, time, sys
#from PIL import Image

import pyneurocl

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return r/4 + g/2 + b/4

clear()
write_string("INIT 1",kerning=False)
show()

print "---> pyneurocl - initialize"
h = pyneurocl.helper(False)
h.init('mnist/topology-mnist-kaggle.txt','mnist/weights-mnist-kaggle.bin')

clear()
write_string("INIT 2",kerning=False)
show()

# initialize the camera and grab a reference to the raw camera capture
print "---> picamera - initialize"
camera = PiCamera()
raw = PiRGBArray(camera)

# allow the camera to warmup
time.sleep(0.1)

# configure camera
#camera.resolution = (128, 112)
#camera.color_effects = (128,128) # turn camera to black and white

recon = "xxxx"

clear()
write_string( recon + "-I", kerning=False )
show()

sleep = 10

while(True):

	if sleep > 0:
		time.sleep(1)
		sleep = sleep-1
		clear()
        	write_string( recon + "-" + str(sleep), kerning=False )
        	show()
	else:

		time.sleep(1)

		sleep = 10

		clear()
                write_string( recon + "-I", kerning=False )
                show()

		# grab an image from the camera
		camera.capture(raw, format="rgb")

        input = rgb2gray(raw.array)

		#print(raw.array.shape)
		#image = Image.fromarray(input, 'L')
		#image.save('my.png')
		#requires imagemagick to be installed!
		#image.show()

		time.sleep(1)

		clear()
		write_string( recon + "-C", kerning=False )
		show()

		print "---> compute ocr output:"
		# NOTE : raw.array.dtype is uint8
		reco_out = h.digit_recognizer( input )
		print reco_out

		blanksize = 0

		if len(reco_out) <= 4:
			blanksize = 4 - len(reco_out)

		recon = ( "_" * blanksize ) + reco_out[0:4]

		clear()
		write_string( recon + "-R", kerning=False )
		show()

		raw.truncate(0)

		time.sleep(2)
