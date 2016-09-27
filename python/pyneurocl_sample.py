import sys

sys.path.append("../lib")

import pyneurocl

h = pyneurocl.helper()
h.init('../nets/alpr/topology-alpr-let2.txt','../nets/alpr/weights-alpr-let2.bin')
h.uninit()
