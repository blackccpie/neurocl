import os, sys, time, threading

microdot = False

if ( os.uname()[1] == 'rapsberry' ):
    try:
        from microdotphat import write_string
        microdot = True
    except ImportError:
        print "---> microdotphat module is not installed on your raspberry"

sys.path.append("../lib")

import pyneurocl

print "---> pyneurocl - start"

h = pyneurocl.helper(False)

def progression_worker(h):
    t_progress = 0
    print "---> pyneurocl - starting progression_worker"
    while (t_progress < 100) :
        time.sleep(1)
        t_progress = h.train_progress()
        if microdot:
            clear()
            write_string( str( t_progress ) + "%" )
            show()
        else:
        	print "--->" + str( t_progress ) + "%"
    return

try:
    print "---> pyneurocl - initialize"
    h.init('../nets/alpr/topology-alpr-let2.txt','../nets/alpr/weights-alpr-let2.bin')

    t = threading.Thread(target=progression_worker,args=(h,))
    t.start()

    # NOTE: python GIL is released during training so that worker thread can call h.train_progress()

    print "---> pyneurocl - train 10 epochs"
    h.train('../nets/alpr/alpr-train-let.txt',10,10)

    t.join()
except Exception:
    print "---> pyneurocl - uninitialize"
    h.uninit()

print "---> pyneurocl - end"
