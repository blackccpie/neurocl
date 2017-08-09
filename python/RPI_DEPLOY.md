### How to deploy OCR bundle on the raspberri pi

#### Main package

Upload & unzip the prebuilt binaries archive in the home directory :

```shell
$ cd
$ tar -zxf neurocl_armv6_ocr.tgz
```

NOTE : the package is included in the git repository in the `neurocl/packages` directory.

#### Dependencies

Install picamera & microdotphat dependencies:

```shell
$ sudo apt-get install python-picamera
$ curl -sS get.pimoroni.com/microdotphat | bash
```

For python script debug purposes (image viewing/saving):

```shell
$ sudo aptitude install python-pil
```

#### Auto-start

Create a startup shell script called *pi_ocr.sh* in your home directory, with the following content:

```shell
#!/bin/sh

cd /home/pi/neurocl_armv6_ocr
export LD_LIBRARY_PATH=/home/pi/neurocl_armv6_ocr
python pyneurocl_ocr.py
```

Configure cron to start your script at boot time:

```shell
$ sudo crontab -e
```

and add the following lines at the end of the file:

```shell
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pi/neurocl_armv6_ocr
@reboot sh /home/pi/pi_ocr.sh &
```

Reboot, and here you are!
