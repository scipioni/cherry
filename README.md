14 giri al minuto, diametro 74mm

2 centesimi di secondo


325cm al minuto

5 cm al secondo


  248  curl http://www.linux-projects.org/listing/uv4l_repo/lrkey.asc | sudo apt-key add -
  249  vi /etc/apt/sources.list
  250  apt-get update
  251  apt-get install uv4l uv4l-raspicam
  252  uv4l --driver raspicam --auto-video_nr --width=640 --height=480 --encoding=yuv420 --framerate 90
  253  top
  254  v4l2-ctl --stream-mmap=3 --stream-count=1000
  255  v4l2-ctl 
  256   v4l2-ctl --all
  257  python test2.py 
  258  python test2.py 
  259  python test2.py 
  260  v4l2-ctl --stream-mmap=3 --stream-count=1000
  261  uv4l --driver raspicam --auto-video_nr --width=640 --height=480 --encoding=yuv420 --framerate 30
  262  uv4l --help
  263  reboot
  264  cd /lab/cherrycasta/
  265  uv4l --help
  266  #modprobe bcm-2835-v4l2
  267  dmesg -c
  268  modprobe bcm-2835-v4l2
  269  apt-cache search v4l2
  270  find /lib/modules/ -name "*v4l2*"
  271  modprobe bcm2835-v4l2

/etc/modules-load.d/modules.conf
# /etc/modules: kernel modules to load at boot time.
#
# This file contains the names of kernel modules that should be loaded
# at boot time, one per line. Lines beginning with "#" are ignored.

cuse
bcm2835-v4l2

