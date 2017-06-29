# setup virtualenv
sudo apt install -y python-virtualenv
sudo apt install -y python-opencv
#create home lib directory
virtualenv lib
. lib/bin/activate
python setup.py develop

# normal usage
. lib/bin/activate

# debug
cherry-run --show --file samples/calibro-28-01.264


# capture video
experiments/capture calibro-24-0X.264
