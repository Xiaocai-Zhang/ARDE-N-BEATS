# ARDE-N-BEATS: A Deep Learning Framework for Traffic Flow Prediction
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.6 and TensorFlow 2.5.0. You can setup a virtual environment by running the code like this:
```
python3 -m venv env
source env/bin/activate
cd env
pip3 install -r requirements.txt
```
## Download data sets
```
cd arde_n_beats
gdown https://drive.google.com/uc?id=1orQYfoFxCz9sxG7WIjBLXHlTm8-7xhdh
unzip data
```
## Running Models
You can run the following commands to replicate the results.
```
python3 script/test_m50.py
python3 script/test_i280.py
python3 script/test_nyc.py
```
The results are saved under the "./para/" folder.
