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
Run the following commands to get the trained models.
```
gdown https://drive.google.com/uc?id=1dWD-9tkUYAbmfS_wFSawc_6oJ7rV4IKa
unzip save
```
Then, you can run the following commands to replicate the results.
```
python3 test/XXXX/test_XXm.py
```
For example, for the I280-S 15-min prediction task, run
```
python3 test/I280-S/test_15m.py
```
The results are saved under the "./hypara/" folder.
