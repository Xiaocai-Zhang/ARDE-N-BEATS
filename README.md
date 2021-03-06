# ARDE-N-BEATS: A Deep Learning Framework for Traffic Flow Prediction
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.6 and TensorFlow 2.5.0. You can setup a virtual environment by running the code like this:
```
python3 -m venv env
source env/bin/activate
cd arde_n_beats
pip3 install -r requirements.txt
```
## Download Data Sets
Run the following commands to download data sets from cloud.
```
gdown https://drive.google.com/uc?id=1orQYfoFxCz9sxG7WIjBLXHlTm8-7xhdh
unzip data
```
## Running Models
Firstly, run the following codes to download the trained models.
```
gdown https://drive.google.com/uc?id=1dWD-9tkUYAbmfS_wFSawc_6oJ7rV4IKa
unzip save
```
Then, you can run the following command to replicate the results.
```
python3 test/XXXX/test_XXm.py
```
For example, for the I280-S 15-min prediction task, run
```
python3 test/I280-S/test_15m.py
```
The results are saved under the "./hypara/" folder.
## Training Models
You can run the following command to train models.
```
python3 train/XXXX/train_XXm.py
```
For example, for training the 60-min prediction models on M50-N data set, run command
```
python3 train/M50-N/train_60m.py
```
