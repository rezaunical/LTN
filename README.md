# Binary Classification with and without LTN 
## same neural network size and structure
### Comparison of LTN for binary classification

# What to expect?
The goal is to evaluate the performance of LTN for binary classification tasks over different datasets. \
The files are: \
1- binary_classification.py # uses the same dataset of the original problem in the LTN paper. \
2- binary_moons.py # uses the moons dataset \
3- binary_iris.py # uses the iris dataset \
4- binary_synthetic # uses a synthetic dataset with 50 features \

# How to use  
If you use Jupyter in your system: \
1- !git clone https://github.com/rezaunical/LTN.git \
2- cd LTN # name of the directory in your system \
3- !pip install ltn # [install the LTN library] \
4- run binary_classification.py # or any other files inside the repository. \


# The results

## 1- [binary_classification.py] 
![binary](https://github.com/rezaunical/LTN/assets/137186931/562756c5-053b-47c4-813f-66d09c935806)

### The selected ones for LTN: 
Epoch 0, train_sat: 0.5000, test_sat: 0.5020, train_accuracy: 0.4800, test_accuracy: 0.2800 \
Epoch 20, train_sat: 0.5021, test_sat: 0.5032, train_accuracy: 0.5200, test_accuracy: 0.5400 \
Epoch 40, train_sat: 0.5041, test_sat: 0.5046, train_accuracy: 0.5600, test_accuracy: 0.5400 \
Epoch 60, train_sat: 0.5070, test_sat: 0.5068, train_accuracy: 0.6200, test_accuracy: 0.5400 \
Epoch 80, train_sat: 0.5111, test_sat: 0.5101, train_accuracy: 0.7400, test_accuracy: 0.6400 \
Epoch 100, train_sat: 0.5171, test_sat: 0.5152, train_accuracy: 0.7400, test_accuracy: 0.7200 \
Epoch 120, train_sat: 0.5254, test_sat: 0.5225, train_accuracy: 0.7400, test_accuracy: 0.8000 \
Epoch 140, train_sat: 0.5364, test_sat: 0.5324, train_accuracy: 0.7200, test_accuracy: 0.7800 \
Epoch 160, train_sat: 0.5498, test_sat: 0.5447, train_accuracy: 0.7200, test_accuracy: 0.8000 \
Epoch 180, train_sat: 0.5649, test_sat: 0.5583, train_accuracy: 0.7200, test_accuracy: 0.8000 \
Epoch 200, train_sat: 0.5807, test_sat: 0.5722, train_accuracy: 0.7400, test_accuracy: 0.7800 \
Epoch 220, train_sat: 0.5975, test_sat: 0.5857, train_accuracy: 0.8000, test_accuracy: 0.7800 \
Epoch 240, train_sat: 0.6164, test_sat: 0.5994, train_accuracy: 0.8200, test_accuracy: 0.8400 \
Epoch 260, train_sat: 0.6373, test_sat: 0.6134, train_accuracy: 0.9000, test_accuracy: 0.8600 \
Epoch 280, train_sat: 0.6595, test_sat: 0.6267, train_accuracy: 0.9200, test_accuracy: 0.8400 \
Epoch 300, train_sat: 0.6807, test_sat: 0.6383, train_accuracy: 0.9000, test_accuracy: 0.8200 \
... \
Epoch 540, train_sat: 0.8563, test_sat: 0.7572, train_accuracy: 1.0000, test_accuracy: 0.8800 

### The selected ones for Non-LTN MLP 
Epoch 354/1000-  loss: 0.4893 - accuracy: 0.6600 - val_loss: 0.4360 - val_accuracy: 0.8000 \
Epoch 355/1000-  loss: 0.4885 - accuracy: 0.6800 - val_loss: 0.4353 - val_accuracy: 0.8000 \
Epoch 356/1000-  loss: 0.4877 - accuracy: 0.7000 - val_loss: 0.4347 - val_accuracy: 0.8000 \
... \
Epoch 470/1000-  loss: 0.3809 - accuracy: 0.9000 - val_loss: 0.3521 - val_accuracy: 0.8400 \
... \
Epoch 738/1000- loss: 0.1667 - accuracy: 1.0000 - val_loss: 0.2012 - val_accuracy: 0.9400 

## 2- [binary_moons.py] 
![moons](https://github.com/rezaunical/LTN/assets/137186931/0b960006-d944-4789-8321-970a7627bbff)

### The selected ones for LTN 
Epoch 0, train_sat: 0.4471, test_sat: 0.4642, train_accuracy: 0.2200, test_accuracy: 0.4200 \
Epoch 20, train_sat: 0.5307, test_sat: 0.5238, train_accuracy: 0.8800, test_accuracy: 0.7200 \
Epoch 40, train_sat: 0.6004, test_sat: 0.5649, train_accuracy: 0.9200, test_accuracy: 0.7600 \
Epoch 60, train_sat: 0.6541, test_sat: 0.5909, train_accuracy: 0.9000, test_accuracy: 0.7400 \
Epoch 80, train_sat: 0.6933, test_sat: 0.6064, train_accuracy: 0.9000, test_accuracy: 0.7600 \
Epoch 100, train_sat: 0.7216, test_sat: 0.6163, train_accuracy: 0.9000, test_accuracy: 0.7600 \
Epoch 120, train_sat: 0.7423, test_sat: 0.6244, train_accuracy: 0.9000, test_accuracy: 0.7600 \
Epoch 140, train_sat: 0.7577, test_sat: 0.6310, train_accuracy: 0.9000, test_accuracy: 0.7800 \
Epoch 160, train_sat: 0.7692, test_sat: 0.6359, train_accuracy: 0.9400, test_accuracy: 0.7800 \
Epoch 180, train_sat: 0.7778, test_sat: 0.6388, train_accuracy: 0.9400, test_accuracy: 0.7800 \
... \
Epoch 780, train_sat: 0.8227, test_sat: 0.6407, train_accuracy: 0.9800, test_accuracy: 0.8200 

### The selected ones for Non-LTN MLP 

Epoch 1/1000-   loss: 0.7040 - accuracy: 0.4200 - val_loss: 0.6877 - val_accuracy: 0.5800 \
... \
Epoch 452/1000- loss: 0.0971 - accuracy: 0.9800 - val_loss: 0.3894 - val_accuracy: 0.780 \
... \
Epoch 910/1000- loss: 0.0266 - accuracy: 1.0000 - val_loss: 0.2111 - val_accuracy: 0.90 \


## 3- [binary_iris.py] 
![iris](https://github.com/rezaunical/LTN/assets/137186931/972b86b4-9831-4d87-9464-41db49777c9d)

###The selected ones for LTN:  
Epoch 0, train_sat: 0.4621, test_sat: 0.4707, train_accuracy: 0.2200, test_accuracy: 0.2400 \
Epoch 20, train_sat: 0.5262, test_sat: 0.5351, train_accuracy: 0.8000, test_accuracy: 0.6600 \
Epoch 40, train_sat: 0.6079, test_sat: 0.6216, train_accuracy: 0.9800, test_accuracy: 1.0000 \
Epoch 60, train_sat: 0.6952, test_sat: 0.7108, train_accuracy: 1.0000, test_accuracy: 1.0000 

### The selected ones for Non-LTN MLP 
Epoch 1/1000-  loss: 1.0407 - accuracy: 0.4000 - val_loss: 0.7410 - val_accuracy: 0.6000 \
... \
Epoch 25/1000- loss: 0.5927 - accuracy: 0.7000 - val_loss: 0.5261 - val_accuracy: 0.9000 \
Epoch 26/1000- loss: 0.5818 - accuracy: 0.8000 - val_loss: 0.5235 - val_accuracy: 0.9600 \
Epoch 27/1000- loss: 0.5715 - accuracy: 0.9400 - val_loss: 0.5212 - val_accuracy: 0.9800 \
Epoch 28/1000- loss: 0.5618 - accuracy: 0.9800 - val_loss: 0.5191 - val_accuracy: 0.9800 \
Epoch 29/1000- loss: 0.5527 - accuracy: 0.9800 - val_loss: 0.5173 - val_accuracy: 0.9800 \
Epoch 30/1000- loss: 0.5442 - accuracy: 1.0000 - val_loss: 0.5157 - val_accuracy: 1.0000 

## 4- [binary_synthetic.py] 
![synthetic](https://github.com/rezaunical/LTN/assets/137186931/964f79b1-e805-462f-9234-39ef1a5d7c5a)

### The selected ones for LTN: 
Epoch 0, train_sat: 0.3684, test_sat: 0.3634, train_accuracy: 0.5120, test_accuracy: 0.5200 \
Epoch 20, train_sat: 0.6438, test_sat: 0.4924, train_accuracy: 0.8420, test_accuracy: 0.6160 \
Epoch 40, train_sat: 0.7365, test_sat: 0.4872, train_accuracy: 0.9360, test_accuracy: 0.6280 \
Epoch 60, train_sat: 0.8086, test_sat: 0.4763, train_accuracy: 0.9720, test_accuracy: 0.6560 \
Epoch 80, train_sat: 0.8730, test_sat: 0.4699, train_accuracy: 0.9880, test_accuracy: 0.6580 \
Epoch 100, train_sat: 0.9135, test_sat: 0.4612, train_accuracy: 0.9920, test_accuracy: 0.6520 \
Epoch 120, train_sat: 0.9354, test_sat: 0.4564, train_accuracy: 0.9940, test_accuracy: 0.6520 \
... \
Epoch 980, train_sat: 0.9556, test_sat: 0.4458, train_accuracy: 0.9940, test_accuracy: 0.6700 

### The selected ones for Non-LTN MLP 
Epoch 1/1000- loss: 1.8175 - accuracy: 0.4580 - val_loss: 1.6153 - val_accuracy: 0.4940 \
... \
Epoch 59/1000- loss: 0.2991 - accuracy: 0.8980 - val_loss: 0.7383 - val_accuracy: 0.6440 \
Epoch 60/1000- loss: 0.2944 - accuracy: 0.9000 - val_loss: 0.7406 - val_accuracy: 0.6480 \
... \
Epoch 199/1000- loss: 0.0215 - accuracy: 0.9980 - val_loss: 1.4197 - val_accuracy: 0.6560 \
Epoch 200/1000- loss: 0.0211 - accuracy: 1.0000 - val_loss: 1.4260 - val_accuracy: 0.6560 \
... \
Epoch 1000/1000- loss: 5.3328e-05 - accuracy: 1.0000 - val_loss: 3.2335 - val_accuracy: 0.6420 


