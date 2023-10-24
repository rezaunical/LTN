# Binary Classification with and without LTN (same neural network size and structure)
Comparison of LTN for binary classification

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

1- [binary_classification.py] \
![binary](https://github.com/rezaunical/LTN/assets/137186931/562756c5-053b-47c4-813f-66d09c935806)

The first 300 epochs for LTN \
Epoch 0, train_sat: 0.5000, test_sat: 0.5020, train_accuracy: 0.4800, test_accuracy: 0.2800\
Epoch 20, train_sat: 0.5021, test_sat: 0.5032, train_accuracy: 0.5200, test_accuracy: 0.5400\
Epoch 40, train_sat: 0.5041, test_sat: 0.5046, train_accuracy: 0.5600, test_accuracy: 0.5400\
Epoch 60, train_sat: 0.5070, test_sat: 0.5068, train_accuracy: 0.6200, test_accuracy: 0.5400\
Epoch 80, train_sat: 0.5111, test_sat: 0.5101, train_accuracy: 0.7400, test_accuracy: 0.6400\
Epoch 100, train_sat: 0.5171, test_sat: 0.5152, train_accuracy: 0.7400, test_accuracy: 0.7200\
Epoch 120, train_sat: 0.5254, test_sat: 0.5225, train_accuracy: 0.7400, test_accuracy: 0.8000\
Epoch 140, train_sat: 0.5364, test_sat: 0.5324, train_accuracy: 0.7200, test_accuracy: 0.7800\
Epoch 160, train_sat: 0.5498, test_sat: 0.5447, train_accuracy: 0.7200, test_accuracy: 0.8000\
Epoch 180, train_sat: 0.5649, test_sat: 0.5583, train_accuracy: 0.7200, test_accuracy: 0.8000\
Epoch 200, train_sat: 0.5807, test_sat: 0.5722, train_accuracy: 0.7400, test_accuracy: 0.7800\
Epoch 220, train_sat: 0.5975, test_sat: 0.5857, train_accuracy: 0.8000, test_accuracy: 0.7800\
Epoch 240, train_sat: 0.6164, test_sat: 0.5994, train_accuracy: 0.8200, test_accuracy: 0.8400\
Epoch 260, train_sat: 0.6373, test_sat: 0.6134, train_accuracy: 0.9000, test_accuracy: 0.8600\
Epoch 280, train_sat: 0.6595, test_sat: 0.6267, train_accuracy: 0.9200, test_accuracy: 0.8400\
Epoch 300, train_sat: 0.6807, test_sat: 0.6383, train_accuracy: 0.9000, test_accuracy: 0.8200\
...\
Epoch 540, train_sat: 0.8563, test_sat: 0.7572, train_accuracy: 1.0000, test_accuracy: 0.8800\

The selected ones for Non-LTN MLP \
Epoch 354/1000-  loss: 0.4893 - accuracy: 0.6600 - val_loss: 0.4360 - val_accuracy: 0.8000\
Epoch 355/1000-  loss: 0.4885 - accuracy: 0.6800 - val_loss: 0.4353 - val_accuracy: 0.8000\
Epoch 356/1000-  loss: 0.4877 - accuracy: 0.7000 - val_loss: 0.4347 - val_accuracy: 0.8000\
...\
Epoch 470/1000-  loss: 0.3809 - accuracy: 0.9000 - val_loss: 0.3521 - val_accuracy: 0.8400\
...\
Epoch 738/1000- loss: 0.1667 - accuracy: 1.0000 - val_loss: 0.2012 - val_accuracy: 0.9400\

2- [binary_moons.py] \




