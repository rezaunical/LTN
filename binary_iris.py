import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ltn
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default=None)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=64)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']

# Load the iris dataset
iris = datasets.load_iris()
data = iris.data
labels = iris.target

# For simplicity, let's only keep Setosa (0) and Versicolour (1) for binary classification
# Discarding Virginica (2)
mask = labels < 2
data = data[mask]
labels = labels[mask]

# Split into training and testing datasets (50 samples each for train and test)
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42)

# Convert to TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices((data_train, labels_train)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((data_test, labels_test)).batch(batch_size)


# Extracting the data points based on labels
data_positive = data[labels == 0]  # Assuming 0 is Setosa (Positive)
data_negative = data[labels == 1]  # Assuming 1 is Versicolour (Negative)

# Plotting
plt.figure(figsize=(4, 4))
plt.scatter(data_positive[:, 0], data_positive[:, 1], c='blue', label='Setosa (Positive)')
plt.scatter(data_negative[:, 0], data_negative[:, 1], c='red', label='Versicolour (Negative)')
plt.title('Iris Data Visualization (First two features)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.grid(True)
plt.show()

# # LTN

A = ltn.Predicate.MLP([2],hidden_layer_sizes=(16,16))

# # Axioms
# 
# ```
# forall x_A: A(x_A)
# forall x_not_A: ~A(x_not_A)
# ```

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2),semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

@tf.function
def axioms(data, labels):
    data = tf.convert_to_tensor(data)
    labels = tf.cast(labels, dtype=tf.bool)
    
    x_A = ltn.Variable("x_A", tf.boolean_mask(data, labels))
    x_not_A = ltn.Variable("x_not_A", tf.boolean_mask(data, tf.math.logical_not(labels)))
    
    axioms = [
        Forall(x_A, A(x_A)),
        Forall(x_not_A, Not(A(x_not_A)))
    ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level



# Initialize all layers and the static graph.

for _data, _labels in ds_test:
    print("Initial sat level %.5f"%axioms(_data, _labels))
    break

# # Training
# 
# Define the metrics

metrics_dict = {
    'train_sat': tf.keras.metrics.Mean(name='train_sat'),
    'test_sat': tf.keras.metrics.Mean(name='test_sat'),
    'train_accuracy': tf.keras.metrics.BinaryAccuracy(name="train_accuracy",threshold=0.5),
    'test_accuracy': tf.keras.metrics.BinaryAccuracy(name="test_accuracy",threshold=0.5)
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step(data, labels):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(data, labels)
        loss = 1.-sat
    gradients = tape.gradient(loss, A.trainable_variables)
    optimizer.apply_gradients(zip(gradients, A.trainable_variables))
    metrics_dict['train_sat'](sat)
    # accuracy
    predictions = A.model(data)
    metrics_dict['train_accuracy'](labels,predictions)

@tf.function
def test_step(data, labels):
    # sat and update
    sat = axioms(data, labels)
    metrics_dict['test_sat'](sat)
    # accuracy
    predictions = A.model(data)
    metrics_dict['test_accuracy'](labels,predictions)


# The classification with MLP (same size network) (without LTN Rules)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(4,)), #the number here should be equal to number of features in dataset
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss for binary classification
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)


import commons

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path,
    track_metrics=20
)
