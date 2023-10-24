import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ltn
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

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


# Generate a challenging synthetic dataset
X, y = make_classification(n_samples=1000, 
                           n_features=50, 
                           n_informative=30, 
                           n_redundant=10, 
                           n_clusters_per_class=2, 
                           class_sep=0.5, 
                           flip_y=0.3, 
                           random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42)

# Visualize the first two dimensions for a glimpse (though this won't capture the complexity)
plt.figure(figsize=(6, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='Class 1')
plt.title('Visualization of First Two Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Data (from the synthetic dataset)
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# # LTN

A = ltn.Predicate.MLP([50],hidden_layer_sizes=(16,16))

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
    labels = tf.cast(labels, tf.bool)
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
    tf.keras.layers.Input(shape=(50,)), #the number here should be equal to number of features in dataset
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
