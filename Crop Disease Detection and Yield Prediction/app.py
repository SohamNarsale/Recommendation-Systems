import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPool2D
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD,RMSprop

dataset = r"dataset_path"
train_ds = keras.utils.image_dataset_from_directory(
    directory=dataset,
    validation_split=0.2,  # Use 20% for validation
    subset="training",     # This is the training subset
    seed=123,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224)
)

# Create the validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    directory=dataset,
    validation_split=0.2,  # Use the same split percentage
    subset="validation",   # This is the validation subset
    seed=123,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224)
)

class_names =  train_ds.class_names
print(class_names)

class_names = train_ds.class_names

plt.figure(figsize=(20, 20))
for images, labels in train_ds.take(1):
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        # Convert one-hot encoded label to class index
        label_index = np.argmax(labels[i])
        
        plt.title(class_names[label_index])
        plt.axis("off")

cnn = tf.keras.models.Sequential()

# Block 1
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[224, 224, 3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())  # Added Batch Normalization
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))  # Added Dropout

# Block 2
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())  # Added Batch Normalization
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.3))  # Added Dropout

# Block 3
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())  # Added Batch Normalization
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.4))  # Increased Dropout

# Block 4
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())  # Added Batch Normalization
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.4))  # Increased Dropout

# Global Average Pooling
cnn.add(tf.keras.layers.GlobalAveragePooling2D())  # Replaced Flatten with GAP

# Fully Connected Layers
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))  # Reduced units
cnn.add(tf.keras.layers.Dropout(0.5))  # Increased Dropout to prevent overfitting
cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))  # Output layer

# Compile the model
cnn.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.summary()

training_history = cnn.fit(
    x=train_ds,              # Training dataset
    validation_data=val_ds,  # Validation dataset
    epochs=10,                # Number of epochs
    verbose=True
)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(training_history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, training_history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, training_history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, training_history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, training_history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# Evaluate the model
test_loss, test_accuracy = cnn.evaluate(val_ds)
print(f"Validation Accuracy: {test_accuracy*100:.2f}%")

import matplotlib.pyplot as plt

# Extract accuracy and loss from training history
epochs = range(1, len(training_history.history['accuracy']) + 1)

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, training_history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, training_history.history['loss'], label='Training Loss')
plt.plot(epochs, training_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store images and labels
val_images_list = []
val_labels_list = []

# Loop through the validation dataset and collect 86 images
for images, labels in val_ds.take(86 // 32 + 1):  # Loop to get enough images (86 in this case)
    val_images_list.append(images)
    val_labels_list.append(labels)
    if len(val_images_list) * 32 >= 86:  # Stop when we have enough images
        break

# Stack the images and labels to form the full list
val_images = np.concatenate(val_images_list, axis=0)[:86]
val_labels = np.concatenate(val_labels_list, axis=0)[:86]

# Get predictions
predictions = cnn.predict(val_images)  # No need to rescale for predictions now
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(val_labels, axis=1)

# Class names inferred from your dataset structure
class_names = val_ds.class_names  # E.g., ['Healthy', 'Late_Blight']

# Plot 86 images with their predicted and true labels
plt.figure(figsize=(20, 40))  # Adjust figure size for better spacing
for i in range(86):
    plt.subplot(10, 9, i + 1)  # 10 rows, 9 columns
    plt.imshow(val_images[i].astype("uint8"))  # Convert to uint8 for proper visualization
    plt.title(f"True: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save the entire model
cnn.save('my_cnn_model.h5')  # Saves as a single HDF5 file

# Load the model
loaded_model = tf.keras.models.load_model('my_cnn_model.h5')

# Evaluate the model
test_loss, test_accuracy = loaded_model.evaluate(val_ds)
print(f"Validation Accuracy: {test_accuracy:.2f}")


######################################################################################################################
####                                                                                                              #### 
######################################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"csv_path")
# Feature engineering and preprocessing
X = data.drop(columns=['Yeild (Q/acre)'])  # Independent variables
y = data['Yeild (Q/acre)']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for models sensitive to feature scaling (SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train models
best_model = XGBRegressor(random_state=42, n_estimators=500, max_depth=10, learning_rate=0.1)
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost - MSE: {mse:.2f}, R^2: {r2:.2f}")

def recommend_optimal_conditions(model, X_sample, increments, target_increase):
    """
    Suggest improvements to inputs for yield optimization.
    Args:
        model: Trained regression model.
        X_sample: Sample input data (as a DataFrame).
        increments: Dictionary of parameter increments (e.g., {'Rain Fall (mm)': 50}).
        target_increase: Desired increase in yield.
    Returns:
        Modified parameters with recommendations.
    """
    X_optimal = X_sample.copy()
    initial_yield = model.predict(X_sample)[0]
    target_yield = initial_yield + target_increase

    for param, increment in increments.items():
        while model.predict(X_optimal)[0] < target_yield:
            X_optimal.loc[:, param] += increment
            if X_optimal.loc[:, param].iloc[0] > X_sample.loc[:, param].iloc[0] * 1.5:
                break  # Stop if the increment exceeds a reasonable limit

    final_yield = model.predict(X_optimal)[0]
    return X_optimal, final_yield

from tabulate import tabulate

# Example usage
sample_input = pd.DataFrame(X_test.iloc[0]).T
increments = {'Rain Fall (mm)': 50, 'Fertilizer': 5, 'Nitrogen (N)': 2}
target_increase = 2  # Targeting 2 Q/acre improvement

optimal_conditions, predicted_yield = recommend_optimal_conditions(
    best_model, sample_input, increments, target_increase
)

# Prepare data for tabular output
original_data = sample_input.reset_index(drop=True)
optimal_data = optimal_conditions.reset_index(drop=True)

# Add predicted yield as a separate row
original_data['Predicted Yield (Q/acre)'] = best_model.predict(sample_input)[0]
optimal_data['Predicted Yield (Q/acre)'] = predicted_yield

# Combine for comparison
comparison_table = pd.concat([original_data, optimal_data], keys=['Original', 'Optimal'])

# Display as a table
print(tabulate(comparison_table, headers="keys", tablefmt="fancy_grid"))


