import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../Emergency_Vehicles/train"
test_dir = "../Emergency_Vehicles/test"

image = cv2.imread(r"../Emergency_Vehicles/train/1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# Show the shape of the image
print(image.shape)

# Data augmentation using ImageDataGenerator
image_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.20,
    zoom_range=0.20,
    fill_mode='nearest',
    horizontal_flip=True,
    rescale=1/255
)

plt.imshow(image_gen.random_transform(image))
plt.show()

# Load train and test dataframes
train_df = pd.read_csv("../Emergency_Vehicles/train.csv")
test_df = pd.read_csv("../Emergency_Vehicles/test.csv")

train_df['emergency_or_not'] = train_df['emergency_or_not'].astype(str)
test_df['emergency_or_not'] = test_df['emergency_or_not'].astype(str)

# Print train dataframe info
train_df.info()

# Create training and validation generators
train_generator = image_gen.flow_from_dataframe(
    dataframe=train_df[:1150],
    directory=train_dir,
    x_col='image_names',
    y_col='emergency_or_not',
    class_mode='binary',
    target_size=(224, 224),
    batch_size=50
)

validation_generator = image_gen.flow_from_dataframe(
    dataframe=train_df[1150:],
    directory=train_dir,
    x_col='image_names',
    y_col='emergency_or_not',
    class_mode='binary',
    target_size=(224, 224),
    batch_size=50
)

model = Sequential([ # Build the model
    Conv2D(filters=32, kernel_size=(3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# print model summary
model.summary()

# train the model
history = model.fit(train_generator, epochs=30, validation_data=validation_generator)

# plot training and validation loss
history_df = pd.DataFrame(history.history)
plt.figure(dpi=200, figsize=(10, 3))
plt.plot(history_df['loss'], label='training_loss')
plt.plot(history_df['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Prepare test data
test_dir_image = []
for i in tqdm(test_df.index):
    img_path = '../Emergency_Vehicles/test/' + test_df['image_names'][i]
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    test_dir_image.append(img)

test = np.array(test_dir_image)

# Predict on test data
pred = model.predict(test)

# Convert predictions to binary
num_ = np.floor(pred)

# Prepare submission
submission = pd.read_csv('../Emergency_Vehicles/sample_submission.csv')
submission['emergency_or_not'] = num_
submission.to_csv('submission.csv', index=False)

# Save the model
model.save('vehicle.h5')

# Load the model
model_vehicle = tf.keras.models.load_model('vehicle.h5', custom_objects={'KerasLayer': hub.KerasLayer})
