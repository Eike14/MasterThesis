import tensorflow as tf
#import tensorflow_datasets as tfds
#import pandas as pd
#from sklearn.metrics import classification_report
import sys
sys.path.append('C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\PoolingLayers')
from RankBasedWeightedPooling import RankBasedWeightedPoolingLayer

print(tf.config.list_physical_devices('GPU'))

data_folder = "C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Dataset\\preprocessed"
# Loading Datasets
train_path = data_folder+"\\train"
val_path = data_folder+"\\val"
test_path = data_folder+"\\test"
train_ds = tf.data.Dataset.load(train_path)
val_ds = tf.data.Dataset.load(val_path)
test_ds = tf.data.Dataset.load(test_path)


#Creating the model
input_shape = (224,224,3)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=32),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    RankBasedWeightedPoolingLayer(alpha=0.5),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    RankBasedWeightedPoolingLayer(alpha=0.5),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(25, activation="softmax")
])
 
def get_model_name(k):
    return "RWPPooling_"+str(k)+".keras"

save_dir = "C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Models\\SimpleModels\\"

model.compile(optimizer="SGD",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

print("Model compiled")

# Training the model
batch_size=32
num_epochs=20
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+"RWPPooling.keras", 
			monitor='val_accuracy', verbose=1, 
			save_best_only=True, mode='max')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=[early_stopping, checkpoint])