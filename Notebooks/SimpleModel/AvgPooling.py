import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Loading Datasets
train_path = "..\\..\\Dataset\\preprocessed\\train"
val_path = "..\\..\\Dataset\\preprocessed\\val"
test_path = "..\\..\\Dataset\\preprocessed\\test"
train_ds = tf.data.Dataset.load(train_path)
val_ds = tf.data.Dataset.load(val_path)
test_ds = tf.data.Dataset.load(test_path)

#Creating the model
input_shape = (224,224,3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape, padding="valid"),
    tf.keras.layers.AvgPool2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.AvgPool2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(25, activation="softmax")
])

model.compile(optimizer="SGD",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["sparse_categorical_accuracy"])

# Training the model
batch_size=32
num_epochs=20
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=[early_stopping])

model.save("..\\..\\Models\\SimpleModels\\AvgPooling.keras")

# Testing the model
test_ds_image = test_ds.map(lambda x,y: x)
test_ds_labels = test_ds.map(lambda x,y: y)

predictions = model.predict(test_ds_image, batch_size=32)
y_pred = np.argmax(predictions, axis=-1)

#Evaluating the results
y_true = np.array([])
for image_batch, labels_batch in test_ds:
    y_true = np.append(y_true, [labels_batch.numpy()])
    
target_names = [
    "Adialer.C",
    "Agent.FYI",
    "Allaple.A",
    "Allaple.L",
    "Alueron.genU",
    "Autorun.K",
    "C2LOP.gen!g",
    "C2LOP.P",
    "Dialplatform.B",
    "Dontovo.A",
    "Fakerean",
    "Instantaccess",
    "Lolyda.AA1",
    "Lolyda.AA2",
    "Lolyda.AA3",
    "Lolyda.AT",
    "Malex.gen!J",
    "Obfuscator.AD",
    "Rbot!gen",
    "Skintrim.N",
    "Swizzor,gen!E",
    "Swizzor.gen!I",
    "VB.AT",
    "Wintrim.BX",
    "Yuner.A"
]

report = classification_report(y_true, y_pred,target_names=target_names, output_dict=True)
pd_report = pd.DataFrame(report).transpose()
print(pd_report)
pd_report.to_csv("..\\..\\Results\\SimpleModel\\AvgPooling.csv")
