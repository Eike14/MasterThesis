import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd

path = "C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Dataset\\raw"
dataset = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                              batch_size=None,
                                                              image_size=(224,224),
                                                              seed=42)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_dataset = dataset.map(lambda x,y: (normalization_layer(x), y))

# Testing and Trying
X, y = zip(*tfds.as_numpy(normalized_dataset))
print(type(X), type(y))

images_np = np.array(X)
labels_np = np.array(y)
    
def create_model():
    input_shape=(224,224,3)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape, padding="same"),
        tf.keras.layers.AvgPool2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.AvgPool2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(25, activation="softmax")
    ])
    return model
    


def get_model_name(k):
    return "AvgPooling_"+str(k)+".h5"

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_var=1
save_dir = "C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Models\\SimpleModels\\"

for train_index, val_index in skf.split(images_np, labels_np):
    # Data
    training_images = images_np[train_index]
    training_labels = labels_np[train_index]
    validation_images = images_np[val_index]
    validation_labels = labels_np[val_index]
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')
    
    # Model Training
    model = create_model()
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer="SGD",
                  metrics=["accuracy"])
    
    history = model.fit(x=training_images,
                        y=training_labels,
                        batch_size=32,
                        epochs=10,
                        callbacks=[early_stopping, checkpoint],
                        validation_data=(validation_images, validation_labels))
    
    model.load_weights(save_dir+"AvgPooling_"+str(fold_var)+".h5")
    
    #Model Evaluating
    predictions = model.predict(validation_images)
    y_pred = np.argmax(predictions, axis=-1)
    report = classification_report(validation_labels, y_pred, output_dict=True)
    
    pd_report = pd.DataFrame(report).transpose()
    pd_report.to_csv("C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\results\\simpleModel\\AvgPoolingFold"+str(fold_var)+".csv")
    print(pd_report)
    tf.keras.backend.clear_session()
    fold_var += 1