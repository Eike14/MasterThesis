import tensorflow as tf
import sys
sys.path.append('C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\PoolingLayers')
from DropoutMaxPooling import DropoutMaxPoolingLayer

print(tf.config.list_physical_devices('GPU'))

data_folder = "C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Dataset\\preprocessed"
# Loading Datasets
train_path = data_folder+"\\train"
val_path = data_folder+"\\val"
test_path = data_folder+"\\test"
train_ds = tf.data.Dataset.load(train_path)
val_ds = tf.data.Dataset.load(val_path)
test_ds = tf.data.Dataset.load(test_path)



def replace_max_by_max_dropout_pooling(model):

    input_layer, *other_layers = model.layers
    assert isinstance(input_layer, tf.keras.layers.InputLayer)

    x = input_layer.output
    for layer in other_layers:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            layer = DropoutMaxPoolingLayer(
                pool_size=3,
                stride=2,
                drop_rate=0.5
            )
        x = layer(x)

    return tf.keras.models.Model(inputs=input_layer.input, outputs=x)

#Creating the model
input_shape = (224,224,3)

vgg = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape, weights="imagenet")
vgg.summary()
"""
vgg_small = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    DropoutMaxPoolingLayer(0.5, pool_size=3),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    DropoutMaxPoolingLayer(0.5, pool_size=3),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    DropoutMaxPoolingLayer(0.5, pool_size=3),
    tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    DropoutMaxPoolingLayer(0.5, pool_size=3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(25, activation="softmax")
])

vgg_small.summary()"""
vgg = replace_max_by_max_dropout_pooling(vgg)
vgg.trainable = False
vgg.summary()

flatten = tf.keras.layers.Flatten()
dense1 = tf.keras.layers.Dense(4096, activation="relu")
dense2 = tf.keras.layers.Dense(4096, activation="relu")
output = tf.keras.layers.Dense(25, activation="softmax")

x = flatten(vgg.output)
x = dense1(x)
x = dense2(x)
predictions = output(x)

model = tf.keras.Model(vgg.input, predictions)
print(model.summary())




def get_model_name(k):
    return "MaxDropoutPooling_"+str(k)+".keras"

save_dir = "C:\\Users\\eikei\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Models\\VGG16\\"

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

print("Model compiled")

# Training the model
batch_size=32
num_epochs=40
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+"MaxDropoutPooling.keras", 
			monitor='val_accuracy', verbose=1, 
			save_best_only=True, mode='max')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    elif (epoch == 3):
        return lr / 10
    elif (epoch == 6):
        return lr / 10
    else:
        return lr

learning_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=[early_stopping, checkpoint])