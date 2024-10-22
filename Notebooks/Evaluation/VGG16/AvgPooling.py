import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import sys
sys.path.append('C:\\Users\\Eike\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\PoolingLayers')

print(tf.config.list_physical_devices('GPU'))

model_folder = "C:\\Users\\Eike\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Models\\VGG16\\"
test_data_folder = "C:\\Users\\Eike\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\Dataset\\preprocessed\\test"
save_folder = "C:\\Users\\Eike\\OneDrive\\Desktop\\Uni\\Masterarbeit\\MasterThesisProject\\results\\VGG16\\AvgPooling.csv"

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

model = tf.keras.models.load_model(model_folder + "AvgPooling.keras")
test_dataset = tf.data.Dataset.load(test_data_folder)


X, y = zip(*test_dataset)

print(type(X), type(y))

images_np = np.array(X)
labels_np = np.array(y)



predictions = model.predict(images_np)
y_pred = np.argmax(predictions, axis=-1)
report = classification_report(labels_np, y_pred, target_names=target_names, output_dict=True)

pd_report = pd.DataFrame(report).transpose()
pd_report.to_csv(save_folder)
print(pd_report)