import os, glob, time, itertools
import io
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' 
import tensorflow as tf
tf.compat.v1.enable_eager_execution()  
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras import layers, losses, optimizers
import csv, random  
import tensorflow.keras
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import utils
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight  
import collections
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
from tensorflow.keras import metrics
tf.compat.v1.enable_eager_execution()
assert tf.__version__.startswith('2.')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
lei = 2
batchsz = 8
amount_min = 100  
trainable = True
def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic
def load_qiao(root, mode='train'):
    path2label = {}
    name2label = {}
    name2amount = {}
    for name in sorted(os.listdir(os.path.join(root))):
        shuliang = 0
        if not os.path.isdir(os.path.join(root, name)):
            continue
        if not name in name2label:
            name2label[name] = len(name2label.keys())  
        genus = name
        for subname in sorted(os.listdir(os.path.join(root, name))):
            if not os.path.isdir(os.path.join(root, name, subname)):
                continue
            for subsubname in sorted(os.listdir(os.path.join(root, name, subname))):
                subsubnamee = os.path.join(root, name, subname, subsubname)  
                path2label[subsubnamee] = name2label[genus]  
                shuliang += 1
        bianma = name2label[genus]
        name2amount[bianma] = shuliang  
    filename = '01-bianmabiao.csv'
    if not os.path.exists(os.path.join(root, filename)):
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            path2label = random_dic(path2label) 
            for img in path2label:
                if name2amount[path2label[img]]>amount_min:
                    label = 1
                elif name2amount[path2label[img]]<=amount_min:
                    label = 0
                writer.writerow([img, label]) 
            print('writter into csv file:', filename)
    bianma = []  
    images = []  
    images1 = []  
    labels = []  
    labels1 = []  
    weight = {} 
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            images.append(img)
            label = int(label)  
            labels.append(label)
    lei = 2 
    if mode == 'train':
        images1 = [x for x in images if 'train' in x]
        for imgg in images1:
            a = images.index(imgg)
            b = labels[a]
            b = int(b)
            labels1.append(b)
        print('db_train：\n'
              'images:', len(images1), ',labels:', len(labels1), ',lei:', lei)
        print('images,labels', images1[0], labels1[0])
        weight = class_weight.compute_class_weight('balanced', np.unique(labels1), labels1)
        weight = dict(enumerate(weight))
        print(weight)

    elif mode == 'val':
        images1 = [x for x in images if 'val' in x]
        for imgg in images1:
            a = images.index(imgg)
            b = labels[a]
            b = int(b)
            labels1.append(b)
        print('db_val：\n'
              'images:', len(images1), ',labels:', len(labels1), ',lei:', lei)
        print('images,labels', images1[0], labels1[0])
    else:
        images1 = [x for x in images if 'test' in x]
        for imgg in images1:
            a = images.index(imgg)
            b = labels[a]
            b = int(b)
            labels1.append(b)
        print('db_test：\n'
              'images:', len(images1), ',labels:', len(labels1), ',lei:', lei)
        print('images,labels', images1[0], labels1[0])
    return images1, labels1, lei, weight 

img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])

def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    x = x * std + mean
    return x

def preprocess(x, y):
    u = tf.strings.substr(x, -1, 1)
    if u == 'p':
        x = tf.io.read_file(x)
        x = tf.image.decode_bmp(x, channels=3)
    else:
        x = tf.io.read_file(x)
        x = tf.image.decode_png(x, channels=3)
    x = tf.image.resize_with_pad(x, 299, 299)
    x = tf.image.random_crop(x, [299, 299, 3])
    x = tf.image.random_brightness(x, 0.5)
    x = tf.image.random_contrast(x, 0, +10)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x) 
    y = tf.convert_to_tensor(y)
    global lei
    lei = int(lei)
    y = tf.one_hot(y, depth=lei)  
    return x, y

images, labels, lei, weight = load_qiao(r'.\database', 'train')

db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)  

images2, labels2, lei2, weight2 = load_qiao(r'.\database', 'val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_vall = db_val.map(preprocess)
db_val = db_val.map(preprocess).batch(batchsz)

images3, labels3, lei3, weight2 = load_qiao(r'.\database', 'test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)
now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
now = str(now)
net = keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, pooling='max')  
net.trainable = trainable
newnet = keras.Sequential([
    net,
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.2),  
    layers.Dense(lei)
])
newnet.summary()
newnet.compile(optimizer=optimizers.Adam(lr=1e-4), loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='accuracy', min_delta=0.001, patience=10)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', min_delta=0.0001, factor=0.5, patience=5, verbose=1)  
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/model-' + str(trainable) + str(amount_min) + now + '.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
history1 = newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=50,
                      callbacks=[early_stopping, reduce_lr, model_checkpoint_callback], class_weight=weight)  
history = history1.history  
print(history)
max_accuracy = max(history['accuracy'])
min_loss = min(history['loss'])
max_val_accuracy = max(history['val_accuracy'])
min_val_loss = min(history['val_loss'])

print('max_accuracy=', max_accuracy, ',min_loss=', min_loss,
      'max_val_accuracy=', max_val_accuracy, ',min_val_loss=', min_val_loss)
test_acc = newnet.evaluate(db_test)
pre = np.argmax(newnet.predict(db_test), axis=-1)
print('pre:', pre)
print(',real:', labels3)
