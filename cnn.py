# installしたもの
# numpy
# keras
# pillow(PIL)

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# 学習用のデータを作る.
image_list = []
label_list = []

# データが格納されたディレクトリ名
basedir='/content/drive/MyDrive/Colab Notebooks/Human AI/Lec02/data/'

# ディレクトリ名(ラベル名)指定(3次元)
label_name=['car','train','ship']

# 訓練データのディレクトリ
traindir=basedir 

# ./data/train 以下の画像を読み込む。
for dir in os.listdir(traindir):
    if dir == ".DS_Store":
        continue

    dir1 = traindir + dir 
    label = 0

    if dir == label_name[0]:  
        label = 0
    elif dir == label_name[1]:
        label = 1
    elif dir == label_name[2]:
        label = 2

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を100x100pixelに変換し、1要素が[R,G,B]3要素を含む配列の100x100の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath).convert('RGB').resize((100, 100)))
            print(filepath)
            print(image.shape)
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0,0], 1 -> [0,1,0] という感じ。
Y = to_categorical(label_list)

# モデルを生成してニューラルネットを構築
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.summary()

# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
#model.fit(image_list, Y, epochs=1000, batch_size=25, callbacks=[early_stopping])
model.fit(image_list, Y, epochs=10, batch_size=25)

#from google.colab import files

# ファイルのアップロードとファイル名取得
uploaded = files.upload()

for fn in uploaded:
    filepath='./'+fn
print(filepath)

image = np.array(Image.open(filepath).convert('RGB').resize((100, 100)))

result = model.predict(np.array([image / 255.]))
print(result)
os.remove(filepath)