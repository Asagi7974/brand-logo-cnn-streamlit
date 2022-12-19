

# ライブラリ一覧
import streamlit as st
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
# from google.colab import files

#
st.title("Brand Logo Recognition WebApp")
st.image("./header.png")
st.markdown("### Supported brands: Adidas, Nike, Puma")
# Developed by Hirose, Nozawa, Yokoyama, Hamasaki and Kinjo
st.caption("Developed by Hirose, Nozawa, Yokoyama, Hamasaki and Kinjo")

st.markdown("## Upload your image")

# # 学習用のデータを作る.
# image_list = []
# label_list = []

# # データが格納されたディレクトリ名
# ## エラーが場合はターミナルで cd と打つ
# # basedir='/User/mizuki/brand-logo-recognirion/images/'

# # ディレクトリ名(ラベル名)指定(3次元)
# label_name=['adidas','nike','puma'] # ,'underarmor'

# # 訓練データのディレクトリ
# # traindir=basedir

# # ./data/train 以下の画像を読み込む。
# ## エラーが出た場合は cd brand-logo-recognition と打つ
# for dir in os.listdir('images'):
#     if dir == ".DS_Store":
#         continue

#     dir1 = 'images/' + dir 
#     label = 0

#     if dir == label_name[0]:  
#         label = 0
#     elif dir == label_name[1]:
#         label = 1
#     elif dir == label_name[2]:
#         label = 2
# #    elif dir == label_name[3]:
# #        label = 3

#     print("os.listdir" + dir)
#     for file in os.listdir(dir1):
#         if file != ".DS_Store":
#             # 配列label_listに正解ラベルを追加
#             label_list.append(label)
#             filepath = dir1 + "/" + file
#             # 画像を100x100pixelに変換し、1要素が[R,G,B]3要素を含む配列の100x100の２次元配列として読み込む。
#             # [R,G,B]はそれぞれが0-255の配列。
#             image = np.array(Image.open(filepath).convert('RGB').resize((100, 100)))
#             print(filepath)
#             print(image.shape)
#             # 出来上がった配列をimage_listに追加。
#             image_list.append(image / 255.)

# # kerasに渡すためにnumpy配列に変換。
# image_list = np.array(image_list)

# # ラベルの配列を1と0からなるラベル配列に変更
# # 0 -> [1,0,0], 1 -> [0,1,0] という感じ。
# print("=========================================")
# print(label_list)
# print("=========================================")
# Y = to_categorical(label_list)
# print("=========================================")
# print(Y)
# print("=========================================")

# # モデルを生成してニューラルネットを構築
# print(1)
# model = Sequential()
# print(2)
# model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3), padding='same', activation='relu')) # うごかない(100,100,3) -> (100,100,4)
# print(3)
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# print(4)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# print(5)

# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# print(6)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# print(7)
# model.add(Dropout(0.2))
# print(8)
# model.add(Flatten())
# print(9)
# model.add(Dense(128, activation='relu'))
# print(10)
# model.add(Dropout(0.2))
# print(11)
# model.add(Dense(128, activation='relu'))
# print(12)
# model.add(Dropout(0.2))
# print(13)
# model.add(Dense(3, activation='softmax')) # 4 -> 3
# print(14)

# model.summary()
# print(15)
# # モデルをコンパイル
# model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
# print(16)
# # 学習を実行。10%はテストに使用。
# #model.fit(image_list, Y, epochs=1000, batch_size=25, callbacks=[early_stopping])
# model.fit(image_list, Y, epochs=10, batch_size=15)
# print(17)

# ファイルのアップロードとファイル名取得
#uploaded = '/User/mizuki/brand-logo-recognition/images/test'

#for fn in uploaded:
#    filepath='./'+fn
#print(filepath)

up_image = st.file_uploader("Choose an image...", type="png")
if up_image is not None:
    st.write("Image has been loaded. Please wait for the result...")
    # 学習用のデータを作る.
    image_list = []
    label_list = []

    # データが格納されたディレクトリ名
    ## エラーが場合はターミナルで cd と打つ
    # basedir='/User/mizuki/brand-logo-recognirion/images/'

    # ディレクトリ名(ラベル名)指定(3次元)
    label_name=['adidas','nike','puma'] # ,'underarmor'

    # 訓練データのディレクトリ
    # traindir=basedir

    # ./data/train 以下の画像を読み込む。
    ## エラーが出た場合は cd brand-logo-recognition と打つ
    for dir in os.listdir('images'):
        if dir == ".DS_Store":
            continue

        dir1 = 'images/' + dir 
        label = 0

        if dir == label_name[0]:  
            label = 0
        elif dir == label_name[1]:
            label = 1
        elif dir == label_name[2]:
            label = 2
    #    elif dir == label_name[3]:
    #        label = 3

        print("os.listdir" + dir)
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
    print("=========================================")
    print(label_list)
    print("=========================================")
    Y = to_categorical(label_list)
    print("=========================================")
    print(Y)
    print("=========================================")

    # モデルを生成してニューラルネットを構築
    print(1)
    model = Sequential()
    print(2)
    model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3), padding='same', activation='relu')) # うごかない(100,100,3) -> (100,100,4)
    print(3)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    print(4)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(5)

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    print(6)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(7)
    model.add(Dropout(0.2))
    print(8)
    model.add(Flatten())
    print(9)
    model.add(Dense(128, activation='relu'))
    print(10)
    model.add(Dropout(0.2))
    print(11)
    model.add(Dense(128, activation='relu'))
    print(12)
    model.add(Dropout(0.2))
    print(13)
    model.add(Dense(3, activation='softmax')) # 4 -> 3
    print(14)

    model.summary()
    print(15)
    # モデルをコンパイル
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    print(16)
    # 学習を実行。10%はテストに使用。
    #model.fit(image_list, Y, epochs=1000, batch_size=25, callbacks=[early_stopping])
    model.fit(image_list, Y, epochs=7, batch_size=15)
    print(17)

    image = np.array(Image.open(up_image).convert('RGB').resize((100, 100)))
    result = model.predict(np.array([image / 255.]))
    st.markdown("## Result")
    st.write(result)
    # 0がadidas, 1がnike, 2がpuma
    # resultの一番大きい値のインデックスを取得
    result_index = np.argmax(result)
    if result_index == 0:
        st.markdown("# This brand is [Adidas]")
    elif result_index == 1:
        st.markdown("# This brand is [Nike]")
    elif result_index == 2:
        st.markdown("# This brand is [Puma]")


# st.button('Activate the camera')
n = 0
if st.button('Activate the camera'):
    n = 1

if n == 1:
    camera_input = st.camera_input("Camera input...")
    # camera_input = st.image(camera_input)
    if camera_input is not None:
        st.write("Image has been loaded. Please wait for the result...")
        # 学習用のデータを作る.
        image_list = []
        label_list = []

        # データが格納されたディレクトリ名
        ## エラーが場合はターミナルで cd と打つ
        # basedir='/User/mizuki/brand-logo-recognirion/images/'

        # ディレクトリ名(ラベル名)指定(3次元)
        label_name=['adidas','nike','puma'] # ,'underarmor'

        # 訓練データのディレクトリ
        # traindir=basedir

        # ./data/train 以下の画像を読み込む。
        ## エラーが出た場合は cd brand-logo-recognition と打つ
        for dir in os.listdir('images'):
            if dir == ".DS_Store":
                continue

            dir1 = 'images/' + dir 
            label = 0

            if dir == label_name[0]:  
                label = 0
            elif dir == label_name[1]:
                label = 1
            elif dir == label_name[2]:
                label = 2
        #    elif dir == label_name[3]:
        #        label = 3

            print("os.listdir" + dir)
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
        print("=========================================")
        print(label_list)
        print("=========================================")
        Y = to_categorical(label_list)
        print("=========================================")
        print(Y)
        print("=========================================")

        # モデルを生成してニューラルネットを構築
        print(1)
        model = Sequential()
        print(2)
        model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 3), padding='same', activation='relu')) # うごかない(100,100,3) -> (100,100,4)
        print(3)
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        print(4)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print(5)

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        print(6)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print(7)
        model.add(Dropout(0.2))
        print(8)
        model.add(Flatten())
        print(9)
        model.add(Dense(128, activation='relu'))
        print(10)
        model.add(Dropout(0.2))
        print(11)
        model.add(Dense(128, activation='relu'))
        print(12)
        model.add(Dropout(0.2))
        print(13)
        model.add(Dense(3, activation='softmax')) # 4 -> 3
        print(14)

        model.summary()
        print(15)
        # モデルをコンパイル
        model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
        print(16)
        # 学習を実行。10%はテストに使用。
        #model.fit(image_list, Y, epochs=1000, batch_size=25, callbacks=[early_stopping])
        model.fit(image_list, Y, epochs=7, batch_size=15)
        print(17)

        image = np.array(Image.open(camera_input).convert('RGB').resize((100, 100)))
        result = model.predict(np.array([image / 255.]))
        st.markdown("## Result")
        st.write(result)
        # 0がadidas, 1がnike, 2がpuma
        # resultの一番大きい値のインデックスを取得
        result_index = np.argmax(result)
        if result_index == 0:
            st.markdown("# This brand is [Adidas]")
        elif result_index == 1:
            st.markdown("# This brand is [Nike]")
        elif result_index == 2:
            st.markdown("# This brand is [Puma]")


# image = np.array(Image.open('./test/adidas.png').convert('RGB').resize((100, 100)))
# result = model.predict(np.array([image / 255.]))
# st.markdown("## adi")
# st.write(result)
# print("adi")
# print(result)
# # os.remove(filepath)

# image = np.array(Image.open('./test/nikenike123.png').convert('RGB').resize((100, 100)))
# result = model.predict(np.array([image / 255.]))
# st.markdown("## nike")
# st.write(result)

# print("nike")
# print(result)
# # os.remove(filepath)

# image = np.array(Image.open('./test/puma.png').convert('RGB').resize((100, 100)))
# result = model.predict(np.array([image / 255.]))
# st.markdown("## puma")
# st.write(result)
# print("puma")
# print(result)
# # os.remove(filepath)

# # image = np.array(Image.open('./test/underarmour.png').convert('RGB').resize((100, 100)))
# # result = model.predict(np.array([image / 255.]))
# # print("under")
# # print(result)
# # os.remove(filepath)

