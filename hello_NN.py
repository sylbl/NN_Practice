import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# sin wave data読み込み
sinWaveData = pd.read_csv("./sin_wave.csv")
sinWaveData = sinWaveData.drop("Unnamed: 0", axis=1)
print(sinWaveData)
print(sinWaveData.max())

# 正解データ用意
sinWaveData['ans'] = sinWaveData['sin'].shift(-1)
print(sinWaveData)

# data正規化
scaler = MinMaxScaler()
scaler.fit(sinWaveData)
sinWaveDataNorm = scaler.transform(sinWaveData)

# 訓練データ/検証データの分割
trainData = sinWaveDataNorm[:1000]
checkData = sinWaveDataNorm[1001:-1]

# 訓練データの生成
INPUT_DAYS = 5
iData = []
aData = []
for i in range(INPUT_DAYS-1, len(trainData)):
    inpSet = []
    for j in range(i-INPUT_DAYS+1, i+1):
        inpSet.append(trainData[j][0])

    iData.append(inpSet)
    aData.append([trainData[i][2]])
iData = np.array(iData)
aData = np.array(aData)


# 検証データの生成
ciData = []
caData = []
for i in range(INPUT_DAYS-1, len(checkData)):
    inpSet = []
    for j in range(i-INPUT_DAYS+1, i+1):
        inpSet.append(checkData[j][0])

    ciData.append(inpSet)
    caData.append([checkData[i][2]])
ciData = np.array(ciData)
caData = np.array(caData)

## NNの構築
INPUT_LAYER_NUM = INPUT_DAYS
HIDDEN_LAYER_NUM = 3
OUTPUT_LAYER_NUM = 1
TRAIN_RATE = 0.2
TRAIN_NUM = 1000

#入出力ホルダー
inpData = tf.placeholder(tf.float32, [None, INPUT_LAYER_NUM])
ansData = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_NUM])

# 入力→隠れ層
w1 = tf.Variable(tf.random_uniform([INPUT_LAYER_NUM, HIDDEN_LAYER_NUM], -1, 1))
b1 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NUM], -1, 1))
o1 = tf.nn.sigmoid(tf.matmul(inpData, w1) + b1)

# 隠れ→出力層
w2 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NUM, OUTPUT_LAYER_NUM], -1, 1))
b2 = tf.Variable(tf.random_uniform([OUTPUT_LAYER_NUM], -1, 1))
output = tf.nn.sigmoid(tf.matmul(o1, w2) + b2)

# 損失関数
loss = tf.reduce_mean(tf.squared_difference(ansData, output))

# 最適化
train = tf.train.AdamOptimizer(TRAIN_RATE).minimize(loss)

# 実行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(TRAIN_NUM):
        dummy_, loss_= sess.run([train, loss],
                            feed_dict = {inpData: iData, ansData: aData})
        
        if i % 100 == 0:
            print(i, ":", loss_)

    out, loss_ = sess.run([output, loss], feed_dict={inpData: iData, ansData: aData})
    print(loss_)

    plt.plot(np.reshape(aData, (1,-1))[0])
    plt.plot(np.reshape(out, (1,-1))[0])
    plt.savefig("./nn-d"+str(INPUT_DAYS)+"t"+str(TRAIN_NUM)+"r"+str(TRAIN_RATE)+"gaku.png")
    plt.show()

    out, loss_ = sess.run([output, loss], feed_dict={inpData: ciData, ansData: caData})
    print(loss_)

    plt.plot(np.reshape(caData, (1,-1))[0])
    plt.plot(np.reshape(out, (1,-1))[0])
    plt.savefig("./nn-d"+str(INPUT_DAYS)+"t"+str(TRAIN_NUM)+"r"+str(TRAIN_RATE)+"chk.png")
    plt.show()

    



