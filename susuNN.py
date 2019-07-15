import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


##--------------------------
# データの用意
##--------------------------
# データファイル読み込み
DATA_FILENAME = './sinwave.csv'
DATA_COLUMN_NAME = 'sin'
#DATA_COLUMN_NAME = 'noise'

data = pd.read_csv(DATA_FILENAME, usecols=[DATA_COLUMN_NAME], squeeze=True)
data = data.values.tolist()

# 予測に使用するデータ数
INPUT_DATA_NUM = 5

# データセットの生成
inpData = []
ansData = []
for i in range(0, len(data) - INPUT_DATA_NUM):
    inpData.append(data[i:i + INPUT_DATA_NUM])
    ansData.append([data[i + INPUT_DATA_NUM]])

# 訓練データと検証データに分割
TRAIN_DATA_NUM = 1000

trainInput = inpData[:TRAIN_DATA_NUM]
testInput = inpData[TRAIN_DATA_NUM:]
trainAns = ansData[:TRAIN_DATA_NUM]
testAns = ansData[TRAIN_DATA_NUM:]


##--------------------------
# NN構築
##--------------------------
INPUT_LAYER_NODE_NUM = INPUT_DATA_NUM
HIDDEN_LAYER_NODE_NUM = 3
OUTPUT_LAYER_NODE_NUM = 1
TRAIN_RATE = 0.2
TRAIN_NUM = 1000


# 入出力ホルダー
inpHld = tf.placeholder(tf.float32, [None, INPUT_LAYER_NODE_NUM])
ansHld = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_NODE_NUM])

# 入力層→隠れ層
w1 = tf.Variable(tf.random_uniform([INPUT_LAYER_NODE_NUM, HIDDEN_LAYER_NODE_NUM], -1, 1))
b1 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NODE_NUM], -1, 1))
o1 = tf.nn.sigmoid(tf.matmul(inpHld, w1) + b1)

# 隠れ層→出力層
w2 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NODE_NUM, OUTPUT_LAYER_NODE_NUM], -1, 1))
b2 = tf.Variable(tf.random_uniform([OUTPUT_LAYER_NODE_NUM], -1, 1))
output = tf.nn.sigmoid(tf.matmul(o1, w2) + b2)

# 損失関数
loss = tf.reduce_mean(tf.squared_difference(ansHld, output))

# 最適化
train = tf.train.AdamOptimizer(TRAIN_RATE).minimize(loss)


##--------------------------
# 学習実行
##--------------------------
with tf.Session() as sess:
    # 初期化
    sess.run(tf.global_variables_initializer())

    # 学習実行
    for i in range(TRAIN_NUM):
        dummy_, loss_ = sess.run([train, loss],
                            feed_dict = {inpHld: trainInput, ansHld: trainAns})
        
        # 定期的に損失関数の値を出力
        if i % 100 == 0:
            print(str(i+1), ":", loss_)


##--------------------------
# 出力
##--------------------------
    # 学習時性能出力
    out, loss_ = sess.run([output, loss], feed_dict={inpHld: trainInput, ansHld: trainAns})
    print("train loss:", loss_)

    plt.plot(np.reshape(trainAns, (-1)))
    plt.plot(np.reshape(out, (-1)))
    plt.savefig("./nn("+DATA_COLUMN_NAME+")-d"+str(INPUT_DATA_NUM)+"t"+str(TRAIN_NUM)+"r"+str(TRAIN_RATE)+"gaku.png")
    plt.show()

    # 汎化性能出力
    out, loss_ = sess.run([output, loss], feed_dict={inpHld: testInput, ansHld: testAns})
    print("test loss :", loss_)

    plt.plot(np.reshape(testAns, (-1)))
    plt.plot(np.reshape(out, (-1)))
    plt.savefig("./nn("+DATA_COLUMN_NAME+")-d"+str(INPUT_DATA_NUM)+"t"+str(TRAIN_NUM)+"r"+str(TRAIN_RATE)+"test.png")
    plt.show()