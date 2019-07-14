import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


##--------------------------
# データの用意
##--------------------------
# データファイル読み込み
DATA_FILENAME = './sinwave.csv'
DATA_COLUMN_NAME = 'sin'
data = pd.read_csv(DATA_FILENAME, usecols=[DATA_COLUMN_NAME])


# 予測に使用するデータ数
INPUT_DATA_NUM = 5

# データセットの生成
inpData = []
ansData = []
for i in range(0, len(data) - INPUT_DATA_NUM):
    print(data[i:i + INPUT_DATA_NUM].values)
    #print(data[i])

print(inpData)

# 訓練データと検証データに分割
TRAIN_DATA_NUM = 1000
trainData = data[:TRAIN_DATA_NUM - 1]
testData = data[TRAIN_DATA_NUM:]



# データセットの生成

##--------------------------
# NN構築
##--------------------------
INPUT_LAYER_NODE_NUM = INPUT_DATA_NUM
HIDDEN_LAYER_NODE_NUM = 6
OUTPUT_LAYER_NODE_NUM = 1
TRAIN_RATE = 0.002
TRAIN_NUM = 5000

# 入出力ホルダー
inpData = tf.placeholder(tf.float32, [None, INPUT_LAYER_NODE_NUM])
ansData = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_NODE_NUM])

# 入力層→隠れ層
w1 = tf.Variable(tf.random_uniform([INPUT_LAYER_NODE_NUM, HIDDEN_LAYER_NODE_NUM], -1, 1))
b1 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NODE_NUM], -1, 1))
o1 = tf.nn.sigmoid(tf.matmul(inpData, w1) + b1)

# 隠れ層→出力層
w2 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NODE_NUM, OUTPUT_LAYER_NODE_NUM], -1, 1))
b2 = tf.Variable(tf.random_uniform([OUTPUT_LAYER_NODE_NUM], -1, 1))
output = tf.nn.sigmoid(tf.matmul(o1, w2) + b2)

# 損失関数
loss = tf.reduce_mean(tf.squared_difference(ansData, output))

# 最適化
train = tf.train.AdamOptimizer(TRAIN_RATE).minimize(loss)


##--------------------------
# 学習実行
##--------------------------
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
    plt.savefig("./nn(n)-d"+str(INPUT_DAYS)+"t"+str(TRAIN_NUM)+"r"+str(TRAIN_RATE)+"gaku.png")
    plt.show()

    out, loss_ = sess.run([output, loss], feed_dict={inpData: ciData, ansData: caData})
    print(loss_)

    plt.plot(np.reshape(caData, (1,-1))[0])
    plt.plot(np.reshape(out, (1,-1))[0])
    plt.savefig("./nn(n)-d"+str(INPUT_DAYS)+"t"+str(TRAIN_NUM)+"r"+str(TRAIN_RATE)+"chk.png")
    plt.show()