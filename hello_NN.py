import tensorflow as tf
import pandas as pd

# sin wave data読み込み

# data正規化

# 訓練データ/検証データの分割

## NNの構築
INPUT_LAYER_NUM = 5
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
        dummy_, loss_, ans_ = sess.run([train, loss, oData],
                                        feed_dict = {iData: iDatas, oData: oDatas})
        print(i, loss)


