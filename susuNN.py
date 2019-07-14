import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


##--------------------------
# データの用意
##--------------------------
DATA_FILENAME = 'sinwave.csv'

# データファイル読み込み

# 学習データと検証データに分割

# 予測に使用するデータ数
INPUT_DATA_NUM = 5

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

