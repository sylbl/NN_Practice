import tensorflow as tf
import numpy as np

# 学習データ用意
inData = []
ansData = []
for i in range(10000):
    dataSet = [(np.random.randint(0,10))/10 for i in range(5)]
    inData.append(dataSet)
    ansData.append([(dataSet[0]*10000 + dataSet[1]*1000 + dataSet[2]*100 + dataSet[3]*10 + dataSet[4])/10000])


## NNの構築
INPUT_LAYER_NUM = 5
HIDDEN_LAYER_NUM = 6
OUTPUT_LAYER_NUM = 1
TRAIN_RATE = 0.001
TRAIN_NUM = 100000

# 入出力ホルダー
iData = tf.placeholder(tf.float32, [None, INPUT_LAYER_NUM])
aData = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_NUM])

# 入力→隠れ層
w1 = tf.Variable(tf.random_uniform([INPUT_LAYER_NUM, HIDDEN_LAYER_NUM], -1, 1))
b1 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NUM], -1, 1))
o1 = tf.nn.sigmoid(tf.matmul(iData, w1) + b1)

# 隠れ→出力層
w2 = tf.Variable(tf.random_uniform([HIDDEN_LAYER_NUM, OUTPUT_LAYER_NUM], -1, 1))
b2 = tf.Variable(tf.random_uniform([OUTPUT_LAYER_NUM], -1, 1))
o2 = tf.nn.sigmoid(tf.matmul(o1, w2) + b2)

# 損失関数
loss = tf.reduce_mean(tf.squared_difference(aData, o2))

# 最適化
train = tf.train.AdamOptimizer(TRAIN_RATE).minimize(loss)


# 実行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAIN_NUM):

        dummy_, loss_ = sess.run([train, loss],
                                feed_dict = {iData: inData, aData: ansData})
        
        if i % 100 == 0:
            print(i, ":", loss_)
    
    print("end train")


    inData = []
    for i in range(10):
        dataSet = [(np.random.randint(0,10))/10 for i in range(5)]
        inData.append(dataSet)

    out = np.hstack((inData, sess.run(o2,feed_dict = {iData: inData})))

    print(out)




