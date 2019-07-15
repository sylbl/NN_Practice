import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
#from sklearn import preprocessing

# 生成データ数
MAKE_DATA_NUM = 200*6

# 合成sin波
x = np.arange(0, MAKE_DATA_NUM, 1)
y5 = np.sin(np.pi/5 * x) / 8
y25 = np.sin(np.pi/25 * x) / 4
y75 = np.sin(np.pi/75 * x) / 2
y200 = np.sin(np.pi/200 * x)

sinW = y5 + y25 + y75 + y200
# 正規化
sinW = prep.minmax_scale(sinW)

# 合成sin波表示
plt.plot(x, sinW)
plt.savefig("./sinWave.png")
plt.show()


# ホワイトノイズの付加
sinN = y5 + y25 + y75 + y200 + np.random.randn(MAKE_DATA_NUM) / 8
# 正規化
sinN = prep.minmax_scale(sinN)

# 合成sin波表示
plt.plot(x, sinN)
plt.savefig("./sinNoise.png")
plt.show()


# save csv
sinData = pd.DataFrame({'sin':sinW, 'noise':sinN})
print(sinData)
sinData.to_csv("./sinwave.csv", index=False)

