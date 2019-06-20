import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


point_num = 200*6

x = np.arange(0, point_num, 1)
y5 = np.sin(np.pi/5 * x) / 8
y25 = np.sin(np.pi/25 * x) / 4
y75 = np.sin(np.pi/75 * x) / 2
y200 = np.sin(np.pi/200 * x)

sinW = y5 + y25 + y75 + y200

plt.plot(x, sinW)
plt.show()
plt.savefig("./sinW.png")

# white noise
sinN = y5 + y25 + y75 + y200 + (1/8*np.random.randn(point_num))

plt.plot(x, sinN)
plt.show()
plt.savefig("./sinN.png")

# save csv
sinData = pd.DataFrame({'sin':sinW,
                        'noise':sinN})
sinData.set_index(x, inplace=True)
sinData.to_csv("./sin_wave.csv")

