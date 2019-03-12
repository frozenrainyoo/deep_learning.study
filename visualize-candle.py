import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_finance import candlestick_ohlc
# import matplotlib as mpl then mpl.use('TkAgg')
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('BitMEX-OHLCV-1d.csv')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

chart_figure = plt.figure(figsize=(10, 5))
chart_figure.set_facecolor('w')
chart_gridspec = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
axes = []
axes.append(plt.subplot(chart_gridspec[0]))
axes.append(plt.subplot(chart_gridspec[1], sharex=axes[0]))
axes[0].get_xaxis().set_visible(False)

x = np.arange(len(df.index))
ohlc = df[['open', 'high', 'low', 'close']].astype(int).values
dohlc = np.hstack((np.reshape(x, (-1, 1)), ohlc))
candlestick_ohlc(axes[0], dohlc, width=0.5, colorup='r', colordown='b')

axes[1].bar(x, df.volume, color='k', width=0.6, align='center')

plt.tight_layout()
plt.show()

