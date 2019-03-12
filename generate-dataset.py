# import ccxt.async_support as ccxt
# generate dataset for bitcoin

import ccxt
import time
import pandas as pd

print('generate dataset')

exchange = ccxt.bitmex({'verbose': True,
                        'enableRateLimit': True,})

symbol = 'BTC/USD'
filename = exchange.name + '-OHLCV-1d.csv'

if exchange.has['fetchOHLCV']:
    time.sleep (exchange.rateLimit / 1000)
    ohlcvs = exchange.fetch_ohlcv (symbol, '1d') # one day
    dataframe = pd.DataFrame(ohlcvs)
    dataframe.to_csv(filename, header=False, index=False)

print('complete generating ohlcv')


# pandas good

# import csv
# f = open('output.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# wr.writerow([1, "김정수", False])
# wr.writerow([2, "박상미", True])
# f.close()
# f = open('data.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# for line in rdr:
#     print(line)
# f.close()