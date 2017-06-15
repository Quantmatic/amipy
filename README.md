# amipy
very fast python backtesting framework based on amibroker backtesting methodology

- event driven
- supports any timeframe
- supports tick aggregation
- fast optimization speeds
- multi-asset class simulations

compatible with IQFeed data and MongoDB,  (e.g. [iq2mongo](https://github.com/Quantmatic/iq2mongo))

also compatible with any other data source, so long as the OHLC dataframe has the following column format:

ohlc = data[:][['open', 'high', 'low', 'close', 'volume']]

view [sample strategy results](https://nbviewer.jupyter.org/github/Quantmatic/amipy/blob/master/examples/BollingerCMF.html) via nbviewer
