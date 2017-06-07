# amipy
very fast python backtesting framework based on amibroker backtesting methodology

compatible with IQFeed data and MongoDB,  (e.g. [iq2mongo](https://github.com/Quantmatic/iq2mongo))

also compatible with any other data source, so long as the OHLC dataframe has the following column format:

ohlc = data[:][['open', 'high', 'low', 'close', 'volume']]

view [sample strategy results](https://nbviewer.jupyter.org/github/Quantmatic/amipy/blob/master/BollingerCMF.html) via nbviewer

<script src="//repl.it/embed/IbUQ/0.js"></script>
