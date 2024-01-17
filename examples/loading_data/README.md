# Loading different sources of data

Depending on what data source we have, the way we load it has to be adjusted. Some data is retrieved as Pandas DataFrames (Google Play Store) that we can save as CSV files; others are gz files (Amazon Reviews). This example folder will give examples to working with the different data sources discussed in the paper.

### Google Play Store Data
Data scraped using the [play_store.py](../scraping/play_store.py) script is saved directly as CSV and can be used almost directly. See [here](./load_google.py) or [here](load_google.ipynb) for an interactive session.

### Amazon Review Data
Data from [Ni et al.](https://nijianmo.github.io/amazon/index.html) is served as zipped json files. With Python and gzip we can unpack them and load the json. See [here](./load_amazon.py) or [here](load_amazon.ipynb) for an interactive session.