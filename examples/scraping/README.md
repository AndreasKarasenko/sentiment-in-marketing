# Scraping

(Web) scraping extracts information from websites that is 
is a method used to extract data from websites. This is done by making a request to the server that hosts the site, which returns the HTML of the webpage. The HTML can then be parsed and the data extracted.

## How It Works

1. **Request Data:** Send a request for specific content using it's URL.

2. **Retrieve Data** The data is oftentimes nested in HTML-Tags as plain text. We can search for the respective tag, iterate over them and retrieve the desired information.

## Libraries and Tools

Python offers a few open source libraries to scrape websites.

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 

- [Scrapy](https://scrapy.org/)

- [Selenium](https://www.selenium.dev/)

Additionally popular sources, such as the Google Play Store, often have pre-built scraper libraries (e.g. [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)).

## Legal Issues

It should be noted, that web scraping is not always allowed. See for examples IMDB's [statement](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX?ref_=helpart_nav_18#). Nonetheless researchers and practitioners oftentimes use data from such sources regardless.

## Notes

As with anything that relies on a certain structure, the functionality is dependent on the sites staying static, and not flagging the user as a bot. If something does not work, or the data is not in a format that you expect, ask Google for up-to-date tutorials. Similar problems arise with API's.

## Examples

[play_store.py](play_store.py) shows how to use google_play_scraper and a simple wrapper function (that can be adjusted or substituted!) to retrieve reviews of products from the Google Play Store. You can adjust the language to retrieve and country. More options exist, but we found these to be sufficient for broad retrieval.

[imdb.py](imdb.py) provides an example function to scrape imdb reviews for specific movies.