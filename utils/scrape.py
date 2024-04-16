import pandas as pd
import requests
from bs4 import BeautifulSoup
from google_play_scraper import Sort, reviews
from tqdm import tqdm

# TODO check for yourself if the bug has been fixed
# https://github.com/JoMingyu/google-play-scraper/issues/209
# def get_playstore_reviews(
#     app: str, lang: str = "de", country: str = "de"
# ) -> pd.DataFrame:
#     """
#     Fetches all reviews for a specific app from the Google Play Store and returns them as a pandas DataFrame.

#     Parameters:
#     app (str): The ID of the app for which to fetch the reviews.
#     lang (str, optional): The language in which to fetch the reviews. Defaults to 'de'.
#     country (str, optional): The country for which to fetch the reviews. Defaults to 'de'.

#     Returns:
#     df: A DataFrame containing all the reviews.
#     """
#     result = reviews_all(
#         app_id=app,
#         sleep_milliseconds=0,
#         lang=lang,
#         country=country,
#         sort=Sort.NEWEST,
#     )

#     df = pd.DataFrame(result)
#     df = df.rename(columns={"at": "date"})  # at is a pandas argument
#     df["year"] = df.date.apply(lambda x: x.year)
#     return df

# current fix for google play:


def get_playstore_reviews(
    app: str, lang: str = "de", country: str = "de", reviews_count: int = 25000
) -> pd.DataFrame:
    """
    Fetches all reviews for a specific app from the Google Play Store and returns them as a pandas DataFrame.

    Parameters:
    app (str): The ID of the app for which to fetch the reviews.
    lang (str, optional): The language in which to fetch the reviews. Defaults to 'de'.
    country (str, optional): The country for which to fetch the reviews. Defaults to 'de'.

    Returns:
    df: A DataFrame containing all the reviews.
    """
    result = []
    continuation_token = None

    with tqdm(total=reviews_count, position=0, leave=True) as pbar:
        while len(result) < reviews_count:
            new_result, continuation_token = reviews(
                app,
                continuation_token=continuation_token,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                filter_score_with=None,
                count=150,
            )
            if not new_result:
                df = pd.DataFrame(result)
                return df
            result.extend(new_result)
            pbar.update(len(new_result))

    df = pd.DataFrame(result)
    return df


# Source: https://medium.com/@wujingyi/web-scraping-movie-reviews-from-imdb-using-beautiful-soup-134dc562b1e6
def get_imdb_reviews(movie_id: str, max_iter: int = 1000):
    """
    This function scrapes reviews of a specific movie from IMDB.

    Parameters:
    movie_id (str): The unique identifier of the movie on IMDB.
    max_iter (int): The maximum number of pages to scrape. Default is 1000.

    Returns:
    pandas.DataFrame: A DataFrame containing the title, review, and rating of each review.

    Raises:
    AssertionError: If max_iter is not greater than 0.
    """
    # There is a difference between the _ajax and the normal version of the URL.
    # Compare https://www.imdb.com/title/tt0111161/reviews/ and https://www.imdb.com/title/tt0111161/reviews/_ajax
    # the title id comes immeadiately after the /title/ part of the URL
    # for Dune (2021) the id is tt1160419
    # all title IDs could be scraped by dissecting the URL of (e.g.) https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm&sort=num_votes%2Cdesc
    # this site lists the most reviewed movies on IMDB
    # all title ids are listed in one line in the HTML code of the site which can be found in a script tag with id=__NEXT_DATA__
    assert max_iter > 0, "max_iter must be greater than 0"
    url = (
        "https://www.imdb.com/title/"
        + movie_id
        + "/reviews/_ajax?ref_=undefined&paginationKey={}"
    )
    data = {"title": [], "review": [], "rating": []}
    key = ""
    # an example pagination key looks like this
    # g4w6ddbmqyzdo6ic4oxwjnrxqlsm2cj63apt36xka3cpsw35pjt6udkyoq4vzprpb4dtn7jevbep3qg2zj4oine4vcbjg
    # and is found as an element in the site's HTML code

    n = 0
    while n < max_iter:
        response = requests.get(url.format(key))
        soup = BeautifulSoup(response.content, "html.parser")
        # key = soup.find("div", class_="load-more-data") # adding this breaks getting more data
        for title, review, rating in zip(
            soup.find_all(class_="title"),
            soup.find_all(class_="text show-more__control"),
            soup.find_all(class_="rating-other-user-rating"),
        ):
            data["title"].append(title.get_text(strip=True))
            data["review"].append(review.get_text())
            data["rating"].append(rating.get_text(strip=True) if rating else "N/A")

        n += 1

    return pd.DataFrame(data)
