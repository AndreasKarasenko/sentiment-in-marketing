import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_imdb_reviews(movie_id: str, max_iter: int = 1000):
    """
    This function scrapes reviews of a specific movie from IMDB.

    Parameters:
    movie_id (str): The unique identifier of the movie on IMDB.
    max_iter (int): The maximum number of pages to scrape. Default is 1000.

    Returns:
    pandas.DataFrame: A DataFrame containing the title, review, and rating of each review.
    """
    assert max_iter > 0, "max_iter must be greater than 0"
    url = (
        "https://www.imdb.com/title/"
        + movie_id
        + "/reviews/_ajax?ref_=undefined&paginationKey={}"
    )
    data = {"title": [], "review": [], "rating": []}
    key = ""  # we need to pass it but not get a new key while iterating

    n = 0
    while n < max_iter:
        response = requests.get(url.format(key))
        soup = BeautifulSoup(response.content, "html.parser")
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
