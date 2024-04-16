import pandas as pd
from google_play_scraper import Sort, reviews_all


def get_playstore_reviews(
    app: str, lang: str = "de", country: str = "de"
) -> pd.DataFrame:
    """
    Fetches all reviews for a specific app from the Google Play Store and returns them as a pandas DataFrame.

    Parameters:
    app (str): The ID of the app for which to fetch the reviews. E.g. "com.ingka.ikea.app"
    lang (str, optional): The language in which to fetch the reviews. Defaults to 'de'.
    country (str, optional): The country for which to fetch the reviews. Defaults to 'de'.

    Returns:
    df: A DataFrame containing all the reviews.
    """
    result = reviews_all(
        app_id=app,
        sleep_milliseconds=0,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
    )

    df = pd.DataFrame(result)
    df = df.rename(columns={"at": "date"})  # at is a pandas argument
    df["year"] = df.date.apply(lambda x: x.year)
    return df
