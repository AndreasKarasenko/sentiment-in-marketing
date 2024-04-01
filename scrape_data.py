from tqdm import tqdm
import pandas as pd
from google_play_scraper import Sort, reviews
# from utils.scrape import get_playstore_reviews

lang = "en",
country = "us"
reviews_count = 34000

# scrape ikea data
result = []
continuation_token = None
app = "com.ingka.ikea.app"

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
        result.extend(new_result)
        pbar.update(len(new_result))

df = pd.DataFrame(result)
df.to_csv("data/ikea_reviews.csv", index=False)

# scrape reddit data
app = "com.reddit.frontpage"
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
        result.extend(new_result)
        pbar.update(len(new_result))

df = pd.DataFrame(result)

df.to_csv("data/reddit_reviews.csv", index=False)

# scrape lidl plus data
app = "com.lidl.eci.lidlplus"
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
        result.extend(new_result)
        pbar.update(len(new_result))

df = pd.DataFrame(result)

df.to_csv("data/lidl_reviews.csv", index=False)