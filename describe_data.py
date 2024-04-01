import pandas as pd
from utils.describe import describe
from utils.preprocess import reviews, google_play, drugs_data, hotel_data, twitter_data

# load the playstore data
ikea = "data/ikea_reviews.csv"
reddit = "data/reddit_reviews.csv"
lidl = "data/lidl_reviews.csv"

# create the dataframes
df = google_play(ikea)
describe(df)

df = google_play(reddit)
describe(df)

df = google_play(lidl)
describe(df)

# Amazon review datasets
luxury_beauty = "data/Luxury_Beauty_5.json.gz"
instant_video = "data/reviews_Amazon_Instant_Video_5.json.gz"
automotive = "data/reviews_Automotive_5.json.gz"
musical_instruments = "data/reviews_Musical_Instruments_5.json.gz"
office_products = "data/reviews_Office_Products_5.json.gz"
patio_lawn_garden = "data/reviews_Patio_Lawn_and_Garden_5.json.gz"

df = reviews(luxury_beauty)
describe(df)

df = reviews(instant_video)
describe(df)

df = reviews(automotive)
describe(df)

df = reviews(musical_instruments)
describe(df)

df = reviews(office_products)
describe(df)

df = reviews(patio_lawn_garden)
describe(df)

# Drugs review dataset
drugs = "data/drugs.zip"

df = drugs_data(drugs)
describe(df)

# Hotel review dataset
hotel = "data/hotel.zip"

df = hotel_data(hotel)
describe(df)

# Twitter dataset
twitter = "data/twitter.zip"

df = twitter_data(twitter)
describe(df)