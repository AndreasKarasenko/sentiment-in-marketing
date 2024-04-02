# short script to create training and test samples for all datasets
import pandas as pd
from utils.preprocess import reviews, google_play, drugs_data, hotel_data, twitter_data
from utils.create_samples import split

# load the playstore data
ikea = "data/ikea_reviews.csv"
reddit = "data/reddit_reviews.csv"
lidl = "data/lidl_reviews.csv"
# Amazon review datasets
luxury_beauty = "data/Luxury_Beauty_5.json.gz"
instant_video = "data/reviews_Amazon_Instant_Video_5.json.gz"
automotive = "data/reviews_Automotive_5.json.gz"
musical_instruments = "data/reviews_Musical_Instruments_5.json.gz"
office_products = "data/reviews_Office_Products_5.json.gz"
patio_lawn_garden = "data/reviews_Patio_Lawn_and_Garden_5.json.gz"
# Drugs review dataset
drugs = "data/drugs.zip"
# Twitter dataset
twitter = "data/twitter.zip"
# Hotel review dataset
hotel = "data/hotel.zip"

# create the dataframes
df = google_play(ikea)
split(df, "ikea_reviews")

df = google_play(reddit)
split(df, "reddit")

df = google_play(lidl)
split(df, "lidl")


df = reviews(luxury_beauty)
split(df, "luxury_beauty")

df = reviews(instant_video)
split(df, "instant_video")

df = reviews(automotive)
split(df, "automotive")

df = reviews(musical_instruments)
split(df, "musical_instruments")

df = reviews(office_products)
split(df, "office_products")

df = reviews(patio_lawn_garden)
split(df, "patio_lawn_garden")

df = drugs_data(drugs)
split(df, "drugs")

df = hotel_data(hotel)
split(df, "hotel")

df = twitter_data(twitter)
split(df, "twitter")