# short script to create subsamples for ChatGPT based few-shot prompting
# we refer to https://link.springer.com/article/10.1007/s40547-024-00143-4
# regarding the reasoning for subsampling
import os
import pandas as pd
from utils.create_samples import subsplit

# we read the data from our samples and create subsamples for train and test data
# we mainly have data with 1-3, 1-5 or 1-10 unique classes
# our test set should be aroun 200 (as per hartmann ..)
# we construct zero- and 6-shot classification


automotive = "automotive_test.csv"
drugs = "drugs_test.csv" # 10
hotel = "hotel_test.csv"
ikea_reviews = "ikea_reviews_test.csv"
instant_video = "instant_video_test.csv"
lidl = "lidl_test.csv"
luxury_beauty = "luxury_beauty_test.csv"
musical_instruments = "musical_instruments_test.csv"
office_products = "office_products_test.csv"
patio_lawn_garden = "patio_lawn_garden_test.csv"
reddit = "reddit_test.csv"
twitter = "twitter_test.csv" # 3

subsplit(name=drugs, n_samples=20)
subsplit(name=twitter, n_samples=66)

# remaining splits
datasets = [automotive, hotel, ikea_reviews, instant_video, lidl, luxury_beauty, musical_instruments, office_products, patio_lawn_garden, reddit]
for i in datasets:
    subsplit(name=i, n_samples=40)