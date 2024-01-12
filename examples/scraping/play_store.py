# a short script that retrieves the reviews from a specific app in the Google Play Store given it's URL identifier
import pandas as pd
from utils.scrape import get_playstore_reviews

# For example, the URL identifier for the app Prime Video is 'com.amazon.avod.thirdpartyclient'.
# The full URL is https://play.google.com/store/apps/details?id=com.amazon.avod.thirdpartyclient&hl=de&gl=US
# The identifier is always the part after 'id=' and before the next '&'.

# using get_reviews() to retrieve the reviews for Prime Video
prime = get_playstore_reviews("com.amazon.avod.thirdpartyclient")

# using get_reviews() to retrieve the reviews for IKEA
ikea = get_playstore_reviews("com.ingka.ikea.app")

# using get_reviews() to retrieve english reviews for IKEA
ikea_en = get_playstore_reviews("com.ingka.ikea.app", lang="en", country="us")

# after this the data can be saved to a csv file
prime.to_csv("./data/prime.csv", index=False)
ikea.to_csv("./data/ikea.csv", index=False)
ikea_en.to_csv("./data/ikea_en.csv", index=False)

# or used for descriptive analysis
print(prime.year.value_counts())