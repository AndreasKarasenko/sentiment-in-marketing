import gzip # for reading the .json.gz file
import json # for extracting the json from string
import pandas as pd # to get the data into a DataFrame
# read zipfile
import zipfile

def reviews(data: str):
    """
    preprocesses the amazon product review data
    
    Parameters:
    data (.json.gz): path to the raw data file
    
    Returns:
    df (pd.DataFrame): the preprocessed data
    """
    
    f = gzip.open(data, 'rt', encoding='utf-8')
    file = f.readlines() # list of strings containing json
    data = [json.loads(i) for i in file] # list of dict objects
    df = pd.DataFrame(data)
    
    df = df.loc[:,["overall", "reviewText", "unixReviewTime"]]
    
    df["date"] = pd.to_datetime(df["unixReviewTime"], unit='s')
    df.rename(columns={"overall": "rating", "reviewText": "review"}, inplace=True)
    
    return df

def google_play(data: str):
    """
    preprocesses the google play store review data
    
    Parameters:
    data (.csv): path to the raw data file
    
    Returns:
    df (pd.DataFrame): the preprocessed data
    """
    
    df = pd.read_csv(data)
    
    df = df.loc[:,["score", "content", "at"]]
    
    # only keep year-month-day
    df["at"] = pd.to_datetime(df["at"]).dt.date
    df.rename(columns={"content": "review", "score": "rating", "at": "date"}, inplace=True)
    
    return df

def drugs_data(data: str):
    """
    preprocesses the drugs.com review data
    
    Parameters:
    data (.zip): path to the raw data file
    
    Returns:
    df (pd.DataFrame): the preprocessed data
    """
    df = "drugsComTest_raw.csv"
    file = zipfile.ZipFile(data, 'r')
    df = pd.read_csv(file.open(df))
    
    
    df = df.loc[:,["rating", "review", "date"]]
    
    df["date"] = pd.to_datetime(df["date"])
    
    return df

def hotel_data(data: str):
    """
    preprocesses the hotel review data
    
    Parameters:
    data (.zip): path to the raw data file
    
    Returns:
    df (pd.DataFrame): the preprocessed data
    """
    
    df = "Datafiniti_Hotel_Reviews.csv"
    file = zipfile.ZipFile(data, 'r')
    df = pd.read_csv(file.open(df))
    
    
    df = df.loc[:,["reviews.rating", "reviews.text", "reviews.date"]]
    df.columns = ["rating", "review", "date"]
    
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.date
    
    return df

def twitter_data(data: str):
    """
    preprocesses the twitter review data
    
    Parameters:
    data (.zip): path to the raw data file
    
    Returns:
    df (pd.DataFrame): the preprocessed data
    """
    
    df = "Tweets.csv"
    file = zipfile.ZipFile(data, 'r')
    df = pd.read_csv(file.open(df))
    
    
    df = df.loc[:,["airline_sentiment", "text", "tweet_created"]]
    df.columns = ["rating", "review", "date"]
    
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.date
    df.rating = df.rating.map({"positive": 3, "neutral": 2, "negative": 1})
    
    return df