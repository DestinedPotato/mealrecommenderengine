from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import streamlit as st

df = pd.read_pickle('df.pickle')
indices = pd.read_pickle('indices.pickle')
raw = pd.read_pickle('rawData.pickle')
reviews = pd.read_pickle('reviews.pickle')

tfidf = TfidfVectorizer(stop_words='english')
tfidfMatrix = tfidf.fit_transform(df['Infos'])
cosineSim = linear_kernel(tfidfMatrix, tfidfMatrix)

st.title("Meals Recommended for you!")
name = st.sidebar.text_input("Enter your user name")
user = reviews[(reviews["User_Name"] == name) & (reviews["Polarity"] == "Positive")].reset_index(drop=True)
st.sidebar.table(user["Recipe"])
count = 0
response = []
for i in user["Recipe"]:
    response.append(st.sidebar.slider('Rating out of 5', min_value=1,max_value=5, key=i, value=3))

def get_recommendations(name, cosineSim, raw):
    index = indices[name]
    simScores = list(enumerate(cosineSim[index]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
    simScores = simScores[1:11]
    foodIndices = [i[0] for i in simScores]
    recommend = pd.DataFrame(df['Name'].iloc[foodIndices]).reset_index(drop=True)
    d = pd.merge(recommend, raw, on=None, left_on="Name", right_on="Name", how="left")
    return d.drop(columns=["Description", "Ingredients", "Preparation"])


def user(userName, cosineSim, raw):
    userInfo = reviews[(reviews["User_Name"] == userName) & (reviews["Polarity"] == "Positive")]

    if len(userInfo) >= 1:
        food = (userInfo.sample(1)).iloc[0, 1]
        return get_recommendations(food, cosineSim, raw)
    else:
        food = (df.sample(1)).iloc[0, 1]
        return get_recommendations(food, cosineSim, raw)


recommended = user(name, cosineSim, raw)
recommended.sort_values("Rating", ascending=False, inplace=True)
recommended = recommended.reset_index(drop=True)

recommendedSorted = recommended.sort_values("Rating", ascending=False)

st.table(recommendedSorted)
