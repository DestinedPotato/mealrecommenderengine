from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import streamlit as st

df = pd.read_pickle('df.pickle')
indices = pd.read_pickle('indices.pickle')
raw = pd.read_pickle('rawData.pickle')
responseDf = pd.read_pickle('response.pickle')

tfidf = TfidfVectorizer(stop_words='english')
tfidfMatrix = tfidf.fit_transform(df['Infos'])
cosineSim = linear_kernel(tfidfMatrix, tfidfMatrix)

st.title("Meals Recommended for you!")


def get_recommendations(food, cosineSim, raw):
    index = indices[food]
    simScores = list(enumerate(cosineSim[index]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
    simScores = simScores[1:11]
    foodIndices = [i[0] for i in simScores]
    recommend = pd.DataFrame(df['Name'].iloc[foodIndices]).reset_index(drop=True)
    d = pd.merge(recommend, raw, on=None, left_on="Name", right_on="Name", how="left")
    return d.drop(columns=["Description", "Ingredients", "Preparation"])


def user(cosineSim, raw):
    food = (df.sample(1)).iloc[0, 1]
    return get_recommendations(food, cosineSim, raw)


if st.button(label='Generate meals', type='primary', key='Generate meals'):
    recommended = user(cosineSim, raw)
    recommended.sort_values("Rating", ascending=False, inplace=True)
    recommended = recommended.reset_index(drop=True)
    recommendedSorted = recommended.sort_values("Rating", ascending=False)
    st.table(recommendedSorted)

response = []
with st.form("Response form"):
    with st.sidebar:
        name = st.text_input("Enter your user name")
        response.append(name)
        for i in range(10):
            response.append(st.slider(label="Rate meal " + str(i+1) + " out of 5", min_value=1, max_value=5, key=i, value=3))
        st.form_submit_button("Submit")
responseDf = pd.DataFrame(data=response)

responseDf.to_pickle("response.pickle")
