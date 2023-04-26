import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objects as go
import plotly as py
import plotly.io as pio
pio.renderers.default = 'chrome'

df = pd.read_pickle('df.pickle')
indices = pd.read_pickle('indices.pickle')
raw = pd.read_pickle('rawData.pickle')
cosineSim = pd.read_pickle('cosineSim.pickle')
reviews = pd.read_pickle('reviews.pickle')

name = st.sidebar.text_input(''' Enter your user name''')
currentUser = reviews[(reviews["User_Name"] == name) & (reviews["Polarity"] == "Positive")].reset_index(drop=True)
st.sidebar.table(currentUser["Recipe"])

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

fig = px.bar(recommendedSorted, x='Name', y='Rating', color='Name', color_discrete_sequence=px.colors.diverging.Geyser, height=600, width=900)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.update_layout(template="plotly_white", xaxis_showgrid=False, yaxis_showgrid=False)

fig.update_traces(marker_line_color='rgb(8,48,107)',
                  marker_line_width=2, opacity=0.6)

fig.update_layout(showlegend=False, title="Rating",
                  xaxis_title="Recommended Recipes",
                  yaxis_title="Rate")

fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

st.dataframe(recommendedSorted)

st.plotly_chart(fig)
