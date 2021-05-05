import plotly.express as px
import streamlit as st
import pandas as pd
import joblib

'''
# Our Last Morning Kick-Off :sob: :sob: :sob: :sob: :sob:

You can write in Markdown *just by typing like this!*

For all your Streamlit questions, check out their [documentation](https://docs.streamlit.io/en/stable/api.html)
'''



'''
## Plot COVID-19 Cases visualization :mask::mask::mask:
'''

#Covid Cases Visualization 
covid = pd.read_csv('https://raw.githubusercontent.com/shinokada/covid-19-stats/master/data/daily-new-confirmed-cases-of-covid-19-tests-per-case.csv')
covid.columns = ['Country', 'Code', 'Date', 'Confirmed', 'Days since confirmed']
covid['Date'] = pd.to_datetime(covid['Date']).dt.strftime('%Y-%m-%d')

st.write(covid)

country_options = covid['Country'].unique().tolist()
date_options = covid['Date'].unique().tolist()

# Select box for date
date = st.selectbox('Which date would you like to see?', date_options, 100)

# Select box for country
country = st.multiselect('Which country would you like to see?', country_options, ['Brazil'])

# Assign new values for variable 'covid' for visualization
covid = covid[covid['Country'].isin(country)]
#covid = covid[covid['Date'] == date]

# Plot bar chart
fig2 = px.bar(covid, x= 'Country', y= 'Confirmed', color= 'Country', title='Global COVID-19 Cases',
				range_y=[0, 35000], animation_frame='Date', animation_group='Country')

#fig2.layout.updatemenus[0].button[0].args[1]['frame']['duration'] = 30
#fig2.layout.updatemenus[0].button[0].args[1]['transition']['duration'] = 5

fig2.update_layout(width=800)

# Show the plot
st.write(fig2)


'''
## Building predictive models
'''


@st.cache(allow_output_mutation=True)
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model('data/sentiment_pipeline.pkl')
text = st.text_input('Enter your review text below', 'BrainStation is awesome')
prediction = model.predict({text})
if prediction == 1:
    'We predict that this is a positive review!'
else:
    'We predict that this is a negative review!'



