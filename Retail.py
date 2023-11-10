import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_extras.mention import mention 
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import re
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objs as go
st.set_page_config(page_title='Retail', layout='wide', page_icon="🛒")

st.write("""
<div style='text-align:right'>
    <h1 style='color:#1e90ff;'>🛒Retail Sales Forecasting</h1>
</div>
""", unsafe_allow_html=True)

def style_metric_cards(
    background_color: str = "#FFF",
    border_size_px: int = 1,
    border_color: str = "#CCC",
    border_radius_px: int = 5,
    border_left_color: str = "#9AD8E1",
    box_shadow: bool = True,
):

    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    st.markdown(
        f"""
        <style>
            div[data-testid="metric-container"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


with open("style1.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
#####################
# Navigation

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #16A2CB;">
  <a class="navbar-brand" href="https://www.linkedin.com/in/vengatesan2612/" target="_blank">Retail Sales Forecast</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="/">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#predict-price">Price</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)



st.markdown('''
<h2 style="color:#1e90ff; margin-bottom: 0;">Predict Week Sales</h2>
<hr style="border: 1px solid #1e90ff; background-color: #1e90ff; margin-top: 0;">
''', unsafe_allow_html=True)
holiday_options = ['True','False']
type_options = ['A', 'B', 'C']
size_options = [151315, 202307,  37392, 205863,  34875, 202505,  70713, 155078,
       125833, 126512, 207499, 112238, 219622, 200898, 123737,  57197,
        93188, 120653, 203819, 203742, 140167, 119557, 114533, 128107,
       152513, 204184, 206302,  93638,  42988, 203750, 203007,  39690,
       158114, 103681,  39910, 184109, 155083, 196321,  41062, 118221]

# Define the widgets for user input
with st.form("my_form"):
    col1,col2,col3=st.columns([5,2,5])
with col1:
    st.write(' ')
    holiday = st.selectbox("Holiday", holiday_options,key=1)
    store = st.slider("Store (Min: 1, Max: 45)", 1.0, 45.0, step=1.0)   
    type = st.selectbox("Type", sorted(type_options),key=3)
    dept = st.slider("Department (Min: 1, Max: 99)", 1.0, 99.0, step=1.0)   
with col3:               
    st.write(' ')
    size = st.selectbox("Size", size_options,key=2)
    year = st.slider("Year (Min: 2010, Max: 2025)", 2010.0, 2025.0, step=1.0)   
    month = st.slider("Month (Min: 1, Max: 12)", 1.0, 12.0, step=1.0)
    week_of_year= st.slider("Week (Min: 1, Max: 48)", 1.0, 48.0, step=1.0)
    submit_button = st.form_submit_button(label="Weekly sales")

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #1e90ff;
            color: #fff5ee;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)


if submit_button:
    import pickle

    with open(r"model.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    with open(r'scaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

    # Convert boolean values to actual boolean
    holiday_bool = (holiday == 'True')
    type_bool = (type == 'True')

    # Create a NumPy array for prediction
    user_input_array = np.array([[store, dept, size, year, month, week_of_year, holiday_bool, type_bool]])

    # Define which columns are numeric and which are categorical
    numeric_cols = [1, 3, 4, 5, 6]
    categorical_cols = [0, 2, 7]

    # Create a ColumnTransformer to handle both numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop=None, sparse=False), categorical_cols)  # Updated to include all columns
        ])

    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(user_input_array)

    # Ensure the number of features matches the model's expectations
    expected_features = X_preprocessed.shape[1]

    if expected_features != 8:  # Adjust this number based on your model's requirements
        st.write(f"Error: Expected 8 features, but got {expected_features} features.")
    else:
        # Make predictions
        prediction = loaded_model.predict(X_preprocessed)

        st.info(f"The predicted week sales is :🔖{prediction[0]}")












