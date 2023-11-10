import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_extras.mention import mention 
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import re
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objs as go
st.set_page_config(page_title='Retail', layout='wide', page_icon="🛒")
import plotly.figure_factory as ff
st.write("""
<div style='text-align:right'>
    <h1 style='color:#1e90ff;'>🛒Retail Sales Forecasting</h1>
</div>
""", unsafe_allow_html=True)
container_style = """
    border-radius: 10px;
    background-color: #f5f5f5;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
"""

# Use st.markdown to display the content with the defined style
st.markdown(
    f'<div style="{container_style}">'
    '<h3>📌Modeling Retail Data Challenge</h3>'
    '<p>One challenge of modeling retail data is the need to make decisions based on limited history. Holidays and select major events come once a year, and so does the chance to see how strategic decisions impacted the bottom line. In addition, markdowns are known to affect sales – the challenge is to predict which departments will be affected and to what extent.</p>'
    '</div>',
    unsafe_allow_html=True
)


# Use st.markdown to display the rephrased content in a second container
st.markdown(
    f'<div style="{container_style}">'
    '<h3 style="color: #333;">🛍️Sales Data Overview</h3>'
    '<p>The dataset includes historical sales data for 45 stores situated in various regions. Each store comprises multiple departments. The company organizes promotional markdown events, particularly before major holidays such as Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks encompassing these holidays hold five times more significance in the evaluation compared to regular weeks.</p>'
    '</div>',
    unsafe_allow_html=True
)


data = {
    'Attribute': ['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1-5', 'CPI', 'Unemployment', 'IsHoliday','Dept','Weekly_Sales','Size','Type'],
    'Details': ['- store number', '- week', '- average temperature in the region',
                '- cost of fuel in the region',
                '- anonymized data related to promotional markdowns.',
                '- consumer price index', '- unemployment rate', '- whether the week is a special holiday week',
                '- the department number',
                '- sales for the given department in the given store',
                '- type A,B,C',
                '- size of store']
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Use Plotly to create a colorful table
fig = ff.create_table(df, colorscale='Earth')

col1,col2 = st.columns([7,3])
with col1:
 st.plotly_chart(fig,use_container_width=True)
with col2:
 st_lottie('https://lottie.host/91ab9567-96f3-4fd6-804b-f97c3a0f1be6/vYJybHorvu.json')
 
 
st.markdown('''
<h2 style="color:#1e90ff; margin-bottom: 0;">Analysis of Historical Retail Data</h2>
<hr style="border: 1px solid #1e90ff; background-color: #1e90ff; margin-top: 0;">
''', unsafe_allow_html=True) 

df = pd.read_csv('retail_pysan.csv')
selected_columns = ['Weekly_Sales', 'Fuel_Price', 'Temperature', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

# Melt the DataFrame to have 'Date' as the common x-axis and the selected columns as values
melted_df = pd.melt(df, id_vars=['Date'], value_vars=selected_columns, var_name='feature', value_name='value')

# Create area plot with subplots using Plotly Express
fig = px.area(melted_df, x='Date', y='value', color='feature', facet_col="feature", facet_col_wrap=2, title='Retail Sales analysis')

# Update layout
fig.update_layout(
    yaxis_title='Values',
    xaxis_title='Date',
    template='plotly_dark',title_x=0.48,height=600
)

st.plotly_chart(fig, use_container_width=True)

average_sales_per_week = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
average_sales_per_week= average_sales_per_week.sort_values('Date', ascending=True)
fig1 = px.line(average_sales_per_week, x='Date', y='Weekly_Sales', title='Total Weekly Sales Over Time',markers=True)

# Customize the layout
fig1.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Weekly Sales',title_x=0.45,height=400
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig1, use_container_width=True)

average_sales_per_date = df.loc[:, ('Date','Weekly_Sales')]
average_sales_per_date['Month'] =pd.DatetimeIndex(average_sales_per_date['Date']).month
average_sales_per_date['Year'] =pd.DatetimeIndex(average_sales_per_date['Date']).year
average_sales_per_month = average_sales_per_date.groupby(by=['Month'], as_index=False)['Weekly_Sales'].sum()

fig2 = px.bar(average_sales_per_month, x='Month', y='Weekly_Sales', title='Total Weekly Sales per Month',color='Weekly_Sales',
    color_continuous_scale='viridis')

fig2.update_layout(
    xaxis_title='Month',
    yaxis_title='Total Weekly Sales',title_x=0.45,height=400
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig2, use_container_width=True)

average_sales_per_year = average_sales_per_date.groupby(by=['Year'], as_index=False)['Weekly_Sales'].sum()

fig3 = px.bar(average_sales_per_year, x='Year', y='Weekly_Sales', title='Total Weekly Sales per Year',color='Weekly_Sales',
    color_continuous_scale='viridis')

fig3.update_layout(
    xaxis_title='Year',
    yaxis_title='Total Weekly Sales',title_x=0.45,height=400
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig3, use_container_width=True)

sales_per_store_type = df.groupby(by=['Type'], as_index=False)['Weekly_Sales'].sum()
sales_per_store_type_n = df.groupby(by=['Date','Type'], as_index=False)['Weekly_Sales'].sum()

fig4 = px.line(sales_per_store_type_n, x='Date', y='Weekly_Sales', title='Weekly Sales per Store Type Over Time',
    color='Type',  # Use the 'Type' column for coloring
    labels={'Weekly_Sales': 'Total Weekly Sales'})

# Customize the layout
fig4.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Weekly Sales',title_x=0.45,height=400
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig4, use_container_width=True)

sales_by_dept = df.groupby(by=['Dept'], as_index=False)['Weekly_Sales'].sum()

fig5 = go.Figure()

# Add vertical lines using vlines
fig5.add_trace(go.Scatter(
    x=sales_by_dept.index,
    y=sales_by_dept['Weekly_Sales'],
    mode='lines+markers',
    line=dict(color='cyan'),
    marker=dict(color='red', size=8),
    name='Sales by Department'
))

# Customize the layout
fig5.update_layout(
    title='Departmentwise Sales',
    xaxis_title='Department',
    yaxis_title='Sales',title_x=0.45,height=400
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig5, use_container_width=True)

train_markdown = df[df.MarkDown2.notnull()]
train_markdown = train_markdown.groupby("Date").agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})
fig6 = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=['Markdown1', 'Markdown2', 'Markdown3', 'Markdown4', 'Markdown5'])

# Add traces for each Markdown column
for i, column in enumerate(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']):
    trace = go.Scatter(x=train_markdown.index, y=train_markdown[column], mode='lines', name=column)
    fig6.add_trace(trace, row=i+1, col=1)

# Customize the layout
fig6.update_layout(
    height=800,
    title_text='Timeline Markdown',
    showlegend=False,title_x=0.45
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig6, use_container_width=True)


train_markdown.index = pd.to_datetime(train_markdown.index)

# Group by month and calculate the mean
train_markdown_month = train_markdown.resample('M').mean()

# Create a stacked bar plot using Plotly Express
fig7 = go.Figure()

for column in train_markdown_month.columns:
    fig7.add_trace(go.Bar(
        x=train_markdown_month.index.month,
        y=train_markdown_month[column],
        name=column
    ))

# Customize the layout
fig7.update_layout(
    barmode='stack',
    title_text='Stacked Monthwise Markdown',
    xaxis_title='Month',
    yaxis_title='Markdown',title_x=0.40
)

# Use Streamlit to display the Plotly figure
st.plotly_chart(fig7, use_container_width=True)

selected_columns = ['Store', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                     'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']

# Subset the DataFrame with selected columns
df_selected = df[selected_columns]

# Calculate the correlation matrix
corr_matrix = df_selected.corr()

# Create a correlation heatmap using Plotly Express
fig8 = px.imshow(
    corr_matrix,
    labels=dict(color='Correlation'),
    x=selected_columns,
    y=selected_columns,
    color_continuous_scale='blues',  # Use a valid Plotly colorscale
    title='Correlation Plot'
)
fig8.update_layout(
    height=500, 
    width=500,title_x=0.40
)
# Use Streamlit to display the Plotly figure
st.plotly_chart(fig8,use_container_width=True)


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












