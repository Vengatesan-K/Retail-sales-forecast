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
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

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
st.markdown("---")
average_sales_per_week = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
average_sales_per_week= average_sales_per_week.sort_values('Date', ascending=True)
fig1 = px.line(average_sales_per_week, x='Date', y='Weekly_Sales', title='Total Weekly Sales Over Time',markers=True)

# Customize the layout
fig1.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Weekly Sales',title_x=0.45,height=400
)
fig1.add_annotation(
    text='Highest Sales : $56.71M(Dec-24-2010)',
    x='2010-12-24',  
    y=56710000,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='red',
    bgcolor='rgba(255, 255, 255, 0.7)',
    opacity=0.8
)
# Use Streamlit to display the Plotly figure
st.plotly_chart(fig1, use_container_width=True)
st.markdown("---")

average_sales_per_week_n = df.groupby(by=['Date','IsHoliday'], as_index=False)['Weekly_Sales'].sum()
colors = {0: 'red', 1: 'green'}
fig9 = px.line(
    average_sales_per_week_n,
    x="Date",
    y="Weekly_Sales",
    color="IsHoliday",color_discrete_map=colors,
    labels={"Weekly_Sales": "Weekly Sales", "Date": "Date"},
    title="Weekly Sales with Holidays",
)

# Update figure height and title position
fig9.update_layout(
    height=400,
    title=dict(
        x=0.45
    )
)

st.plotly_chart(fig9,use_container_width=True)
st.markdown("---")


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
fig2.add_annotation(
    text='Highest Sales : $551M(April)',
    x=4,  
    y=551998331,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='red',
    bgcolor='rgba(255, 255, 255, 0.7)',
    opacity=0.8
)
# Use Streamlit to display the Plotly figure
st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")
average_sales_per_year = average_sales_per_date.groupby(by=['Year'], as_index=False)['Weekly_Sales'].sum()

fig3 = px.bar(average_sales_per_year, x='Year', y='Weekly_Sales', title='Total Weekly Sales per Year',color='Weekly_Sales',
    color_continuous_scale='viridis')

fig3.update_layout(
    xaxis_title='Year',
    yaxis_title='Total Weekly Sales',title_x=0.45,height=400
)
fig3.add_annotation(
    text='Highest Sales : $2B(2011)',
    x=2011,  
    y=2069889649,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='red',
    bgcolor='rgba(255, 255, 255, 0.7)',
    opacity=0.8
)
# Use Streamlit to display the Plotly figure
st.plotly_chart(fig3, use_container_width=True)
st.markdown("---")
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
fig4.add_annotation(
    text='Best Sales over type : A',
    x='2012-09-14',  
    y=25000000,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='blue',
    bgcolor='rgba(255, 255, 255, 0.7)',
    opacity=0.8
)
# Use Streamlit to display the Plotly figure
st.plotly_chart(fig4, use_container_width=True)
st.markdown("---")
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
fig5.add_annotation(
    text='Highest Sales - 267.50M(Dept-76)',
    x=76,  
    y=279508700,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor='green',
    bgcolor='rgba(255, 255, 255, 0.7)',
    opacity=0.8
)
# Use Streamlit to display the Plotly figure
st.plotly_chart(fig5, use_container_width=True)
st.markdown("---")
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
st.markdown("---")

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
st.markdown("---")
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
high_corr_threshold = 0.9  # You can adjust this threshold based on your preference

high_corr_text = ""
for i, column in enumerate(selected_columns):
    for j, row in enumerate(selected_columns):
        correlation_value = corr_matrix.iloc[i, j]
        if abs(correlation_value) > high_corr_threshold and i != j:
            high_corr_text += f"<li>({column} and {row} : {correlation_value:.2f})</li>"

import streamlit as st

# Your high correlation text
high_corr_text = """
    <ul>
        <li>The retail weekly sales experience heightened levels during December and February, primarily attributable to increased holiday shopping activity and high markdowns (discounted prices)</li>
        <li>Weekly sales of stores of bigger sizes are generally higher than stores of smaller sizes. Stores of type A are the largest, followed by B and then C being the smallest size stores. However, the minimum weekly sales of store B are higher than that of store A.</li>
        <li>🧮 High Correlation Columns : {high_corr_text} </li>
        <!-- Add more items as needed -->
    </ul>
"""

# Custom CSS styling
custom_css = """
    <style>
        .high-corr-container {
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
            margin-top: 20px;
        }

        .bullet-list {
            list-style-type: disc;
            padding-left: 20px;
        }
    </style>
"""

# Display the custom-styled content
st.markdown(f"{custom_css}<div class='high-corr-container'><h4>💡 Insights:</h4><ul class='bullet-list'>{high_corr_text}</ul></div>", unsafe_allow_html=True)
st.markdown("---")


df.Date = pd.to_datetime(df.Date)
store_1 = df[(df.Store == 1) & (df.Dept == 1)].sort_values('Date')
df_1 = store_1[['Date', 'Weekly_Sales']]
df_1.columns = ['ds', 'y']
m = Prophet(interval_width=.95, 
            daily_seasonality=False, yearly_seasonality=True,
            weekly_seasonality=True,seasonality_mode='multiplicative').fit(df_1)
future = m.make_future_dataframe(periods=10, freq='W')
forecast = m.predict(future)

fig9 = px.line(forecast, x='ds', y=['yhat'], title='Prophet Forecast')
fig9.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower Bound'))
fig9.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper Bound'))

st.plotly_chart(fig9,use_container_width=True)
import plotly.subplots as sp
fig_components = sp.make_subplots(rows=3, cols=1, subplot_titles=('Trend', 'Weekly', 'Yearly'))

# Plotting components
fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'), row=1, col=1)
fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'], mode='lines', name='Weekly'), row=2, col=1)
fig_components.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], mode='lines', name='Yearly'), row=3, col=1)

# Customize layout
fig_components.update_layout(title_text='Prophet Components', height=800)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig_components,use_container_width=True)

import os

class suppress_stdout_stderr(object):

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
def getCrossValidationData(m):
    with suppress_stdout_stderr():
        c_v = cross_validation(m, 
                               initial='120W',   # Initially, the model will be trained in 120 weeks.
                               period='2W',      # After each model tested, we'll add 2 more weeks.
                               horizon ='2W',    # The forecasting will happen in a range of 2 weeks.
                               parallel="processes",   # To acellerate the cross-validation.
                              )
    return c_v

def getPerfomanceMetrics(m):
    return performance_metrics(getCrossValidationData(m), 
                               rolling_window=1, # Generate metrics for the whole (100%) seen data.
                              )
    
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2011-02-06', '2012-02-05', '2013-02-03']),
  'lower_window': -2,
  'upper_window': 2,
})
easter = pd.DataFrame({
  'holiday': 'easter',
  'ds': pd.to_datetime(['2010-04-05', '2011-04-25', '2012-04-09', '2013-04-01']),
  'lower_window': -2,
  'upper_window': 1,
})
mothers_day = pd.DataFrame({
    'holiday': "mother's day",
    'ds': pd.to_datetime(['2010-05-09', '2011-05-08', '2012-05-13', '2013-02-12']),
    'lower_window': -3,
    'upper_window': 0,
})
fathers_day = pd.DataFrame({
    'holiday': "father's day",
    'ds': pd.to_datetime(['2010-06-19', '2011-06-19', '2012-06-17', '2013-06-16']),
    'lower_window': -3,
    'upper_window': 0,
})
halloween = pd.DataFrame({
    'holiday': "father's day",
    'ds': pd.to_datetime(['2010-10-31', '2011-10-31', '2012-10-31', '2013-10-31']),
    'lower_window': -3,
    'upper_window': 2,
})
black_friday = pd.DataFrame({
    'holiday': "black friday",
    'ds': pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']),
    'lower_window': 0,
    'upper_window': 0,
})
cyber_monday = pd.DataFrame({
    'holiday': "cyber monday",
    'ds': pd.to_datetime(['2010-11-29', '2011-11-28', '2012-12-26', '2013-12-02']),
    'lower_window': 0,
    'upper_window': 0,
})

holidays = pd.concat((superbowls, 
                      easter,
                      mothers_day, 
                      fathers_day, 
                      halloween, 
                      black_friday, 
                      cyber_monday))


m = Prophet(holidays=holidays,
            interval_width=.95, 
            daily_seasonality=False)
m.add_country_holidays(country_name='US')
with suppress_stdout_stderr():
    m.fit(df_1)
    
end_date = st.date_input('Select the end date for forecasting:', min_value=df_1['ds'].max(), max_value=pd.Timestamp.now())

future = pd.DataFrame({'ds': pd.date_range(start=df_1['ds'].min(), end=end_date, freq='W')})

forecast = m.predict(future)

fig_forecast = px.line(forecast, x='ds', y=['yhat'], title=f'Prophet Forecast (Until {end_date})')
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower Bound'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper Bound'))

st.plotly_chart(fig_forecast,use_container_width=True)





#st.markdown("<hr>", unsafe_allow_html=True)
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
        <a class="nav-link" href="#analysis-of-historical-retail-data">Analysis</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#predict-week-sales">Forecasting Sales</a>
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












