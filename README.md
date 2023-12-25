# 🛒Retail-sales-forecast
> Designed to offer a user-friendly interface that allows users to input a specific date and receive information about what happened on that particular day, especially in terms of retail sales.

## Context
The Challenge - One challenge of modeling retail data is the need to make decisions based on limited history. Holidays and select major events come once a year, and so does the chance to see how strategic decisions impacted the bottom line. In addition, markdowns are known to affect sales – the challenge is to predict which departments will be affected and to what extent.

## Content
You are provided with historical sales data for 45 stores located in different regions - each store contains a number of departments. The company also runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks.

### Stores
***Anonymized information about the 45 stores, indicating the type and size of store.***

### Features
***Contains additional data related to the store, department, and regional activity for the given dates.***

- Store - the store number
- Date - the week
- Temperature - average temperature in the region
- Fuel_Price - cost of fuel in the region
- MarkDown1-5 - anonymized data related to promotional markdowns. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA
- CPI - the consumer price index
- Unemployment - the unemployment rate
- IsHoliday - whether the week is a special holiday week

### Sales
***Historical sales data, which covers to 2010-02-05 to 2012-11-01. Within this tab you will find the following fields :***

- Store - the store number
- Dept - the department number
- Date - the week
- Weekly_Sales -  sales for the given department in the given store
- IsHoliday - whether the week is a special holiday week

## Problem Statement

- Predict the department-wide sales for each store for the following year.
  - Model the effects of markdowns on holiday weeks.
    - Provide recommended actions based on the insights drawn, with prioritization placed on largest business impact.

## Exploratory Data Analysis (EDA)
![newplot (45)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/32e8600b-ce79-4bf3-b193-a851211b5413)

![newplot (46)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/2db84236-3130-4ac5-9b2b-9f6ef9b015eb)

![newplot (47)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/d7670473-8998-4d2d-98a4-3d95b55d1431)

![newplot (48)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/a2ce7407-7666-4895-93a7-526d6ea07b85)

![newplot (49)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/353e287d-918b-4e2c-8c5c-60071211b4ea)

![newplot (52)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/e3040e6e-68a2-4921-a8bc-8040e3efd1b2)

![newplot (53)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/6b097b52-53d1-494d-89c4-4700deb98bcf)

![newplot (54)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/f9d4033d-98fd-4224-8bf2-24218c37ce32)

![newplot (55)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/3404be84-b0ad-429c-9cb8-77036bb83a1f)
***
## TimeSeries Forecasting using Prophet
![newplot (56)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/c8ede5fe-562d-40f7-adea-a8361f4e663e)

![newplot (57)](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/1f5f01eb-26fc-48d8-a3bb-3528673e6f17)

## Predictive analysis using Regression 
![Screenshot 2023-12-25 184850](https://github.com/Vengatesan-K/IMDB-Movies-Analysis/assets/128688827/1b69b0bd-afaf-4e5f-a0aa-e444610e1a31)


