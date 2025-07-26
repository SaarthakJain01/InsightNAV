import streamlit as st
from mftool import Mftool
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import re

mf = Mftool()

def clean_names(text):
  text = re.sub("[^a-zA-Z0-9 ]", "", text)
  text = re.sub(" +", " ", text) # Replace multiple spaces with a single space
  text = text.strip()
  text = text.lower()
  words = text.split()
  # Remove duplicates while maintaining order
  seen = set()
  text = " ".join([word for word in words if not (word in seen or seen.add(word))])
  return text

def get_nav(code):
  nav_df = mf.get_scheme_historical_nav(code = code, as_Dataframe=True)
  nav_df.drop('dayChange', axis=1, inplace=True)
  nav_df = nav_df.reset_index()
  nav_df['date'] = pd.to_datetime(nav_df['date'], dayfirst=True)
  nav_df['nav'] = pd.to_numeric(nav_df['nav'])
  nav_df.sort_values(by='date', inplace=True)
  nav_df.reset_index(drop=True, inplace=True)
  nav_df.rename(columns={'date': 'Date', 'nav': 'NAV'}, inplace=True)
  return nav_df

def get_index_data(benchmark, nav_df):
    index_code = {
    "Nifty 50": '^NSEI',
    "Nifty Next 50": '^NSMIDCP',
    "Sensex": '^BSESN',
    "Nifty Midcap 150": 'NIFTYMIDCAP150.NS',
    "Nifty Smallcap 250": 'NIFTYSMLCAP250.NS'
    }
    start_d = str(nav_df['Date'].min().date())
    end_d = str(nav_df['Date'].max().date())

    index_data = yf.download(index_code[benchmark], start=start_d, end=end_d)
    index_data = index_data['Close'].reset_index()
    index_data.columns = ['Date', 'Price']
    return index_data

def get_norm_data(nav_df, index_df):
  nav_df = nav_df.copy()
  index_df = index_df.copy()
  base_nav = nav_df['NAV'].iloc[0]
  nav_df['MF'] = nav_df['NAV']/base_nav
  base_index = index_df['Price'].iloc[0]
  index_df['Index'] = index_df['Price']/base_index

  merged_df = pd.merge(nav_df[['Date', 'MF']], index_df[['Date', 'Index']], on='Date', how='inner')
#   melted_df = merged_df.melt('Date', var_name='Series', value_name='Value')

  return merged_df

def get_financial_year_span(df, date_column='Date'):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    def get_financial_year(date):
        return date.year if date.month >= 4 else date.year - 1
    df['financial_year'] = df[date_column].apply(get_financial_year)
    financial_years = df['financial_year'].unique()
    if len(financial_years) == 0:
        return 0
    return financial_years.max() - financial_years.min() + 1

def annualized_volatility(df, date_column, value_column):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df['daily_return'] = df[value_column].pct_change()

    one_year_ago = df[date_column].max() - pd.DateOffset(years=1)
    df = df[df[date_column] > one_year_ago]

    returns = df['daily_return'].dropna()
    daily_vol = returns.std()
    return daily_vol * np.sqrt(252)

def rolling_cagr(df, date_column, value_column):
    # Ensure datetime and sort
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    # Define financial year function
    def get_financial_year(date):
        if date.month >= 4:
            return date.year
        else:
            return date.year - 1

    # Add financial year column
    df['financial_year'] = df[date_column].apply(get_financial_year)

    # Group by financial year and calculate returns
    fy_returns = []

    for fy, group in df.groupby('financial_year'):
        group = group.sort_values(date_column)
        start_nav = group.iloc[0][value_column]
        end_nav = group.iloc[-1][value_column]
        if start_nav != 0:
            return_pct = (end_nav - start_nav) / start_nav
            fy_returns.append({'period': f"{fy}-{fy+1}", 'CAGR': return_pct})

    return pd.DataFrame(fy_returns)

def three_years_rolling_cagr(df, date_column, value_column):
    # Prepare data
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    # Function to get FY label
    def get_financial_year(date):
        return date.year if date.month >= 4 else date.year - 1

    df['financial_year'] = df[date_column].apply(get_financial_year)
    all_years = sorted(df['financial_year'].unique())

    results = []

    for i in range(len(all_years) - 2):
        fy_start = all_years[i]
        fy_end = fy_start + 3

        start_date = pd.Timestamp(f"{fy_start}-04-01")
        end_date = pd.Timestamp(f"{fy_end}-03-31")

        # Get closest NAVs to start and end date
        start_df = df[df[date_column] >= start_date]
        end_df = df[df[date_column] <= end_date]

        if not start_df.empty and not end_df.empty:
            start_nav = start_df.iloc[0][value_column]
            end_nav = end_df.iloc[-1][value_column]

            if start_nav != 0:
                cagr = (end_nav / start_nav) ** (1 / 5) - 1
                results.append({
                    'period': f"{fy_start}-{fy_end}",
                    'CAGR': cagr
                })

    return pd.DataFrame(results)

def five_years_rolling_cagr(df, date_column, value_column):
    # Prepare and clean data
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    # Assign financial year
    def get_financial_year(date):
        return date.year if date.month >= 4 else date.year - 1

    df['financial_year'] = df[date_column].apply(get_financial_year)
    all_years = sorted(df['financial_year'].unique())

    results = []

    for i in range(len(all_years) - 4):  # 5-year window
        fy_start = all_years[i]
        fy_end = fy_start + 5

        start_date = pd.Timestamp(f"{fy_start}-04-01")
        end_date = pd.Timestamp(f"{fy_end}-03-31")

        # Find NAVs on or after start and on or before end
        start_df = df[df[date_column] >= start_date]
        end_df = df[df[date_column] <= end_date]

        if not start_df.empty and not end_df.empty:
            start_nav = start_df.iloc[0][value_column]
            end_nav = end_df.iloc[-1][value_column]

            if start_nav > 0:
                cagr = (end_nav / start_nav) ** (1 / 5) - 1
                results.append({
                    'period': f"{fy_start}-{fy_end}",
                    'CAGR': cagr
                })

    return pd.DataFrame(results)

def merge_cagr_s(nav_df, index_df, period = 1):
    if period == 1:
        cagr_nav = rolling_cagr(nav_df, 'Date', 'NAV')
        cagr_index = rolling_cagr(index_df, 'Date', 'Price')
    elif period == 3:
        cagr_nav = three_years_rolling_cagr(nav_df, 'Date', 'NAV')
        cagr_index = three_years_rolling_cagr(index_df, 'Date', 'Price')
    else:
        cagr_nav = five_years_rolling_cagr(nav_df, 'Date', 'NAV')
        cagr_index = five_years_rolling_cagr(index_df, 'Date', 'Price')
    # Add label column to each DataFrame
    cagr_nav['type'] = 'Mutual Fund'
    cagr_index['type'] = 'Index'
    # Combine both into one DataFrame
    combined_cagr = pd.concat([cagr_nav, cagr_index], ignore_index=True)
    return combined_cagr

def calculate_sharpe_ratio(nav_df, risk_free_rate=0.065):
    nav_df = nav_df.sort_values('Date').copy()
    nav_df['returns'] = nav_df['NAV'].pct_change()

    daily_returns = nav_df['returns'].dropna()

    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    # Sharpe Ratio: mean excess return / std dev of excess return
    excess_returns = daily_returns - daily_rf
    sharpe_daily = excess_returns.mean() / excess_returns.std()

    # Annualize Sharpe Ratio
    sharpe_annualized = sharpe_daily * np.sqrt(252)

    return round(sharpe_annualized, 3)

def calculate_alpha_beta(nav_df, index_df, risk_free_rate):
    nav_df['fund_return'] = nav_df['NAV'].pct_change()
    index_df['market_return'] = index_df['Price'].pct_change()
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
    nav_df['excess_fund'] = nav_df['fund_return'] - daily_risk_free_rate
    index_df['excess_market'] = index_df['market_return'] - daily_risk_free_rate
    merged_df = pd.merge(nav_df[['Date', 'excess_fund']], index_df[['Date', 'excess_market']], on='Date', how='inner')
    merged_df.dropna(inplace=True)
    
    X = sm.add_constant(merged_df['excess_market'])
    model = sm.OLS(merged_df['excess_fund'], X).fit()

    alpha_daily = model.params['const']
    beta = model.params['excess_market']
    alpha_annual = alpha_daily * 252

    return alpha_daily, alpha_annual, beta

def plot_two_lines(df, x_col, y_cols, title='Line Plot', x_label='X', y_label='Y'):
    theme = st.get_option("theme.base") or "light"  # Fallback to light if not available
    # Dynamic styling
    if theme == 'dark':
        template = 'plotly_dark'
        font_color = 'white'
        legend_bg = 'rgba(0, 0, 0, 0.7)'
        border_color = 'white'
    else:
        template = 'plotly_white'
        font_color = 'black'
        legend_bg = 'rgba(255, 255, 255, 0.7)'
        border_color = 'black'
    fig = go.Figure()

    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[col],
            mode='lines',
            name=col
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor=legend_bg,
            bordercolor=border_color,
            borderwidth=1,
            font=dict(color=font_color)
        )
    )
    return fig

def plot_fund_vs_index(df, fund_label='Mutual Fund', index_label='Index', title='Fund vs Index Comparison'):
    theme = st.get_option("theme.base") or "light"  # Fallback to light if not available

    # Dynamic styling
    if theme == 'dark':
        template = 'plotly_dark'
        font_color = 'white'
        legend_bg = 'rgba(0, 0, 0, 0.7)'
        border_color = 'white'
    else:
        template = 'plotly_white'
        font_color = 'black'
        legend_bg = 'rgba(255, 255, 255, 0.7)'
        border_color = 'black'

    pivot_df = df.pivot(index='period', columns='type', values='CAGR').reset_index()
    greater_mask = pivot_df[fund_label] > pivot_df[index_label]
    less_mask = pivot_df[fund_label] < pivot_df[index_label]

    fig = go.Figure()

    # Lines
    fig.add_trace(go.Scatter(
        x=pivot_df['period'], y=pivot_df[fund_label],
        mode='lines+markers', name=fund_label,
        line=dict(color='#1f77b4', width=2), line_shape='spline',
        marker=dict(size=6), hovertemplate=f'Period: %{{x}}<br>{fund_label}: %{{y:.2%}}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=pivot_df['period'], y=pivot_df[index_label],
        mode='lines+markers', name=index_label,
        line=dict(color='#7f7f7f', width=2), line_shape='spline',
        marker=dict(size=6), hovertemplate=f'Period: %{{x}}<br>{index_label}: %{{y:.2%}}<extra></extra>'
    ))

    # Points where Fund > Index
    fig.add_trace(go.Scatter(
        x=pivot_df['period'][greater_mask],
        y=pivot_df[fund_label][greater_mask],
        mode='markers',
        name=f'{fund_label} > {index_label}',
        marker=dict(color='green', size=10, line=dict(color='white', width=1.5))
    ))

    # Points where Fund < Index
    fig.add_trace(go.Scatter(
        x=pivot_df['period'][less_mask],
        y=pivot_df[fund_label][less_mask],
        mode='markers',
        name=f'{fund_label} < {index_label}',
        marker=dict(color='red', size=10, line=dict(color='white', width=1.5))
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Period',
        yaxis_title='CAGR',
        xaxis=dict(type='category', tickangle=45),
        yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='gray'),
        template=template,
        font=dict(family='Open Sans', size=13, color=font_color),
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor=legend_bg,
            bordercolor=border_color,
            borderwidth=1,
            font=dict(color=font_color)
        ),
        margin=dict(l=40, r=40, t=80, b=60),
        height=500
    )
    return fig