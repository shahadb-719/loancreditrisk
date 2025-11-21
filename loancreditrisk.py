# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:30:13 2025

@author: shahad
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monte Carlo Loan Risk Simulation", layout="centered")
st.title("Monte Carlo: Loan Interest Rate Risk")

st.sidebar.header ("Loan simulation inputs")

# loan simulation inputs, all options allow you to use both a sliding bar or text(number) input
loan_amount = st.sidebar.slider("Loan amount (1 – 500,000)",min_value=1, max_value=500000,step=1000)
loan_amount = st.sidebar.number_input( "",min_value=1, max_value=500000, value=loan_amount,step=1000)

years = st.sidebar.slider("Loan term (years)",min_value=1,max_value=99,value=5,step=1)
years = st.sidebar.number_input("",min_value=1,max_value=99,value=years,step=1)


payments_per_year = st.sidebar.slider("payments per year",min_value=1,max_value=24,value=12,step=4)
payments_per_year = st.sidebar.number_input("",min_value=1,max_value=24,value=payments_per_year,step=4)


# Interest rate & risk 
st.sidebar.subheader("Interest rate risk")

interest_rate = st.sidebar.slider("Base annual interest rate ( e.g. 0.3 = 30%)",min_value=0.00001,max_value=1.00,value=0.05,step=0.01)
interest_rate = st.sidebar.number_input( "",min_value=0.00001,max_value=1.00,value=interest_rate,step=0.01)

#standard deviation of interest rates
rate_vol = st.sidebar.slider("Interest rate volatility ((% of mean, e.g. 0.3 = 30%))",min_value=0.0,max_value=0.5,value=0.02,step=0.011)
rate_vol = st.sidebar.number_input("",min_value=0.0,max_value=0.5,value=rate_vol,step=0.01)

#business cashflows
st.sidebar.subheader ("Business cashflows")

business_cash_flow = st.sidebar.slider("Expected annual business cash flow",min_value=0,max_value=10_000_000,value=40_000,step=5_000)
business_cash_flow = st.sidebar.number_input("",min_value=0,max_value=10_000_000,value=business_cash_flow,step=5_000)

#standard deviation of cash flow
cash_flow_volatility= st.sidebar.slider("Cash flow volatility (% of mean, e.g. 0.3 = 30%)",min_value=0.0,max_value=2.0,value=0.3,step=0.05)
cash_flow_volatility = st.sidebar.number_input( "",min_value=0.0,max_value=2.0,value=cash_flow_volatility,step=0.05)


# Monte Carlo controls
st.sidebar.subheader ("Monte Carlo Controls")

#each trial simulates 1 business
n_trials= st.sidebar.slider("Number of Monte Carlo scenarios",min_value=500,max_value=100000,value=5000,step=500)
n_trials = st.sidebar.number_input("",min_value=500,max_value=100000,value=n_trials,step=500)
run_sim = st.button("Run Monte Carlo")



def loan_payment(loan_amount, r, years, payments_per_year):
#calculates the annual loan payments for a fixed rate loan
    n = years* payments_per_year  # yearly payments
    r = r/ payments_per_year # annual interets -> interest per payment
    payment = loan_amount * r / (1 - (1 + r) ** (-n)) # annual loan payment (loan x interest/ 1-(1+interest)^-years)
    return payment

def total_interest_paid(loan_amount, r, years, payments_per_year):
#calculates how much of the payment is interest 
    n = years * payments_per_year # total number of payments
    payment = loan_payment(loan_amount, r, years, payments_per_year)
    total_paid = payment * n  # total paid over all periods
    return total_paid - loan_amount  # interest = total paid - principal

if run_sim: #checks if the simulation button is pressed (monte carlo controls)
 if run_sim:
    progress = st.progress(0)
    status = st.empty()

    for i in range(100):
        progress.progress(i + 1)
        status.write(f"Running simulation… {i+1}%")   

        
    n_trials = int(n_trials) #makes sure the number of scenarios is an integer

    # 1) Simulate random interest rates around the original rate, rate vol is the standard deviation, loc is the mean, size = # of monte carlo sims
    simulated_rates = np.random.normal(loc=interest_rate,scale=rate_vol,size=n_trials)

    # 2) Compute annual loan payment for each scenario based on the simulated interest rates, one payment amount per trial
    annual_payments = loan_payment(loan_amount, simulated_rates, years, payments_per_year)  # shape: (n_trials,)

    # 3) calculates standard deviation = Simulate business cash flows for each year and scenario
    simulated_cashflow = cash_flow_volatility * business_cash_flow
    
    # shape: (n_trials, years)
    cash_flows = np.random.normal(loc=business_cash_flow,scale=simulated_cashflow,size=(n_trials, years))
    
    # 4) Default condition: if any year cash_flow < annual_payment
    # make annual_payments shape (n_trials, 1) so it can compare to each year
    required = annual_payments.reshape(-1, 1) 
    default_matrix = cash_flows < required   # check if payments were not covered, true if they weren't
    default_flags = default_matrix.any(axis=1)  # True if default in any year

    # 5) Calculate total interest for each scenario
    total_interests = total_interest_paid( loan_amount,simulated_rates,years,payments_per_year)

    # Build DataFrame
    df = pd.DataFrame({"simulated_rate": simulated_rates,"annual_payment": annual_payments,"total_interest": total_interests,"default": default_flags})


    # The display set up
    st.subheader("Loan & business setup")
    st.write(f"**Loan amount:** ${loan_amount:,.0f}")
    st.write(f"**Base interest rate:** {interest_rate*100:.2f}%")

    st.write(f"**Loan term:** {years: } years")
    st.write(f"**Expected annual cash flow:** ${business_cash_flow:}")
    st.write(f"**Cash flow volatility:** {cash_flow_volatility*100:.2f}% of mean")
    st.write(f"**Interest rate volatility (σ):** {rate_vol*100:.2f}%")
    st.write(f"**Number of scenarios:** {n_trials:,}")
    
    #  Calculates the probability of default 
    pd_estimate = df["default"].mean()  # probability (fraction) of scenarios with default
    st.subheader("Estimated probability of default (PD)")
    st.write(f"**PD over {years} years:** {pd_estimate*100:.2f}%")
    # shows how many scenarios defaulted:
    st.write(f"Scenarios with default: {df['default'].sum():,} / {n_trials:,}") # 0 = False, 1 = True, probabilities of default
    
    #  Monte Carlo summary statistics, does not show cashflows as it is not a part of the actual loan
    st.subheader("Monte Carlo results (summary)")
    st.write(df[["simulated_rate", "annual_payment", "total_interest"]].describe().T)
    
    # Histogram of total interest (how much the borrower actually pays in ech scenario)
    st.subheader("Distribution of total interest paid")
    fig, ax = plt.subplots(figsize=(6, 3.5))    # Create graph
    ax.hist(df["total_interest"], bins=30)
    ax.set_xlabel("Total interest paid over the life of the loan")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Histogram of simulated interest rates (all the randomly generated rates)
    st.subheader("Distribution of simulated interest rates")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))    # Create graph
    #plots the histogram, converts decimals to % then outs them in bins 
    ax2.hist(df["simulated_rate"] * 100, bins=30) 
    ax2.set_xlabel("Interest rate (%)")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    
    st.subheader("Probability of Default by Interest Rate Bucket")

    fig_bucket, ax_bucket = plt.subplots(figsize=(7, 3.5))

    # Bar chart instead of line plot
    #Probability of Default by Interest Rate Bucket (Histogram) 
    st.subheader("Probability of Default by Interest Rate Bucket")
    
    # Create interest rate buckets (x-axis)
    df["rate_bucket"] = pd.cut(df["simulated_rate"], bins=10)
    
    # Compute Probability per bucket, gives the average of a true (1) or false value (0) (y-axis)
    pd_by_bucket = df.groupby("rate_bucket")["default"].mean()
    fig_bucket, ax_bucket = plt.subplots(figsize=(7, 3.5))    # Create graph
    
    # Convert intrevals to string for labeling (only x-axis)
    bucket_labels = pd_by_bucket.index.astype(str)
    ax_bucket.bar(bucket_labels, pd_by_bucket.values, width=0.8) # Plot bar chart
    
    # Fix tick labels, makes sure the labels don't overlap (x-axis)
    ax_bucket.set_xticks(range(len(bucket_labels)))
    ax_bucket.set_xticklabels(bucket_labels, rotation=45, ha="right")
    
    ax_bucket.set_ylabel("Probability of Default")
    ax_bucket.set_xlabel("Interest Rate Bucket")
    ax_bucket.set_title("PD by Interest Rate Bucket")
    
    st.pyplot(fig_bucket)

        
     # Pie chart for default vs no default
    st.subheader("Default vs No Default")
    fig3, ax3 = plt.subplots() #makes the graph
    counts = df["default"].value_counts()
    labels = ["No default", "Default"]
    #counts the # of defaults and no defaults
    values = [counts.get(False, 0), counts.get(True, 0)]
    ax3.pie(values, labels=labels, autopct="%1.1f%%") #makes pie chart
    st.pyplot(fig3)
    
    st.info(
        "The Monte Carlo simulation is meant to estimate the probability of default \n"
        "for a small business loan \n\n"
        
        "The probability of default here is estimated by the percentage of simulated\n "
        "scenarios where the business cash flow is not sufficient to cover the \n"
        "loan payment in at least one year.\n \n"
        
        "Parameters considered are interest rate,and their volatilities = simulated \n"
        "interest rates, cashflows and their volatilities = simulated cashflows,\n"
        "the loan amount, years (t) and the number of payments per year. \n \n" 
        
        "The user also gets to choose the amount of trials they would like to run \n"
        "each trial represents a possible future for the business, the higher \n"
        " the number of trials is the 'better' the estimate of probabilities is \n"
    )

              
