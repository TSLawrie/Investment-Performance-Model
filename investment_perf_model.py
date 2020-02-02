"""Program to model the performance of investments over time."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def inv_model(
    # All investment types
    platform_charge_250 = 0.0045,      # Platform's monthly fund charge on the first £250,000 (as a fraction of the total value of all funds)
    platform_charge_1 = 0.0025,      # Platform's monthly fund charge on the value between £250,000 - £1m (as a fraction of the total value of all funds)
    platform_charge_2 = 0.001,      # Platform's monthly fund charge on the value between £1m - £2m (as a fraction of the total value of all funds)
    platform_charge_over = 0,      # Platform's monthly fund charge on the value over £2m (as a fraction of the total value of all funds)
    inv_return = 0.17,        # Average yearly return across all investment funds
    inv_ocr = 0.005,     # Investment fund's OCR (yearly charge) (Avg Fund: 1, Avg Trust: 1.3)
    initial_payment = 4000,         # Initial amount deposited into account
    monthly_payment = 200,       # Amount deposited in ISA each month (starting at the same time as the initial deposit)
    total_deposited = 1000,        # Running total of amount deposited (set as same as initial payment here)
    inv_months = 40*12,      # Months of investment
    total_months = 0,       # Total months of investment already passed
    total_platform_fee = 0,      # Total platform fee already payed
    salary_increase_rate = 0.04,     # Average yearly increase in salary
    inflation = 0.02,        # Average yearly inflation

    # Investment trusts only
    platform_charge_cap = None,        # Platform's cap to monthly charges (45/12 for HL) (set to None if n/a)
    trades = 12,        # Amount of trades placed in a year
    trade_charge = 0,      # Flat fee per trade
    stamp_duty = 0.00      # Charge applied to every trade
    ):
    """
    Calculate the expected total value of Stocks and Shares ISA over a certain period of time. Tuned for Hargreaves Landown.
    
    Returns:
        total_value (float): final value of investments
        monthly_payment (float): monthly deposit amount
        inv_months (int): months of investment passed
        total_platform_fee (float): runnning total of charges payed
        isa (DataFrame): log of stats

    """
    # Runnning totals
    total_value = initial_payment

    isa = pd.DataFrame(columns=['Year', 'Month', 'Total Value', 'Amount Deposited', 'Investment Returns', 'Charges', 'Total Charges', 'Total Charges (%)', 'Fund Charges','Inflation'])         # Create DataFrame to keep logs of various stats

    # Calculations done once a month (at end of month)
    for n in range(total_months, inv_months):

        total_value = (total_value + monthly_payment * (1 - stamp_duty) - trade_charge * trades / 12) * (1 + inv_return/12)     # Add this month's returns to the total value
        platform_fee = min(filter(None, [platform_charge_cap, platform_charge_calc(total_value, platform_charge_250, platform_charge_1, platform_charge_2, platform_charge_over)]))
        total_value -= platform_fee       # Subtract this month's platform fee from the total value
        inv_fee = total_value * inv_ocr / 12
        total_value -= inv_fee         # Subtract this month's investment fund fee from the total value
        total_value /= 1 + inflation / 12       # Adjust the total value for inflation
        total_deposited += monthly_payment
        total_platform_fee += platform_fee + monthly_payment * stamp_duty + trade_charge * trades / 12 + inv_fee

        isa = isa.append({'Year': (n+1)/12, 'Month': (n+1), 'Total Value': total_value, 'Monthly Deposit': monthly_payment, 'Amount Deposited': total_deposited, 'Investment Returns': ((total_value/total_deposited)*100)-100, 'Charges': platform_fee+monthly_payment*stamp_duty+trade_charge*trades/12+inv_fee, 'Total Charges': total_platform_fee, 'Total Charges (%)': total_platform_fee/total_value*100, 'Fund Charges': inv_fee,'Inflation': inflation}, ignore_index=True)        # Log the stats

        # Increase monthly payment proportional to average yearly salary increase
        if (n+1) % 12 == 0 and n/12 < 40:
            monthly_payment *= 1 + salary_increase_rate

    tot_val(isa)
    inv_ret(isa)

    return total_value, monthly_payment, inv_months, total_platform_fee, total_deposited, isa


def platform_charge_calc(total_value, platform_charge_250, platform_charge_1, platform_charge_2, platform_charge_over):
    """
    Calculate the platform's charges for each fund value bracket (4 brackets for HL).
    
    Args:
        total_value (float): Total value of the investment funds
        platform_charge_250 (float): Platform's monthly fund charge on the first £250,000 (as a fraction of the total value of all funds)
        platform_charge_1 (float): Platform's monthly fund charge on the value between £250,000 - £1m (as a fraction of the total value of all funds)
        platform_charge_2 (float): Platform's monthly fund charge on the value between £1m - £2m (as a fraction of the total value of all funds)
        platform_charge_over (float): Platform's monthly fund charge on the value over £2m (as a fraction of the total value of all funds)
    
    Returns:
        float: Platform charge taken at the end of the month

    """
    bracket_limit = 0
    bracket = 1
    platform_charge = 0

    while total_value > bracket_limit and bracket < 5:

        if bracket == 1:
            bracket_limit = 250000
            platform_charge = min(total_value, bracket_limit) * platform_charge_250 / 12

        elif bracket == 2:
            bracket_limit = 1000000
            platform_charge += (min(total_value, bracket_limit) - 250000) * platform_charge_1 / 12

        elif bracket == 3:
            bracket_limit = 2000000
            platform_charge += (min(total_value, bracket_limit) - 1000000) * platform_charge_2 / 12

        else:
            bracket_limit = 250000
            platform_charge += (total_value - 2000000) * platform_charge_over / 12

        bracket += 1

    return platform_charge


def tot_val(isa):
    """
    Plot the total value over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Total Value'
    y_max = isa.iloc[-1].at['Total Value']
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Total Value']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('\xA3 {:,.0f}'.format(y_max), xy=(x_max, y_max), xytext=(x_max-(x_max-x_min)*1/8, y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()

def inv_ret(isa):
    """
    Plot the investment returns over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Investment Returns'
    y_max = isa.iloc[-1].at['Investment Returns']
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Investment Returns']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('{:,.0f} %'.format(y_max), xy=(x_max, y_max), xytext=(x_max-(x_max-x_min)*1/14, y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()

def mon_dep(isa):
    """
    Plot the monthly deposits over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Monthly Deposit'
    y_max = isa.iloc[-1].at['Monthly Deposit']
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Monthly Deposit']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('\xA3 {:,.0f}'.format(y_max), xy=(x_max, y_max), xytext=(x_max-(x_max-x_min)*1/8, y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()

def amo_dep(isa):
    """
    Plot the total amount deposited over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Amount Deposited'
    y_max = isa.iloc[-1].at['Amount Deposited']
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Amount Deposited']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('\xA3 {:,.0f}'.format(y_max), xy=(x_max, y_max), xytext=(x_max-(x_max-x_min)*1/8, y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()

def cha(isa):
    """
    Plot the monthly charges over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Charges'
    y_max = isa.iloc[-1].at['Charges']
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Charges']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('\xA3 {:,.2f}'.format(y_max), xy=(x_max, y_max), xytext=(x_max-(x_max-x_min)*0.15, y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()

def tot_cha(isa):
    """
    Plot the total charges over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Total Charges'
    y_max = isa.iloc[-1].at['Total Charges']
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Total Charges']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('\xA3 {:,.0f}'.format(y_max), xy=(x_max, y_max), xytext=(x_max-(x_max-x_min)*1/8, y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()

def tot_cha_pc(isa):
    """
    Plot the total charges as a percentage of the total value over time.
    
    Args:
        isa (DataFrame): Log of stats
    """
    x = 'Year'
    y = 'Total Charges (%)'
    y_max = isa['Total Charges (%)'].max() * 1.05
    x_max = isa.iloc[-1].at['Year']
    y_min = isa.iloc[0].at['Total Charges (%)']
    x_min = isa.iloc[0].at['Year']
    value_graph = sns.relplot(x=x, y=y, kind='line', data=isa, aspect=2)
    plt.annotate('{:,.2f}'.format(isa['Total Charges (%)'].max()), xy=(x_max, y_max), xytext=(isa.loc[isa['Total Charges (%)'] == isa['Total Charges (%)'].max()]['Year'], y_max))
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.grid(axis='y', linewidth=0.25)
    value_graph.fig.autofmt_xdate()


def compare_inv_ret(isa1, isa2):
    """
    Plot and compare the total value of 2 different investments over time.
    
    Args:
        isa1 (DataFrame): Log of first investment's stats
        isa2 (DataFrame): Log of second investment's stats
    """
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(x=isa1['Year'], 
                y=isa1['Investment Returns'],
                color='#ff0066',
                ax=ax)
    sns.lineplot(x=isa2['Year'], 
                y=isa2['Investment Returns'], 
                color='#00ffff',
                ax=ax)    
    ax.legend(['ISA 1', 'ISA 2'])
    #plt.xlim(0, 15)
    #plt.ylim(0, 12000)
    plt.grid(axis='y', linewidth=0.25)
    plt.show()

def compare_tot_val(isa1, isa2):
    """
    Plot and compare the total value of 2 different investments over time.
    
    Args:
        isa1 (DataFrame): Log of first investment's stats
        isa2 (DataFrame): Log of second investment's stats
    """
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(x=isa1['Year'], 
                y=isa1['Total Value'],
                color='#ff0066',
                ax=ax)
    sns.lineplot(x=isa2['Year'], 
                y=isa2['Total Value'], 
                color='#00ffff',
                ax=ax)    
    ax.legend(['ISA 1', 'ISA 2'])
    #plt.xlim(0, 15)
    #plt.ylim(0, 12000)
    plt.grid(axis='y', linewidth=0.25)
    plt.show()

def compare_tot_cha(isa1, isa2):
    """
    Plot and compare the total charges of 2 different investments over time.
    
    Args:
        isa1 (DataFrame): Log of first investment's stats
        isa2 (DataFrame): Log of second investment's stats
    """
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(x=isa1['Year'], 
                y=isa1['Total Charges'],
                color='#ff0066',
                ax=ax)
    sns.lineplot(x=isa2['Year'], 
                y=isa2['Total Charges'], 
                color='#00ffff',
                ax=ax)    
    ax.legend(['ISA 1', 'ISA 2'])
    #plt.xlim(0, 15)
    #plt.ylim(0, 12000)
    plt.grid(axis='y', linewidth=0.25)
    plt.show()

def compare_tot_cha_pc(isa1, isa2):
    """
    Plot and compare the total charges as a percentage of the total value of 2 different investments over time.
    
    Args:
        isa1 (DataFrame): Log of first investment's stats
        isa2 (DataFrame): Log of second investment's stats
    """
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(x=isa1['Year'], 
                y=isa1['Total Charges (%)'],
                color='#ff0066',
                ax=ax)
    sns.lineplot(x=isa2['Year'], 
                y=isa2['Total Charges (%)'], 
                color='#00ffff',
                ax=ax)
    idx = np.argwhere(np.diff(np.sign(isa2['Total Charges (%)'] - isa1['Total Charges (%)']))).flatten()        # Find the x-coordinate of the intersepts (use only first if 1 intersection, with second if 2, etc)
    plt.annotate('\xA3 {:,.0f}'.format(isa1.at[idx[0]-1, 'Total Value']), xy=(isa1.at[idx[0]-1, 'Year'], isa1.at[idx[0]-1, 'Total Charges (%)']), xytext=(isa1.at[idx[0]-1, 'Year'], isa1.at[idx[0]-1, 'Total Charges (%)']*7/8))
    #plt.annotate('\xA3 {:,.0f}'.format(isa1.at[idx[1]-1, 'Total Value']), xy=(isa1.at[idx[1]-1, 'Year'], isa1.at[idx[1]-1, 'Total Charges (%)']), xytext=(isa1.at[idx[1]-1, 'Year']*32/34, isa1.at[idx[1]-1, 'Total Charges (%)']*9/10))
    ax.legend(['ISA 1', 'ISA 2'])
    #plt.xlim(0, 15)
    #plt.ylim(0, 12000)
    plt.grid(axis='y', linewidth=0.25)
    plt.show()


def mix_inv():
    """
    Model the performance of switching between different investment types over time.
    
    Returns:
        total_value (float): final value of investments
        isa1 (DataFrame): log of stats for first period
        isa2 (DataFrame): log of stats for second period
        isa3 (DataFrame): log of stats for third period

    """
    period1 = 70
    period2 = 40*12
    #period3 = 40*12

    total_value, monthly_payment, total_months, total_platform_fee, total_deposited, isa1 = inv_model(
                                                platform_charge_250 = 0.0045,
                                                platform_charge_1 = 0.0025,
                                                platform_charge_2 = 0.001,
                                                platform_charge_over = 0,
                                                inv_return = 0.16,
                                                inv_ocr = 0.0012,
                                                initial_payment = 1000,
                                                monthly_payment = 200,
                                                total_deposited = 1000,
                                                inv_months = period1,
                                                total_months = 0,
                                                total_platform_fee = 0,
                                                salary_increase_rate = 0.04,
                                                inflation = 0.02,

                                                # Investment trusts only
                                                platform_charge_cap = None,
                                                trades = 12,
                                                trade_charge = 0,
                                                stamp_duty = 0.00)
    
    total_value, monthly_payment, total_months, total_platform_fee, total_deposited, isa2 = inv_model(
                                                platform_charge_250 = 0.0025,
                                                platform_charge_1 = 0.001,
                                                platform_charge_2 = 0.0005,
                                                platform_charge_over = 0,
                                                inv_return = 0.16,
                                                inv_ocr = 0.0012,
                                                initial_payment = total_value,
                                                monthly_payment = monthly_payment,
                                                total_deposited = total_deposited,
                                                inv_months = period2,
                                                total_months = total_months,
                                                total_platform_fee = total_platform_fee,
                                                platform_charge_cap = 30/12,
                                                trades = 12,
                                                trade_charge = 1.5,
                                                stamp_duty = 0.005,
                                                salary_increase_rate = 0.04,
                                                inflation = 0.02)

    #total_value, monthly_payment, total_months, total_platform_fee, total_deposited, isa3 = inv_model(
    #                                            inv_ocr = 0.01,
    #                                            salary_increase_rate = 0.04,
    #                                            initial_payment = total_value,
    #                                            monthly_payment = monthly_payment,
    #                                            total_deposited = total_deposited,
    #                                            inv_months = period3,
    #                                            total_months = total_months,
    #                                            total_platform_fee = total_platform_fee,
    #                                            platform_charge_cap = 0,
    #                                            trades = 0,
    #                                            stamp_duty = 0)

    return total_value, isa1, isa2, #isa3


total_value, monthly_payment, inv_months, total_platform_fee, total_deposited, isa = inv_model()
#total_value, isa1, isa2 = mix_inv()
print("\xA3 {:,.0f}".format(total_value))
#isa_mix = isa1.append(isa2, ignore_index = True)#.append(isa3, ignore_index = True)
