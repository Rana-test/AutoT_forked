from __future__ import print_function
from api_helper import ShoonyaApiPy
import os
import pyotp
import pandas as pd
from datetime import datetime, time, timezone
import time as sleep_time
import yfinance as yf
import yaml
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText
import pytz
import numpy as np
import math
### Upstox login 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyotp
import sys
# sys.path.append('/home/rana/trading/newAuto/AutoT/env/lib/python3.10/site-packages')
import upstox_client
import time
from upstox_client.rest import ApiException
from pprint import pprint
import requests
import pandas as pd
import logging
from stema_v1 import get_minute_data, run_hourly_trading_strategy, update_stema_tb
from zoneinfo import ZoneInfo

holiday_dict ={
    '2025-05-01':'2025-04-30',
    '2025-10-02':'2025-10-01',
    '2025-12-25':'2025-12-24',
}

logging.basicConfig(level=logging.INFO)

live=True
upstox_instruments = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz")
sender_email =''
receiver_email=''
email_password=''

def login_upstox(UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN):

    configuration = upstox_client.Configuration()
    UPSTOX_API_KEY="9bff3d90-0499-4435-9052-758d5cad6d15"
    UPSTOX_URL = f"https://api-v2.upstox.com/login/authorization/dialog?response_type=code&client_id={UPSTOX_API_KEY}&redirect_uri=https%3A%2F%2Fwww.google.com"
    UPSTOX_API_SECRET="yt53tk18dx"
    UPSTOX_MOB_NO="9731811400"
    UPSTOX_CLIENT_PASS="ONRG76QYCMY4FYLUHBCU6PCRAYCMYFB7"
    UPSTOX_CLIENT_PIN="182418"

    def wait_for_page_load(driver, timeout=30):
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script('return document.readyState') == 'complete')
        
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument("--headless") 
    driver = webdriver.Chrome(options=options)
    driver.get(UPSTOX_URL)
    wait_for_page_load(driver)
    username_input_xpath = '//*[@id="mobileNum"]'
    username_input_element = driver.find_element(By.XPATH, username_input_xpath)
    username_input_element.clear()
    username_input_element.send_keys(UPSTOX_MOB_NO)
    get_otp_button_xpath = '//*[@id="getOtp"]'
    get_otp_button_element = driver.find_element(By.XPATH, get_otp_button_xpath)
    get_otp_button_element.click()
    client_pass = pyotp.TOTP(UPSTOX_CLIENT_PASS).now()
    text_box = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "otpNum")))
    text_box.clear()
    text_box.send_keys(client_pass)
    wait = WebDriverWait(driver, 10)
    continue_button = wait.until(EC.element_to_be_clickable((By.ID, "continueBtn")))
    continue_button.click()
    # XPath for the pin input field
    text_box = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "pinCode")))
    text_box.clear()
    text_box.send_keys(UPSTOX_CLIENT_PIN)
    continue_button = wait.until(EC.element_to_be_clickable((By.ID, "pinContinueBtn")))
    continue_button.click()
    redirect_url = WebDriverWait(driver, 10).until(
        lambda d: "?code=" in d.current_url
    )
    # Retrieve the token from the URL
    token = driver.current_url.split("?code=")[1]

    url = 'https://api.upstox.com/v2/login/authorization/token'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'code': token,
        'client_id': UPSTOX_API_KEY,
        'client_secret': UPSTOX_API_SECRET,
        'redirect_uri': "https://www.google.com",
        'grant_type': 'authorization_code',
    }

    response = requests.post(url, headers=headers, data=data)

    print(response.status_code)
    print(response.json())
    access_token=response.json().get("access_token")

    # Configure OAuth2 access token for authorization: OAUTH2
    configuration = upstox_client.Configuration()
    configuration.access_token = access_token

    upstox_opt_api = upstox_client.OptionsApi(upstox_client.ApiClient(configuration))
    upstox_charge_api = upstox_client.ChargeApi(upstox_client.ApiClient(configuration))

    return upstox_client, upstox_opt_api, upstox_charge_api

def get_india_vix(api):
    return round(float(api.get_quotes(exchange="NSE", token=str(26017))['lp']),2)

def is_within_time_range():
    # Get the current UTC time
    now = datetime.now(ZoneInfo("Asia/Kolkata")).time()

    # Define the start and end times
    start_time = time(9, 15)  # 3:45 a.m. UTC
    end_time = time(15, 30)   # 10:00 a.m. UTC

    # Check if the current time is within the range
    return start_time <= now <= end_time

def login(userid, password, vendor_code, api_secret, imei, TOKEN):
    twoFA = pyotp.TOTP(TOKEN).now()
    api = ShoonyaApiPy()
    login_response = api.login(userid=userid, password=password, twoFA=twoFA, vendor_code=vendor_code, api_secret=api_secret, imei=imei)   
    if login_response['stat'] == 'Ok':
        print('Logged in sucessfully')
        return api
    else:
        print(f"Login failed: {login_response.get('emsg', 'Unknown error')}")
        print('Logged in failed')
        return None
    
def init_creds():
    global sender_email
    global receiver_email
    global email_password
    if os.path.exists('helpers/creds.yml'):
        try:
            with open('helpers/creds.yml', 'r') as file:
                data = yaml.safe_load(file)
                sender_email = data.get("EMAIL_USER")
                receiver_email = data.get("EMAIL_TO")
                email_password = data.get("EMAIL_PASS")
                TOKEN = data.get("TOKEN")
                userid=data.get("userid")
                password=data.get("password")
                vendor_code=data.get("vendor_code")
                api_secret=data.get("api_secret")
                imei=data.get("imei")
                UPSTOX_API_KEY=data.get("UPSTOX_API_KEY")
                UPSTOX_URL=data.get("UPSTOX_URL")
                UPSTOX_API_SECRET=data.get("UPSTOX_API_SECRET")
                UPSTOX_MOB_NO=data.get("UPSTOX_MOB_NO")
                UPSTOX_CLIENT_PASS=data.get("UPSTOX_CLIENT_PASS")
                UPSTOX_CLIENT_PIN=data.get("UPSTOX_CLIENT_PIN")

        except yaml.YAMLError as e:
            print("Error loading YAML file:", e)
    else:
        # Email configuration
        sender_email = os.getenv("EMAIL_USER")
        receiver_email = os.getenv("EMAIL_TO")
        email_password = os.getenv("EMAIL_PASS")
        TOKEN = os.getenv("TOKEN")
        userid=os.getenv("userid")
        password=os.getenv("password")
        vendor_code=os.getenv("vendor_code")
        api_secret=os.getenv("api_secret")
        imei=os.getenv("imei")
        UPSTOX_API_KEY=os.getenv("UPSTOX_API_KEY")
        UPSTOX_URL=os.getenv("UPSTOX_URL")
        UPSTOX_API_SECRET=os.getenv("UPSTOX_API_SECRET")
        UPSTOX_MOB_NO=os.getenv("UPSTOX_MOB_NO")
        UPSTOX_CLIENT_PASS=os.getenv("UPSTOX_CLIENT_PASS")
        UPSTOX_CLIENT_PIN=os.getenv("UPSTOX_CLIENT_PIN")
    
    return userid, password, vendor_code, api_secret, imei, TOKEN, sender_email, receiver_email, email_password, UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN

def is_within_timeframe(start, end):
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    logging.info(now)
    start_time = now.replace(hour=int(start.split(':')[0]), minute=int(start.split(':')[1]), second=0, microsecond=0)
    end_time = now.replace(hour=int(end.split(':')[0]), minute=int(end.split(':')[1]), second=0, microsecond=0)
    return start_time <= now <= end_time

def identify_session():
    # IST to UTC
    # 8:30 == 3:00
    # 9:00 ==  3:30
    # 9:15 == 3:45
    # 12:25 == 6:55
    # 12:27 == 6:57
    # 12:30 == 7:00
    # 15:30 == 10:00

    if is_within_timeframe("08:30", "12:27"):
        return {"session": "session1", "start_time": "08:30", "end_time": "12:27"}
    elif is_within_timeframe("12:30", "15:30"):
        return {"session": "session2","start_time": "12:30", "end_time": "15:30"}
    return None

def send_email(subject, body):
    global sender_email
    global receiver_email
    global email_password
    # Create email
    msg = MIMEMultipart()
    # Add body to email
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))
    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(sender_email, email_password)  # Login to your email account
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)  # Send the email
        print("Email sent successfully!")
        # time.sleep(180)
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

def send_email_plain(subject, body):
    global sender_email
    global receiver_email
    global email_password
    # Create email
    msg = MIMEMultipart()
    # Add body to email
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(sender_email, email_password)  # Login to your email account
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)  # Send the email
        print("Email sent successfully!")
        # time.sleep(180)
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

def dict_to_table_manual(data):
    """Converts a dictionary to a formatted table string without external libraries."""
    max_key_length = max(len(str(k)) for k in data.keys())
    max_value_length = max(len(str(v)) for v in data.values())

    table = f"{'Key'.ljust(max_key_length)} | {'Value'.ljust(max_value_length)}\n"
    table += "-" * (max_key_length + max_value_length + 3) + "\n"

    for k, v in data.items():
        table += f"{str(k).ljust(max_key_length)} | {str(v).ljust(max_value_length)}\n"

    return table

def insert_if_not_exists(df, new_record):
    match_columns = ['token', 'daybuyqty', 'daysellqty', 'cfbuyqty', 'cfsellqty']
    # Check if there is any existing record with the same values in the match columns
    exists = (df[match_columns] == new_record[match_columns].iloc[0]).all(axis=1).any()
    
    # Insert only if the record does not exist
    if not exists:
        df = pd.concat([df, new_record], ignore_index=True)
    
    return df

def write_to_trade_book(api):
    trade_csv = "trade_book.csv"
    dtype_map={}
    cols = ['trantype', 'tsym', 'qty', 'fillshares', 'flqty', 'flprc', 'avgprc', 'exch_tm', 'remarks', 'exchordid']
    
    for col in cols:
        dtype_map[col]=str

    # Load existing trade history or create an empty DataFrame
    if os.path.exists(trade_csv):
        trade_csv_df = pd.read_csv(trade_csv, dtype=str)
    else:
        trade_csv_df = pd.DataFrame(columns=cols)

    new_rec_df = pd.DataFrame(api.get_trade_book())
    if len(new_rec_df) > 0:
        new_rec_df = new_rec_df[['trantype', 'tsym', 'qty', 'fillshares', 'flqty', 'flprc', 'avgprc', 'exch_tm', 'remarks', 'exchordid']]
        # Ensure column order and types match by creating a copy before modification
        trade_csv_df = trade_csv_df.copy()

        # Convert all columns to string, ensuring NaN values don't cause warnings
        new_rec_df = new_rec_df.fillna("").astype(str)
        trade_csv_df = trade_csv_df.fillna("").astype(str)

        for col, dtype in dtype_map.items():
            if col in new_rec_df.columns:
                new_rec_df.loc[:, col] = new_rec_df[col].astype(dtype)  # Use .loc to avoid SettingWithCopyWarning
            if col in trade_csv_df.columns:
                trade_csv_df.loc[:, col] = trade_csv_df[col].astype(dtype)  # Use .loc to avoid SettingWithCopyWarning

        # Remove duplicates and keep only new records
        # trade_csv_df = insert_if_not_exists(trade_csv_df, new_rec_df)
        match_columns = ['exchordid']
        # Check if there is any existing record with the same values in the match columns
        exists = (trade_csv_df[match_columns] == new_rec_df[match_columns].iloc[0]).all(axis=1).any()
        
        # Insert only if the record does not exist
        if not exists:
            trade_csv_df = pd.concat([trade_csv_df, new_rec_df], ignore_index=True)

        # Save to CSV with fixed float format (2 decimal places)
        trade_csv_df.to_csv(trade_csv, index=False, float_format="%.2f")

    return trade_csv_df

def write_to_trade_history(trade_book_df):
    trade_hist = "trade_history.csv"
    dtype_map={}
    cols = ['stat', 'uid', 'actid', 'exch', 'tsym', 's_prdt_ali', 'prd', 'token',
       'instname', 'dname', 'frzqty', 'pp', 'ls', 'ti', 'mult', 'prcftr',
       'daybuyqty', 'daysellqty', 'daybuyamt', 'daybuyavgprc', 'daysellamt',
       'daysellavgprc', 'cfbuyqty', 'cfsellqty', 'cfbuyamt', 'cfbuyavgprc',
       'cfsellamt', 'cfsellavgprc', 'openbuyqty', 'opensellqty', 'openbuyamt',
       'openbuyavgprc', 'opensellamt', 'opensellavgprc', 'dayavgprc', 'netqty',
       'netavgprc', 'upldprc', 'netupldprc', 'lp', 'urmtom', 'bep',
       'totbuyamt', 'totsellamt', 'totbuyavgprc', 'totsellavgprc', 'rpnl',
       'PnL', 'type', 'sp', 'expiry', 'Days_to_Expiry', 'exit_breakeven_per',
       'exit_loss_per']
    
    for col in cols:
        dtype_map[col]=str

    # Load existing trade history or create an empty DataFrame
    if os.path.exists(trade_hist):
        trade_hist_df = pd.read_csv(trade_hist, dtype=str)
    else:
        trade_hist_df = pd.DataFrame(columns=trade_book_df.columns)

    # Ensure column order and types match by creating a copy before modification
    trade_book_df = trade_book_df.copy()
    trade_hist_df = trade_hist_df.copy()

    # Convert all columns to string, ensuring NaN values don't cause warnings
    trade_book_df = trade_book_df.fillna("").astype(str)
    trade_hist_df = trade_hist_df.fillna("").astype(str)

    for col, dtype in dtype_map.items():
        if col in trade_book_df.columns:
            trade_book_df.loc[:, col] = trade_book_df[col].astype(dtype)  # Use .loc to avoid SettingWithCopyWarning
        if col in trade_hist_df.columns:
            trade_hist_df.loc[:, col] = trade_hist_df[col].astype(dtype)  # Use .loc to avoid SettingWithCopyWarning

    # Remove duplicates and keep only new records
    trade_hist_df = insert_if_not_exists(trade_hist_df, trade_book_df)
    # Save to CSV with fixed float format (2 decimal places)
    trade_hist_df.to_csv(trade_hist, index=False, float_format="%.2f")

    return trade_hist_df

def get_positions(api):
    try:
        pos_df = pd.DataFrame(api.get_positions())
        pos_df = pos_df[(~pos_df['dname'].isna())]
        # pos_df = pos_df[(pos_df['netqty']!="0")] # Not needed
        # Identify any record that needs to be added to tradehistory.csv
        # Read tradehistory.csv and get tradehistory_df
        # Add the record to trade_history_df
        # Save trade_history_df to tradehistory.csv
        # Update pos_df with relevant values from trade_history_df
        pos_df["PnL"] = -1 * (pos_df["netupldprc"].astype(float) - pos_df["lp"].astype(float)) * pos_df["netqty"].astype(float)
        pos_df["totsellamt"] = pos_df["totsellamt"].astype(float)
        pos_df["netqty"] = pos_df["netqty"].astype(int)
        pos_df['type'] = pos_df['dname'].apply(lambda x: x.split()[3])
        pos_df['sp'] = pos_df['dname'].apply(lambda x: x.split()[2])
        pos_df['expiry'] = pos_df['dname'].apply(lambda x: x.split()[1])  # Extract expiry date
        pos_df['expiry'] = pd.to_datetime(pos_df['expiry'], format="%d%b%y")
        current_date = pd.Timestamp.today().normalize()
        pos_df['Days_to_Expiry'] = pos_df['expiry'].apply(lambda x: (x - current_date).days)
        # pos_df['exit_breakeven_per']= pos_df.apply(lambda x: exit_params[x['Days_to_Expiry']]['distance_from_breakeven'],axis=1)
        pos_df['exit_breakeven_per']="0"
        # pos_df['exit_loss_per']= pos_df.apply(lambda x: exit_params[x['Days_to_Expiry']]['loss_multiple'],axis=1)
        pos_df['exit_loss_per']=0.5
        return pos_df
    except Exception as e:
        return None

def calc_expected_move(index_price: float, vix: float, days: int) -> float:
    daily_volatility = (vix/100) / np.sqrt(365)  # Convert annualized VIX to daily volatility
    expected_move = index_price * daily_volatility * np.sqrt(days)
    return expected_move

def get_atm_iv(upstox_opt_api, expiry_date, current_index_price):
    strike_interval = 50
    remainder = math.fmod(current_index_price, strike_interval)
    if remainder > strike_interval / 2:
        atm_strike = math.ceil(current_index_price / strike_interval) * strike_interval
    else:
        atm_strike = math.floor(current_index_price / strike_interval) * strike_interval

    api_response = upstox_opt_api.get_put_call_option_chain(instrument_key="NSE_INDEX|Nifty 50", expiry_date=expiry_date)
    for sp in api_response.data:
        if sp.strike_price == atm_strike:
            return((float(sp.call_options.option_greeks.iv)+float(sp.put_options.option_greeks.iv))/2)
    return None   
global session_var_file, sess_var_df
session_var_file = "session_var.csv"
if os.path.exists(session_var_file):
    sess_var_df = pd.read_csv(session_var_file)
else:
    sess_var_df = pd.DataFrame(columns=['session_var', 'value'])
    var_init = pd.DataFrame([
        {'session_var': 'counter', 'value': 0},
        {'session_var': 'exit_confirm', 'value': 0},
        {'session_var': 'entry_confirm', 'value': 0},
        {'session_var': 'ce_short', 'value': 99999},
        {'session_var': 'ce_long', 'value': -99999},
    ])
    var_init = var_init.astype(sess_var_df.dtypes.to_dict(), errors='ignore')
    sess_var_df = pd.concat([sess_var_df, var_init], ignore_index=True)

def stop_loss_order(pos_df, api, live=False):
    for i,pos in pos_df.iterrows():
        # Exit only the loss making side
        # if pos["PnL"] < 1: #Exit complete leg
        tradingsymbol = pos["tsym"]  # Trading symbol of the position
        netqty = int(pos["netqty"])  # Net quantity of the position
        if netqty != 0:  # Ensure position exists
            transaction_type = "BUY" if netqty < 0 else "SELL"
            quantity = abs(netqty)  # Exit full position
            prd_type = 'M'
            exchange = 'NFO' 
            # disclosed_qty= 0
            price_type = 'MKT'
            price=0
            trigger_price = None
            retention='DAY'
            buy_sell = 'S' if netqty>0 else 'B'
            if live:
                ret = api.place_order(buy_or_sell=buy_sell, product_type=prd_type, exchange=exchange, tradingsymbol=tradingsymbol, quantity=quantity, discloseqty=quantity,price_type=price_type, price=price,trigger_price=trigger_price, retention=retention, remarks="STOP LOSS ORDER")
                # Check Stema_trade book and update status to CLOSED
                update_stema_tb(tradingsymbol, pos['type'])
            else:
                print(f'buy_or_sell=buy_sell, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks="STOP LOSS ORDER"')
            
            subject = f"STOP LOSS or PROFIT EXIT TRIGGERED for {tradingsymbol}"
            email_body = f"STOP LOSS or PROFIT EXIT TRIGGERED for {tradingsymbol}"
            send_email(subject, email_body)

def format_trade_metrics(metrics):
    total_pnl = metrics.get("Total_PNL", "N/A")
    india_vix = metrics.get("INDIA_VIX", "N/A")
    expiry_details = metrics.get("Expiry_Details", {})
    
    data = []
    for expiry, details in expiry_details.items():
        if "Error" in details:
            data.append([expiry, details["Error"], "", "", "", "", "", "", "", "", ""])
        else:
            data.append([
                expiry, details.get("PNL", "N/A"), details.get("CE_Strike", "N/A"),
                details.get("PE_Strike", "N/A"), details.get("Current_Index_Price", "N/A"),
                details.get("ATM_IV", "N/A"), details.get("Expected_Movement", "N/A"),
                details.get("Lower_Breakeven", "N/A"), details.get("Upper_Breakeven", "N/A"),
                details.get("Breakeven_Range", "N/A"), details.get("Breakeven_Range_Per", "N/A"),
                details.get("Near_Breakeven", "N/A"), details.get("Max_Profit", "N/A"), details.get("Max_Loss", "N/A"),
                details.get("Realized_Premium", "N/A")
            ])
    
    df = pd.DataFrame(data, columns=[
        "Expiry", "PNL", "CE Strike", "PE Strike", "Current Index Price", "ATM IV", "Expected Movement", "Lower Breakeven", 
        "Upper Breakeven", "Breakeven Range", "Breakeven %", "Near Breakeven", "Max Profit", "Max Loss", "Realized_Premium"
    ])
    
    table_html = df.to_html(index=False, border=1)
    
    email_body = f"""
    <html>
    <head>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body>
        <p><strong>Total PNL:</strong> {total_pnl}</p>
        <p><strong>INDIA VIX:</strong> {india_vix}</p>
        {table_html}
    </body>
    </html>
    """
    
    return email_body

# Removing Chandlier_exit_tv
# def monitor_trade(finvasia_api, upstox_opt_api, ce_short, ce_long):
def monitor_trade(finvasia_api, upstox_opt_api):
    global session_var_df
    global session_var_file
    logging.info("Getting positions")
    pos_df = get_positions(finvasia_api)
    # vix = get_india_vix(api)
    if pos_df is None:
        return {'get_pos Error':"Error getting position Info"} 
    total_pnl=0
    metrics = {"Total_PNL": total_pnl}
    expiry_metrics = {}
    current_index_price = float(finvasia_api.get_quotes(exchange="NSE", token=str(26000))['lp'])
    for expiry, group in pos_df.groupby("expiry"):
        # Removing Chandlier_exit_tv
        # Chandlier Exit
        # ce_exit=False
        order_type = group['type'].iloc[-1]
        # Removing Chandlier_exit_tv
        # if order_type =="CE" and current_index_price>ce_short:
        #     ce_exit = True
        # elif order_type =="PE" and current_index_price<ce_long:
        #     ce_exit = True
        expiry_date_str = expiry.strftime('%Y-%m-%d')
        atm_iv = get_atm_iv(upstox_opt_api, expiry_date_str, current_index_price)
        # expected_move = calc_expected_move(current_index_price, vix, group['Days_to_Expiry'].mean().astype(int))
        expected_move = calc_expected_move(current_index_price, atm_iv, group['Days_to_Expiry'].mean().astype(int)) #atm_iv based expected move
        current_pnl = float((-1 * (group["netupldprc"].astype(float)-group["lp"].astype(float)) * group["netqty"].astype(float)).sum())
        max_profit = float((-1 * group["netupldprc"].astype(float) * group["netqty"].astype(float)).sum())
        # total_premium_collected = (group["totsellamt"] / abs(group["netqty"])).sum()
        current_premium = float((group["netupldprc"].astype(float) * group["netqty"].astype(float)).sum())*-1
        current_qty =  group["netqty"].astype(float).sum()*-1
        # Record to insert is any with netqty = 0
        rec_ins = group[group['netqty'].astype(int)==0]
        if not rec_ins.empty:
            #Insert record into tradebook_df if it already doesn't exists
            trade_hist_df = write_to_trade_history(rec_ins)
        else:
            trade_hist_df = pd.read_csv("trade_history.csv", dtype=str)
        #Read tradebook_df for the given expiry to retrive rec_ins
        # Calculate premium collected
        
        trade_hist_df = trade_hist_df[trade_hist_df['expiry']==expiry_date_str]
        # realized_qty = float(((trade_hist_df['daybuyqty'].astype(int)+trade_hist_df['cfsellqty'].astype(int))/2).sum())
        # realized_premium = float((trade_hist_df['upldprc'].astype(float)-trade_hist_df['totbuyavgprc'].astype(float)).sum())*realized_qty
        act_realized_premium = (trade_hist_df['upldprc'].astype(float)*trade_hist_df['cfsellqty'].astype(int)-trade_hist_df['totbuyavgprc'].astype(float)*trade_hist_df['daybuyqty'].astype(int)).sum()
        realized_premium=0
        # if current_qty!=0:
        #     total_premium_collected_per_option = (current_premium + realized_premium) /current_qty
        # else:
        #     total_premium_collected_per_option=0
        # # current_pnl+=realized_premium # Realized premium is always 0?
        # # max_profit+=realized_premium # Realized premium is always 0?

        ce_rows = group[group["type"] == "CE"]
        pe_rows = group[group["type"] == "PE"]
        
        # if not ce_rows.empty and not pe_rows.empty:
        stop_loss_per = group['exit_loss_per'].mean().astype(float)
        max_loss = float(-1 * stop_loss_per * (max_profit-realized_premium))
        if ce_rows.empty:
            ce_strike = 0
            upper_breakeven=999999
        else:
            # ce_breakeven_factor = current_qty/(-1*ce_rows["netqty"].sum())
            # ce_strike = float((ce_rows["sp"].astype(float) * ce_rows["netqty"].abs()).sum() / ce_rows["netqty"].abs().sum())
            ce_strike = ce_rows["sp"].astype(float).min()
            # upper_breakeven = float(ce_strike + total_premium_collected_per_option*ce_breakeven_factor - current_index_price * ce_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            # upper_breakeven = float(ce_strike - current_index_price * ce_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            # upper_breakeven=float(ce_strike - expected_move + total_premium_collected_per_option)
            upper_breakeven=float(ce_strike)

        if pe_rows.empty:
            pe_strike = 0
            lower_breakeven = 0
        else:
            # pe_breakeven_factor = current_qty/(-1*pe_rows["netqty"].sum())
            # pe_strike = float((pe_rows["sp"].astype(float) * pe_rows["netqty"].abs()).sum() / pe_rows["netqty"].abs().sum())
            pe_strike = pe_rows["sp"].astype(float).max()
            # lower_breakeven = float(pe_strike - total_premium_collected_per_option*pe_breakeven_factor + current_index_price * pe_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            # lower_breakeven = float(pe_strike + current_index_price * pe_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            # lower_breakeven = float(pe_strike + expected_move-total_premium_collected_per_option)
            lower_breakeven = float(pe_strike)

        if ce_strike!=0 and pe_strike!=0:
            breakeven_range = upper_breakeven - lower_breakeven
            near_breakeven = min(100 * (current_index_price - lower_breakeven) / current_index_price,  
                                100 * (upper_breakeven - current_index_price) / current_index_price)
        elif ce_strike!=0:
            breakeven_range = upper_breakeven - current_index_price
            near_breakeven = 100 * (breakeven_range) / current_index_price
        elif pe_strike!=0:
            breakeven_range = current_index_price - lower_breakeven
            near_breakeven = 100 * (breakeven_range) / current_index_price
        else:
            breakeven_range = 0
            near_breakeven = 0

        expiry_metrics[expiry] = {
            "PNL": round(current_pnl, 2),
            "CE_Strike": round(ce_strike, 2),
            "PE_Strike": round(pe_strike, 2),
            "Current_Index_Price": current_index_price,
            "ATM_IV": round(atm_iv, 2),
            "Expected_Movement": round(expected_move, 2),
            "Lower_Breakeven": round(lower_breakeven, 2),
            "Upper_Breakeven": round(upper_breakeven, 2),
            "Breakeven_Range": round(breakeven_range, 2),
            "Breakeven_Range_Per": round(100 * breakeven_range / current_index_price, 2),
            "Near_Breakeven": round(near_breakeven, 2),
            "Max_Profit": round(max_profit, 2),
            "Max_Loss": round(max_loss, 2),
            "Realized_Premium": round(act_realized_premium, 2),
        }
        total_pnl+=current_pnl

        # Removing Chandlier_exit_tv
        # stop_loss_condition = ce_exit or ((current_index_price < lower_breakeven or current_index_price > upper_breakeven)) or current_pnl < max_loss or (current_pnl > 0.90 * max_profit)
        stop_loss_condition = ((current_index_price < lower_breakeven or current_index_price > upper_breakeven)) or current_pnl < max_loss or (current_pnl > 0.90 * max_profit)

        if stop_loss_condition and (current_pnl < 0.90 * max_profit):
            stop_loss_order(group, finvasia_api, live=live)
            expiry_metrics[expiry] = {
            "PNL": round(current_pnl, 2),
            "CE_Strike": round(ce_strike, 2),
            "PE_Strike": round(pe_strike, 2),
            "Current_Index_Price": current_index_price,
            "ATM_IV": round(atm_iv, 2),
            "Expected_Movement": round(expected_move, 2),
            "Lower_Breakeven": "STOP_LOSS",
            "Upper_Breakeven": "STOP_LOSS",
            "Breakeven_Range": "STOP_LOSS",
            "Breakeven_Range_Per": "STOP_LOSS",
            "Near_Breakeven": round(near_breakeven, 2),
            "Max_Profit": round(max_profit, 2),
            "Max_Loss": round(max_loss, 2),
            "Realized_Premium": round(act_realized_premium, 2),
        }
        elif stop_loss_condition and (current_pnl > 0.90 * max_profit):
            stop_loss_order(group, finvasia_api, live=live)
            expiry_metrics[expiry] = {
            "PNL": round(current_pnl, 2),
            "CE_Strike": round(ce_strike, 2),
            "PE_Strike": round(pe_strike, 2),
            "Current_Index_Price": current_index_price,
            "ATM_IV": round(atm_iv, 2),
            "Expected_Movement": round(expected_move, 2),
            "Lower_Breakeven": "PROFIT_EXIT",
            "Upper_Breakeven": "PROFIT_EXIT",
            "Breakeven_Range": "PROFIT_EXIT",
            "Breakeven_Range_Per": "PROFIT_EXIT",
            "Near_Breakeven": round(near_breakeven, 2),
            "Max_Profit": round(max_profit, 2),
            "Max_Loss": round(max_loss, 2),
            "Realized_Premium": round(act_realized_premium, 2),
        }

    metrics["Expiry_Details"] = expiry_metrics
    metrics["Total_PNL"] = round(total_pnl,2)
    trade_hist_df = pd.read_csv("trade_history.csv", dtype=str)  
    return metrics, float(trade_hist_df['rpnl'].astype(float).sum())

def main():
    global session_var_file, sess_var_df, live
    logging.info("Checking if today is a holiday")
    dt = str(datetime.now(ZoneInfo("Asia/Kolkata")).date())
    if dt in holiday_dict:
        logging.info("Exiting since today is a holiday")
        # Debug
        exit(0)
    logging.info("Inside Main")
    session = identify_session()
    logging.info(f"Identified Session: {session}")
    if not session:
        print("No active trading session.")
        return
    
    # Login
    userid, password, vendor_code, api_secret, imei, TOKEN, sender_email, receiver_email, email_password, UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN = init_creds()
    api = login(userid, password, vendor_code, api_secret, imei, TOKEN)
    upstox_client, upstox_opt_api, upstox_charge_api = login_upstox(UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN)
    logging.info(f"Logged into APIs")
    while is_within_timeframe("08:30", "09:15"):
        print("Initializing")
        logging.info("Initializing")
        sleep_time.sleep(60)

    write_to_trade_book(api)
    counter = 0
    exit_confirm = sess_var_df[sess_var_df['session_var'] == 'exit_confirm']['value'].iloc[0]
    entry_confirm = sess_var_df[sess_var_df['session_var'] == 'entry_confirm']['value'].iloc[0]
    # Removing Chandlier_exit_tv
    # ce_short = sess_var_df[sess_var_df['session_var'] == 'ce_short']['value'].iloc[0]
    # ce_long = sess_var_df[sess_var_df['session_var'] == 'ce_long']['value'].iloc[0]
    logging.info(f"Loaded session variables: {sess_var_df}")
    # Start Monitoring
    while is_within_timeframe(session.get('start_time'), session.get('end_time')):
        logging.info(f"Monitoring Trade")
        # Removing Chandlier_exit_tv
        # metrics, total_profit = monitor_trade(api, upstox_opt_api, ce_short, ce_long)
        metrics, total_profit = monitor_trade(api, upstox_opt_api)
        if metrics =="STOP_LOSS":
            send_email("STOP LOSS HIT - QUIT", "STOP LOSS HIT")
        else:
        #     subject = f"FINVASIA: MTM:{metrics['Total_PNL']} | NEAR_BE:{metrics['Near_Breakeven']} | RANGE:{metrics['Breakeven_Range_Per']}| MAX_PROFIT:{metrics['Max_Profit']} | MAX_LOSS: {metrics['Max_Loss']}"
            if counter % 10 == 0:
                logging.info(f"Senidng status mail")
                subject = "FINVASIA STATUS"
                metrics["INDIA_VIX"] = get_india_vix(api)
                email_body = format_trade_metrics(metrics)
                send_email(subject, email_body)
            if counter % 15 ==0:
                # subject, email_body = once_an_hour(api, expiry, upstox_opt_api)
                # send_email_plain(subject, email_body)
                stema_min_df = get_minute_data(api,now=None)
                logging.info(f"Got historical data")
                # Removing Chandlier_exit_tv
                # return_msgs, entry_confirm, exit_confirm, ce_short, ce_long = run_hourly_trading_strategy(live, api, upstox_opt_api, upstox_charge_api, upstox_instruments, stema_min_df, entry_confirm, exit_confirm, total_profit,current_time=None )
                return_msgs, entry_confirm, exit_confirm = run_hourly_trading_strategy(live, api, upstox_opt_api, upstox_charge_api, upstox_instruments, stema_min_df, entry_confirm, exit_confirm, total_profit,current_time=None )
                print(f'Number of email messages: {len(return_msgs)}')
                for msg in return_msgs:
                    send_email_plain(msg['subject'], msg['body'])
            counter+=1
        sleep_time.sleep(60)
  
    # Logout
    logging.info(f"Saving session variables")
    sess_var_df.loc[sess_var_df['session_var']=='counter','value']=counter
    sess_var_df.loc[sess_var_df['session_var']=='exit_confirm','value']=exit_confirm
    sess_var_df.loc[sess_var_df['session_var']=='entry_confirm','value']=entry_confirm
    # Removing Chandlier_exit_tv
    # sess_var_df.loc[sess_var_df['session_var']=='ce_short','value']=ce_short
    # sess_var_df.loc[sess_var_df['session_var']=='ce_long','value']=ce_long
    sess_var_df.to_csv(session_var_file, index=False)
    write_to_trade_book(api)
    api.logout()

if __name__ =="__main__":
    main()