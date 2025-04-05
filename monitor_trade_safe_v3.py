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

live=True
# times=1.75
# stop_loss_per=0.5
# exit_params = {
#     day: {"distance_from_breakeven": dist, "loss_multiple": profit}
#     for day, dist, profit in [
#         (30, 3.28, 0.00),
#         (29, 3.05, 0.00),
#         (28, 2.83, 0.00),
#         (27, 2.61, 0.00),
#         (26, 2.41, 0.00),
#         (25, 2.21, 0.02),
#         (24, 2.02, 0.15),
#         (23, 1.84, 0.27),
#         (22, 1.66, 0.39),
#         (21, 1.50, 0.50),
#         (20, 1.34, 0.61),
#         (19, 1.20, 0.72),
#         (18, 1.06, 0.82),
#         (17, 0.93, 0.92),
#         (16, 0.81, 1.01),
#         (15, 0.69, 1.10),
#         (14, 0.59, 1.01),
#         (13, 0.49, 0.92),
#         (12, 0.40, 0.82),
#         (11, 0.32, 0.72),
#         (10, 0.25, 0.61),
#         (9, 0.19, 0.50),
#         (8, 0.13, 0.39),
#         (7, 0.09, 0.27),
#         (6, 0.05, 0.15),
#         (5, 0.02, 0.02),
#         (4, 0.00, 0.00),
#         (3, 0.00, 0.00),
#         (2, 0.00, 0.00),
#         (1, 0.00, 0.00),
#         (0, 0.00, 0.00),
#     ]
# }
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

    return upstox_client, upstox_opt_api

def calc_expected_move(index_price: float, vix: float, days: int) -> float:
    daily_volatility = (vix/100) / np.sqrt(365)  # Convert annualized VIX to daily volatility
    expected_move = index_price * daily_volatility * np.sqrt(days)
    return expected_move

def stop_loss_order(pos_df, api, live, sender_email, receiver_email, email_password):
    for i,pos in pos_df.iterrows():
        # Exit only the loss making side
        if pos["PnL"] < 1:
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
                if live:
                    ret = api.place_order(buy_or_sell="B", product_type=prd_type, exchange=exchange, tradingsymbol=tradingsymbol, quantity=quantity, discloseqty=quantity,price_type=price_type, price=price,trigger_price=trigger_price, retention=retention, remarks="STOP LOSS ORDER")
                else:
                    print(f'buy_or_sell="B", product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks="STOP LOSS ORDER"')
                
                subject = f"STOP LOSS TRIGGERED for {tradingsymbol} at {price}."
                email_body = f"STOP LOSS TRIGGERED for {tradingsymbol} at {price}."
                send_email(sender_email, receiver_email, email_password, subject, email_body)

def get_india_vix(api):
    return round(float(api.get_quotes(exchange="NSE", token=str(26017))['lp']),2)

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
    

def is_within_time_range():
    # Get the current UTC time
    now = datetime.now(pytz.utc).time()

    # Define the start and end times
    start_time = time(3, 45)  # 3:45 a.m. UTC
    end_time = time(21, 00)   # 10:00 a.m. UTC

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
    now = datetime.now(timezone.utc) 
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

    if is_within_timeframe("03:00", "06:55"):
        return {"session": "session1", "start_time": "03:45", "end_time": "06:57"}
    elif is_within_timeframe("07:00", "20:00"):
        return {"session": "session2","start_time": "07:00", "end_time": "20:00"}
    return None

def send_email(sender_email, receiver_email, email_password, subject, body):
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

def send_email_plain(sender_email, receiver_email, email_password, subject, body):
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

def monitor_trade(api, upstox_opt_api, sender_email, receiver_email, email_password):
    pos_df = get_positions(api)
    # vix = get_india_vix(api)
    if pos_df is None:
        return {'get_pos Error':"Error getting position Info"} 
    total_pnl=0
    metrics = {"Total_PNL": total_pnl}
    expiry_metrics = {}
    current_index_price = float(api.get_quotes(exchange="NSE", token=str(26000))['lp'])
    
    for expiry, group in pos_df.groupby("expiry"):
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

        total_premium_collected_per_option = (current_premium + realized_premium) /current_qty
        current_pnl+=realized_premium
        max_profit+=realized_premium

        ce_rows = group[group["type"] == "CE"]
        pe_rows = group[group["type"] == "PE"]
        
        # if not ce_rows.empty and not pe_rows.empty:
        stop_loss_per = group['exit_loss_per'].mean().astype(float)
        max_loss = float(-1 * stop_loss_per * (max_profit-realized_premium))
        if ce_rows.empty:
            ce_strike = 0
            upper_breakeven=999999
        else:
            ce_breakeven_factor = current_qty/(-1*ce_rows["netqty"].sum())
            ce_strike = float((ce_rows["sp"].astype(float) * ce_rows["netqty"].abs()).sum() / ce_rows["netqty"].abs().sum())
            # upper_breakeven = float(ce_strike + total_premium_collected_per_option*ce_breakeven_factor - current_index_price * ce_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            # upper_breakeven = float(ce_strike - current_index_price * ce_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            upper_breakeven=float(ce_strike - expected_move + total_premium_collected_per_option)

        if pe_rows.empty:
            pe_strike = 0
            lower_breakeven = 0
        else:
            pe_breakeven_factor = current_qty/(-1*pe_rows["netqty"].sum())
            pe_strike = float((pe_rows["sp"].astype(float) * pe_rows["netqty"].abs()).sum() / pe_rows["netqty"].abs().sum())
            # lower_breakeven = float(pe_strike - total_premium_collected_per_option*pe_breakeven_factor + current_index_price * pe_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            # lower_breakeven = float(pe_strike + current_index_price * pe_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            lower_breakeven = float(pe_strike + expected_move-total_premium_collected_per_option)

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

        stop_loss_condition = (current_index_price < lower_breakeven or current_index_price > upper_breakeven) and current_pnl < max_loss
        
        if stop_loss_condition:
            stop_loss_order(group, api, sender_email, receiver_email, email_password,live)
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

    metrics["Expiry_Details"] = expiry_metrics
    metrics["Total_PNL"] = round(total_pnl,2)
      
    return metrics

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

#####################################DIRECTIONAL#################
import pandas as pd
import numpy as np
from api_helper import get_time
from datetime import datetime, timedelta
import math
import calendar
import ta
import yfinance as yf

month_mapping = {
    '1': 'JAN', '2': 'FEB', '3': 'MAR', '4': 'APR', '5': 'MAY', '6': 'JUN',
    '7': 'JUL', '8': 'AUG', '9': 'SEP', 'O': 'OCT', 'N': 'NOV', 'D': 'DEC'
}

def round_to_previous_15_45(dt):
    hour = dt.hour
    minute = dt.minute

    # Only apply rounding if time is between 9:15 and 15:30
    start = dt.replace(hour=9, minute=15, second=0, microsecond=0)
    end = dt.replace(hour=15, minute=30, second=0, microsecond=0)

    if dt < start:
        return start
    if dt > end:
        return end

    # Round down to previous 15 or 45 minute
    if minute >= 45:
        rounded_minute = 45
    elif minute >= 15:
        rounded_minute = 15
    else:
        # If before 15 past, round down to the previous hour at 45
        dt -= timedelta(hours=1)
        rounded_minute = 45

    return dt.replace(minute=rounded_minute, second=0, microsecond=0)


def get_data(api):
    nifty_token = '26000'  # NSE|26000 is the Nifty 50 index
    
    # Get current time
    now = datetime.now()

    # Round current time
    rounded_now = round_to_previous_15_45(now)

    # Time 5 days ago
    rounded_ten_days_ago = rounded_now - timedelta(days=30)

    # Desired format
    fmt = "%d-%m-%Y %H:%M:%S"

    start_secs = get_time(rounded_ten_days_ago.strftime(fmt))  # dd-mm-YYYY HH:MM:SS
    end_secs   = get_time(rounded_now.strftime(fmt))

    bars = api.get_time_price_series(
        exchange  = 'NSE',
        token     = nifty_token,
        starttime = int(start_secs),
        endtime   = int(end_secs),
        interval  = 60          # 60-minute candles
    )

    df = pd.DataFrame.from_dict(bars)
    df.rename(columns={
        'into': 'open',
        'inth': 'high',
        'intl': 'low',
        'intc': 'close'
    }, inplace=True)
    df['datetime'] = pd.to_datetime(df['time'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    df.sort_index(ascending=True, inplace=True)
    return df

def compute_EMA20(df):
    ema_period = 20
    df['EMA20'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    return df

def compute_supertrend(df, atr_period=10, multiplier=3.5):
    high = df['high']
    low = df['low']
    close = df['close']
    df['tr'] = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    hl2 = (high + low) / 2
    df['basic_upper'] = hl2 + multiplier * df['atr']
    df['basic_lower'] = hl2 - multiplier * df['atr']
    # Initialize with object dtype to support NaN and bool
    df['final_upper'] = np.nan
    df['final_lower'] = np.nan
    df['supertrend'] = np.nan
    df['st_up'] = pd.Series(dtype='object')  # Allows NaN and bool
    first_valid_idx = df['atr'].first_valid_index()
    if first_valid_idx is not None:
        i = df.index.get_loc(first_valid_idx)
        df.at[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
        df.at[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
        hl2_i = (high.iloc[i] + low.iloc[i]) / 2
        # Initial direction: match expected downtrend (23781) if close < HL2
        if close.iloc[i] < hl2_i:
            df.at[df.index[i], 'st_up'] = False
            df.at[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
        else:
            df.at[df.index[i], 'st_up'] = True
            df.at[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
        for j in range(i + 1, len(df)):
            prev_j = j - 1
            if df['basic_upper'].iloc[j] < df['final_upper'].iloc[prev_j] or close.iloc[prev_j] > df['final_upper'].iloc[prev_j]:
                df.at[df.index[j], 'final_upper'] = df['basic_upper'].iloc[j]
            else:
                df.at[df.index[j], 'final_upper'] = df['final_upper'].iloc[prev_j]
            if df['basic_lower'].iloc[j] > df['final_lower'].iloc[prev_j] or close.iloc[prev_j] < df['final_lower'].iloc[prev_j]:
                df.at[df.index[j], 'final_lower'] = df['basic_lower'].iloc[j]
            else:
                df.at[df.index[j], 'final_lower'] = df['final_lower'].iloc[prev_j]
            if df['st_up'].iloc[prev_j]:
                if close.iloc[j] < df['supertrend'].iloc[prev_j]:
                    df.at[df.index[j], 'st_up'] = False
                    df.at[df.index[j], 'supertrend'] = df['final_upper'].iloc[j]
                else:
                    df.at[df.index[j], 'st_up'] = True
                    df.at[df.index[j], 'supertrend'] = df['final_lower'].iloc[j]
            else:
                if close.iloc[j] > df['supertrend'].iloc[prev_j]:
                    df.at[df.index[j], 'st_up'] = True
                    df.at[df.index[j], 'supertrend'] = df['final_lower'].iloc[j]
                else:
                    df.at[df.index[j], 'st_up'] = False
                    df.at[df.index[j], 'supertrend'] = df['final_upper'].iloc[j]
    df.drop(columns=['tr', 'basic_upper', 'basic_lower'], inplace=True)
    return df

def get_signal(df):
    df['position'] = 0
    df['signal'] = pd.Series(dtype='object')
    for i in range(1, len(df)):
        prev_pos = df['position'].iat[i - 1]
        if pd.isna(df['supertrend'].iat[i]):
            df.at[df.index[i], 'position'] = prev_pos
            continue
        st_up = df['st_up'].iat[i]
        close_i = df['close'].iat[i]
        ema_i = df['EMA20'].iat[i]
        low_i = df['low'].iat[i]
        high_i = df['high'].iat[i]
        if prev_pos == -1 and not st_up:  # Put exit
            df.at[df.index[i], 'signal'] = 'PUT_EXIT'
            df.at[df.index[i], 'position'] = 0
        elif prev_pos == 1 and st_up:  # Call exit
            df.at[df.index[i], 'signal'] = 'CALL_EXIT'
            df.at[df.index[i], 'position'] = 0
        elif prev_pos == 0:
            if low_i > ema_i and st_up:  # Put entry
                df.at[df.index[i], 'signal'] = 'PUT_ENTRY'
                df.at[df.index[i], 'position'] = -1
            elif high_i < ema_i and not st_up:  # Call entry
                df.at[df.index[i], 'signal'] = 'CALL_ENTRY'
                df.at[df.index[i], 'position'] = 1
        else:
            df.at[df.index[i], 'position'] = prev_pos
    
    # Get the latest signal and its timestamp
    signal_series = df['signal'].dropna()
    if not signal_series.empty:
        latest_signal = signal_series.iloc[-1]  # Last non-null signal
        latest_timestamp = signal_series.index[-1]  # Timestamp of latest signal
        # Current time (April 5, 2025, per system date)
        current_time = pd.Timestamp('2025-04-05 00:00:00')  # Adjust time as needed
        time_diff = (current_time - latest_timestamp).total_seconds() / 60  # Minutes
        return latest_signal, time_diff
    else:
        return None, None  # No signals found

def get_nifty_rsi():
    # Fetch NIFTY 50 data (^NSEI) for the last 20 days to ensure enough data for RSI calculation
    nifty = yf.download("^NSEI", period="6wk", interval="1d")

    # Ensure 'Close' is a Pandas Series (not a DataFrame)
    close_prices = nifty["Close"].dropna().squeeze()

    # Calculate RSI using the ta library (14-period RSI)
    rsi_indicator = ta.momentum.RSIIndicator(close_prices, window=14)
    nifty["RSI"] = rsi_indicator.rsi()

    # Get the latest RSI value
    latest_rsi = nifty["RSI"].iloc[-1]

    return round(latest_rsi,2)

def get_india_vix(api):
    return round(float(api.get_quotes(exchange="NSE", token=str(26017))['lp']),2)


def find_last_thursday(year, month):
    """Finds the last Thursday of a given month and year."""
    last_day = datetime(year, month, 1).replace(day=calendar.monthrange(year, month)[1])
    while last_day.weekday() != 3:  # Thursday is weekday 3
        last_day -= timedelta(days=1)
    return last_day.day

def get_next_thursday_between_4_and_12_days():
    today = datetime.today().date()

    for i in range(5, 12):  # Start from 5 (to be > 4) up to 11 (< 12)
        target_date = today + timedelta(days=i)
        if target_date.weekday() == 3:  # 0=Monday, ..., 3=Thursday
            return target_date.strftime('%Y-%m-%d')
    
    return None  # If no Thursday found in range


def convert_option_symbol(input_symbol):
    # Ensure input is valid
    if not isinstance(input_symbol, str) or len(input_symbol) < 12:
        raise ValueError("Invalid symbol format. Expected format: NIFTY2530622300PE or NIFTY25FEB22450PE")

    # Extract the underlying index (e.g., NIFTY)
    index = input_symbol[:5]

    # Extract the year
    year_prefix = input_symbol[5:7]

    # Determine if it's a weekly or monthly expiry
    remaining_part = input_symbol[7:-2]  # Excludes index, year, and option type

    # Check if the month part is in the three-letter format (e.g., FEB) for monthly options
    if remaining_part[:3].isalpha():
        # Monthly expiry format: NIFTY25FEB22450PE
        month_abbr = remaining_part[:3].upper()
        strike_price = remaining_part[3:]
        expiry_day = None  # Monthly expiry day to be determined separately
    else:
        # Weekly expiry format: NIFTY2530622300PE (where "306" means 6th March)
        month_code = remaining_part[0]  # Could be a single letter (O, N, D) or a digit (1-9)
        day = remaining_part[1:3]  # Extract the day (e.g., '06')
        strike_price = remaining_part[3:]

        # Convert month code to full month abbreviation
        if month_code in month_mapping:
            month_abbr = month_mapping[month_code]
        else:
            raise ValueError(f"Invalid month code: {month_code}")

        expiry_day = day  # Weekly expiry has an explicit day

    # Determine the final expiry format
    if expiry_day:
        # Weekly expiry format: NIFTY06MAR25P22300
        new_symbol = f"{index}{expiry_day}{month_abbr}{year_prefix}{'P' if input_symbol[-2:] == 'PE' else 'C'}{strike_price}"
    else:
        # Monthly expiry format: Find the last Thursday of the month
        expiry_year = int(f"20{year_prefix}")
        expiry_month = datetime.strptime(month_abbr, "%b").month
        expiry_day = find_last_thursday(expiry_year, expiry_month)  # Function to get last Thursday

        new_symbol = f"{index}{expiry_day}{month_abbr}{year_prefix}{'P' if input_symbol[-2:] == 'PE' else 'C'}{strike_price}"

    return new_symbol

def get_option_chain(upstox_opt_api, instrument, expiry):
    # option_chain = upstox_opt_api.get_option_chain(symbol=instrument, expiry_date=expiry)
    option_chain = upstox_opt_api.get_put_call_option_chain(instrument_key=instrument,expiry_date=expiry)
    return option_chain.data

def get_nearest_delta_options(option_chain_data, upstox_instruments, delta):
    call_symbol = None
    put_symbol = None
    min_call_diff = float("inf")
    min_put_diff = float("inf")

    for option in option_chain_data:
        # Access call and put options using dot notation
        call_option = option.call_options
        put_option = option.put_options

        # Process call options
        if call_option and call_option.option_greeks:
            call_symb = upstox_instruments[upstox_instruments['instrument_key']==call_option.instrument_key].tradingsymbol.values[0]
            call_delta = call_option.option_greeks.delta
            call_oi = float(call_option.market_data.oi)
            if call_delta is not None and abs(call_delta - delta) < min_call_diff and call_symb[-4:-2] == '00' and call_oi>10000:
                min_call_diff = abs(call_delta - delta)
                call_symbol = call_option.instrument_key
                upstox_ce_ltp = call_option.market_data.ltp
                co = call_option

        # Process put options
        if put_option and put_option.option_greeks:
            put_symb = upstox_instruments[upstox_instruments['instrument_key']==put_option.instrument_key].tradingsymbol.values[0]
            put_delta = put_option.option_greeks.delta
            put_oi = float(put_option.market_data.oi) if put_option.market_data.oi is not None else 0
            if put_delta is not None and abs(put_delta + delta) < min_put_diff and put_symb[-4:-2] == '00' and put_oi>10000:
                min_put_diff = abs(put_delta + delta)
                put_symbol = put_option.instrument_key
                upstox_pe_ltp = put_option.market_data.ltp
                po = put_option

    call_bid = float(co.market_data.bid_price)
    call_ask = float(co.market_data.ask_price)
    call_bid_ask = call_ask-call_bid
    put_bid = float(po.market_data.bid_price)
    put_ask = float(po.market_data.ask_price)
    put_bid_ask = put_ask-put_bid
    return call_symbol, put_symbol, upstox_ce_ltp, upstox_pe_ltp, po, co , co.market_data.oi, po.market_data.oi, co.option_greeks.delta, po.option_greeks.delta, call_bid_ask, put_bid_ask

def get_positions_directional(finvasia_api, instrument, expiry,trade_qty,upstox_instruments, delta):
    SPAN_Expiry = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d-%b-%Y").upper()
    trade_details={}
    option_chain = get_option_chain(instrument, expiry)
    upstox_ce_instrument_key, upstox_pe_instrument_key, upstox_ce_ltp, upstox_pe_ltp,  po, co, call_oi, put_oi, call_delta, put_delta, call_bid_ask, put_bid_ask = get_nearest_delta_options(option_chain,upstox_instruments,delta)
    trade_details['call_oi']=call_oi
    trade_details['put_oi']=put_oi
    trade_details['call_delta']=call_delta*100
    trade_details['put_delta']=put_delta*100
    trade_details['call_bid_ask']=call_bid_ask
    trade_details['put_bid_ask']=put_bid_ask
    upstox_ce_symbol = upstox_instruments[upstox_instruments['instrument_key']==upstox_ce_instrument_key]['tradingsymbol'].values[0]
    # trade_details['upstox_ce'] = upstox_ce_symbol
    upstox_pe_symbol = upstox_instruments[upstox_instruments['instrument_key']==upstox_pe_instrument_key]['tradingsymbol'].values[0]
    # trade_details['upstox_pe'] = upstox_pe_symbol
    instruments = []
    instruments.append({"instrument_key": upstox_ce_instrument_key, "quantity": trade_qty, "transaction_type": "SELL", "product": "D", "price": upstox_ce_ltp})
    instruments.append({"instrument_key": upstox_pe_instrument_key, "quantity": trade_qty, "transaction_type": "SELL", "product": "D", "price": upstox_pe_ltp})
    current_index_price = round(float(finvasia_api.get_quotes(exchange="NSE", token=str(26000))['lp']),2)
    atm_iv =0
    strike_interval = 50
    remainder = math.fmod(current_index_price, strike_interval)
    if remainder > strike_interval / 2:
        atm_strike = math.ceil(current_index_price / strike_interval) * strike_interval
    else:
        atm_strike = math.floor(current_index_price / strike_interval) * strike_interval
    for sp in option_chain:
        if sp.strike_price == atm_strike:
            atm_iv = (float(sp.call_options.option_greeks.iv)+float(sp.put_options.option_greeks.iv))/2

    trade_details['current_index_price']=current_index_price
    lower_range = round((int(upstox_pe_symbol[-7:-2])-current_index_price)/current_index_price*100,2)
    trade_details['lower_range']=lower_range
    upper_range = round((int(upstox_ce_symbol[-7:-2])-current_index_price)/current_index_price*100,2)
    trade_details['upper_range']=upper_range
    trading_range = int(upstox_ce_symbol[-7:-2])-int(upstox_pe_symbol[-7:-2])
    trade_details['trading_range']=trading_range//2
    trade_details['range_per']=round(trading_range/current_index_price*100,2)

    # Build instruments for finvasia
    fin_pe_symbol = convert_option_symbol(upstox_pe_symbol)
    fin_ce_symbol = convert_option_symbol(upstox_ce_symbol)
    trade_details['fin_pe_symbol']=fin_pe_symbol
    trade_details['fin_ce_symbol']=fin_ce_symbol

    span_res = finvasia_api.span_calculator('FA417461',[
            {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":SPAN_Expiry,"optt":"PE","strprc":str(fin_pe_symbol[-5:])+".00","buyqty":"0","sellqty":str(trade_qty),"netqty":"0"},
            {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":SPAN_Expiry,"optt":"CE","strprc":str(fin_ce_symbol[-5:])+".00","buyqty":"0","sellqty":str(trade_qty),"netqty":"0"}
        ])

    trade_margin=float(span_res['span_trade']) + float(span_res['expo_trade'])
    trade_details['trade_margin']=trade_margin
    finvasia_pe_ltp=float(finvasia_api.get_quotes(exchange="NFO", token=str(fin_pe_symbol))['lp'])
    trade_details['finvasia_pe_ltp']=finvasia_pe_ltp
    finvasia_ce_ltp=float(finvasia_api.get_quotes(exchange="NFO", token=str(fin_ce_symbol))['lp'])
    trade_details['finvasia_ce_ltp']=finvasia_ce_ltp
    tot_fin_premium = round(trade_qty*(finvasia_pe_ltp+finvasia_ce_ltp),2)
    trade_details['tot_fin_premium']=tot_fin_premium
    fin_mtm_per= round(tot_fin_premium/trade_margin*100,2)
    trade_details['fin_mtm_per']=str(fin_mtm_per)+"%"
    trade_details['INDIA_VIX']=get_india_vix(finvasia_api)
    try:
        trade_details['INDIA_VIX_RSI']=get_nifty_rsi()
        trade_details['ATM_IV']=atm_iv
    except:
        trade_details['INDIA_VIX_RSI']=-1
        trade_details['ATM_IV']=-1
    return trade_details

def once_an_hour(finvasia_api, upstox_opt_api):
    expiry = '2025-04-09'
    subject = f"Directional: {expiry} |"
    email_body = f"Directional: {expiry} /n"
    upstox_instruments = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz")
    instrument = "NSE_INDEX|Nifty 50" 
    trade_qty=300
    email_body+=f"Trade Qty: {trade_qty} /n"

    # Execute the program
    df = get_data(finvasia_api)
    df = compute_EMA20(df)
    df = compute_supertrend(df)
    signal, timestamp = get_signal(df)
    email_body+=f"Signal: {signal} /n"
    email_body+=f"Timestamp: {timestamp} /n"

    main_leg = get_positions_directional(finvasia_api, instrument, expiry,trade_qty,upstox_instruments, 0.4)
    hedge_leg = get_positions_directional(finvasia_api, instrument, expiry,trade_qty,upstox_instruments, 0.25)
    if signal == "CALL_ENTRY":
        # print(f"SELL {main_leg['fin_ce_symbol']} {trade_qty} qty | DELTA: {round(main_leg['call_delta'],2)}")
        # print(f"BUY {hedge_leg['fin_ce_symbol']} {trade_qty} qty | DELTA: {round(hedge_leg['call_delta'],2)}")
        email_body += f"SELL {main_leg['fin_ce_symbol']} {trade_qty} qty | DELTA: {round(main_leg['call_delta'],2)} /n"
        email_body += f"BUY {hedge_leg['fin_ce_symbol']} {trade_qty} qty | DELTA: {round(hedge_leg['call_delta'],2)} /n"
    elif signal == "PUT_ENTRY":
        email_body += f"SELL {main_leg['fin_pe_symbol']} {trade_qty} qty | DELTA: {round(main_leg['call_delta'],2)} /n"
        email_body += f"BUY {hedge_leg['fin_pe_symbol']} {trade_qty} qty | DELTA: {round(hedge_leg['call_delta'],2)} /n"
    elif signal == "CALL_EXIT":
        email_body += "EXIT CE SELL and CE BUY POSITIONS /n"
    elif signal == "PUT_EXIT":
        email_body += "EXIT PE SELL and CE BUY POSITIONS /n"
    if timestamp>90:
        subject += "NO ACTION"
        email_body += "NO ACTION /n"
    else:
        subject += "TAKE ACTION"
        email_body += "TAKE ACTION /n"
    
    return subject, email_body

##########################END DIRECTIONAL########################

def main():
    session = identify_session()
    if not session:
        print("No active trading session.")
        return
    
    # Login
    userid, password, vendor_code, api_secret, imei, TOKEN, sender_email, receiver_email, email_password, UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN = init_creds()
    api = login(userid, password, vendor_code, api_secret, imei, TOKEN)
    upstox_client, upstox_opt_api = login_upstox(UPSTOX_API_KEY, UPSTOX_URL, UPSTOX_API_SECRET, UPSTOX_MOB_NO, UPSTOX_CLIENT_PASS, UPSTOX_CLIENT_PIN)

    while is_within_timeframe("03:00", "03:45"):
        print("Initializing")
        sleep_time.sleep(60)

    counter=0

    # Start Monitoring
    while is_within_timeframe(session.get('start_time'), session.get('end_time')):
        metrics = monitor_trade(api, upstox_opt_api, sender_email, receiver_email, email_password)
        
        if metrics =="STOP_LOSS":
            send_email(sender_email, receiver_email, email_password, "STOP LOSS HIT - QUIT", "STOP LOSS HIT")
        else:
        #     subject = f"FINVASIA: MTM:{metrics['Total_PNL']} | NEAR_BE:{metrics['Near_Breakeven']} | RANGE:{metrics['Breakeven_Range_Per']}| MAX_PROFIT:{metrics['Max_Profit']} | MAX_LOSS: {metrics['Max_Loss']}"
            if counter % 10 == 0:
                subject = "FINVASIA STATUS"
                metrics["INDIA_VIX"] = get_india_vix(api)
                email_body = format_trade_metrics(metrics)
                send_email(sender_email, receiver_email, email_password, subject, email_body)
            if counter % 60 ==0:
                subject, email_body = once_an_hour(api, upstox_opt_api)
                send_email_plain(sender_email, receiver_email, email_password, subject, email_body)
            counter+=1
        sleep_time.sleep(60)
  
    # Logout
    api.logout()

if __name__ =="__main__":
    main()