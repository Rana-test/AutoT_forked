import os
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText
import requests
import zipfile
import yaml
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from datetime import datetime, timedelta, date
import time
import pyotp
from api_helper import ShoonyaApiPy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import upstox_client
from upstox_client.rest import ApiException
import pytz

import pyotp
format_line="__________________________________________________________"

sender_email=None
receiver_email=None
email_password=None
userid=None
password=None
vendor_code=None
api_secret=None
imei=None
TOKEN=None
UPSTOX_API_KEY=None
UPSTOX_API_SECRET=None
UPSTOX_CLIENT_ID=None
UPSTOX_URL=None
UPSTOX_MOB_NO=None
UPSTOX_CLIENT_PASS=None
UPSTOX_CLIENT_PIN=None

def init_creds():
    global sender_email, receiver_email, email_password, userid, password, vendor_code, api_secret, imei, TOKEN
    global UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_CLIENT_ID, UPSTOX_URL,UPSTOX_MOB_NO,UPSTOX_CLIENT_PASS,UPSTOX_CLIENT_PIN

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
                UPSTOX_API_SECRET=data.get("UPSTOX_API_SECRET")
                UPSTOX_CLIENT_ID=data.get("UPSTOX_CLIENT_ID")
                UPSTOX_URL=data.get("UPSTOX_URL")
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
        UPSTOX_API_SECRET=os.getenv("UPSTOX_API_SECRET")
        UPSTOX_CLIENT_ID=os.getenv("UPSTOX_CLIENT_ID")
        UPSTOX_URL=os.getenv("UPSTOX_URL")
        UPSTOX_MOB_NO=os.getenv("UPSTOX_MOB_NO")
        UPSTOX_CLIENT_PASS=os.getenv("UPSTOX_CLIENT_PASS")
        UPSTOX_CLIENT_PIN=os.getenv("UPSTOX_CLIENT_PIN")

def send_email(subject, global_vars):
    global sender_email, receiver_email, email_password
    init_creds()
    # Create email
    msg = MIMEMultipart()
    # Add body to email
    with open('logs/app.log', 'r') as f:
        body = f.read() 
        # Send the email
        if global_vars.get('live'):
            subject += "|||LIVE|||"
            print('|||LIVE|||'+ subject, body)
        else:
            subject += "|||DUMMY|||"
            # h.send_email('|||DUMMY|||'+ email_subject, body)
            print('|||DUMMY|||'+ subject, body)
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

def login(logger):
    global userid, password, vendor_code, api_secret, imei, TOKEN
    init_creds()
    twoFA = pyotp.TOTP(TOKEN).now()
    api = ShoonyaApiPy()
    login_response = api.login(userid=userid, password=password, twoFA=twoFA, vendor_code=vendor_code, api_secret=api_secret, imei=imei)   
    if login_response['stat'] == 'Ok':
        print('Logged in sucessfully')
        return api
    else:
        logger.info(f"Login failed: {login_response.get('emsg', 'Unknown error')}")
        print('Logged in failed')
        return None

def wait_for_page_load(driver, timeout=30):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script('return document.readyState') == 'complete')

def upstox_login(logger):
    global UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_CLIENT_ID, UPSTOX_URL,UPSTOX_MOB_NO,UPSTOX_CLIENT_PASS,UPSTOX_CLIENT_PIN
    init_creds()
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

    return access_token

def logger_init():
    logger = logging.getLogger('Auto_Trader')
    logger.setLevel(logging.DEBUG)  # Set the logging level for the logger
    # Rotating file handler (file size limit of 1 MB, keeps 5 backup files)
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=1_000_000, backupCount=5)
    # Create a logging format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def upstox_logger_init():
    logger = logging.getLogger('Auto_Trader')
    logger.setLevel(logging.DEBUG)  # Set the logging level for the logger
    # Rotating file handler (file size limit of 1 MB, keeps 5 backup files)
    file_handler = RotatingFileHandler('logs/upstox_app.log', maxBytes=1_000_000, backupCount=5)
    # Create a logging format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def get_symbol_data():
    # If symbols are not present then download them
    root = 'https://api.shoonya.com/'
    masters = ['NSE_symbols.txt.zip', 'NFO_symbols.txt.zip'] 
    for zip_file in masters:   
        if not os.path.exists(zip_file[:-4]): 
            print(f'downloading {zip_file}')
            url = root + zip_file
            r = requests.get(url, allow_redirects=True)
            open(zip_file, 'wb').write(r.content)
            file_to_extract = zip_file.split()
        
            try:
                with zipfile.ZipFile(zip_file) as z:
                    z.extractall()
                    print("Extracted: ", zip_file)
            except:
                print("Invalid file")

            os.remove(zip_file)
            print(f'remove: {zip_file}')

def logout():
    """Logout from Upstox."""
    # Implement Upstox logout
    pass

def upstox_get_current_positions(market_quote_api_instance, logger, instrument_df, global_vars, open_positions=None):
    global total_m2m, config,email_subject
    closed_m2m=0
    open_m2m=0
    if open_positions is not None:
        for index, row in open_positions.iterrows():
            instrument_key = instrument_df[(instrument_df.tradingsymbol==row['tsym'])].iloc[0].instrument_key
            lp = market_quote_api_instance.ltp(instrument_key, "2.0").data['NSE_FO:'+row['tsym']].last_price
            open_positions['lp'] = open_positions['lp'].astype(float)
            open_positions.at[index, 'lp'] = float(lp)
    # else:
    #     # Fetch the latest data (positions, LTP, etc.)
    #     all_positions_data=[]
    #     ret = api.get_positions()
    #     if ret is None:
    #         logger.info("Issue fetching Positions")
    #         email_subject = "Issue fetching Positions..."
    #         return None, 0, 0
    #     else:
    #         mtm = 0
    #         pnl = 0
    #         for i in ret:
    #             if i['tsym'][:5]=='NIFTY' or i['tsym'][:9] =='BANKNIFTY':
    #                 if int(i['netqty'])<0:
    #                     buy_sell = 'S'
    #                 elif int(i['netqty'])>0:
    #                     buy_sell = 'B'
    #                 elif int(i['netqty'])==0:
    #                     buy_sell = 'NA'
                    
    #                 # Error handling in case of wrong LTP value returned by API
    #                 if float(i['lp'])<1:
    #                     logger.info("Below 1 LP issue. Check manually")
    #                     email_subject = "Below 1 LP issue. Check manually"
    #                     return None, 0, 0

    #                 all_positions_data.append({
    #                     'buy_sell': buy_sell, 
    #                     'tsym':i['tsym'], 
    #                     'qty': i['netqty'], 
    #                     'remarks':'Existing Order', 
    #                     'upldprc': i['upldprc'], 
    #                     'netupldprc': i['netupldprc'], 
    #                     'lp':i['lp'], 
    #                     'ord_type':i['tsym'][12],
    #                     'rpnl':i['rpnl'],
    #                     'cfbuyqty': i['cfbuyqty'],
    #                     'cfsellqty': i['cfsellqty'],                
    #                     'daybuyamt':i['daybuyamt'],
    #                     'daysellamt':i['daysellamt']
    #                     })
    #         all_positions_df = pd.DataFrame(all_positions_data)
            
    #         if not all_positions_df.empty:
    #             # Calculate Total M2M
    #             closed_positions = all_positions_df[all_positions_df['buy_sell']=="NA"]
    #             closed_positions = closed_positions.copy() 
    #             if not closed_positions.empty:
    #                 closed_positions.loc[:,'totcfbuyamt'] = closed_positions.upldprc.astype(float)*closed_positions.cfbuyqty.astype(int)
    #                 closed_positions.loc[:,'totcfsellamt'] = closed_positions.upldprc.astype(float)*closed_positions.cfsellqty.astype(int)
    #                 closed_positions.loc[:,'netbuy']=closed_positions['daybuyamt'].astype(float)+closed_positions['totcfbuyamt']
    #                 closed_positions.loc[:,'netsell']=closed_positions['daysellamt'].astype(float)+closed_positions['totcfsellamt']
    #                 closed_positions.loc[:,'net_prft']=closed_positions['netsell']-closed_positions['netbuy']
    #                 closed_m2m = round(float(closed_positions['net_prft'].sum()),2)
    #                 print(f"Closed M2M: {closed_m2m}")
    #                 logger.info(format_line)
    #                 logger.info(f"<<<TODAY'S CLOSED POSITION : {closed_m2m} | ADJUST TARGET PROFIT>>>")
    #                 logger.info(format_line)
    #                 del closed_positions

    #             open_positions = all_positions_df[~(all_positions_df['buy_sell']=="NA")]
    #             open_positions = open_positions.copy() 

    # if not open_positions.empty:
    #     open_positions['net_profit']=(open_positions['lp'].astype(float)-open_positions['netupldprc'].astype(float))*open_positions['qty'].astype(float)
    #     open_m2m =round(float(open_positions['net_profit'].sum()),2)
    #     logger.info(format_line)
    #     logger.info("<<<CURRENT POSITION>>>")
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         logger.info("\n%s",open_positions[['buy_sell', 'tsym', 'qty', 'netupldprc', 'lp']])

    # if global_vars is not None:
    #     # total_m2m = closed_m2m+open_m2m+float(global_vars.get('Past_M2M'))
    #     total_m2m = closed_m2m+open_m2m
    # else:
        total_m2m = closed_m2m+open_m2m
        
    return open_positions, total_m2m, closed_m2m

def get_current_positions(api, logger, global_vars= None, open_positions=None):
    global total_m2m, config,email_subject
    closed_m2m=0
    open_m2m=0
    if open_positions is not None:
        # global_vars.get('positions_data')=='csv':
        # Read position_data.csv
        nfo_df = pd.read_csv(global_vars.get('nfo_sym_path'))
        # Get current ltp of each position and update lp
        for index, row in open_positions.iterrows():
            row_token = nfo_df[(nfo_df.TradingSymbol==row['tsym'])].iloc[0].Token
            res=api.get_quotes(exchange="NFO", token=str(row_token))
            open_positions['lp'] = open_positions['lp'].astype(float)
            open_positions.at[index, 'lp'] = float(res['lp'])
    else:
        # Fetch the latest data (positions, LTP, etc.)
        all_positions_data=[]
        ret = api.get_positions()
        if ret is None:
            logger.info("Issue fetching Positions")
            email_subject = "Issue fetching Positions..."
            return None, 0, 0
        else:
            mtm = 0
            pnl = 0
            for i in ret:
                if i['tsym'][:5]=='NIFTY' or i['tsym'][:9] =='BANKNIFTY':
                    if int(i['netqty'])<0:
                        buy_sell = 'S'
                    elif int(i['netqty'])>0:
                        buy_sell = 'B'
                    elif int(i['netqty'])==0:
                        buy_sell = 'NA'
                    
                    # Error handling in case of wrong LTP value returned by API
                    if float(i['lp'])<1:
                        logger.info("Below 1 LP issue. Check manually")
                        email_subject = "Below 1 LP issue. Check manually"
                        return None, 0, 0

                    all_positions_data.append({
                        'buy_sell': buy_sell, 
                        'tsym':i['tsym'], 
                        'qty': i['netqty'], 
                        'remarks':'Existing Order', 
                        'upldprc': i['upldprc'], 
                        'netupldprc': i['netupldprc'], 
                        'lp':i['lp'], 
                        'ord_type':i['tsym'][12],
                        'rpnl':i['rpnl'],
                        'cfbuyqty': i['cfbuyqty'],
                        'cfsellqty': i['cfsellqty'],                
                        'daybuyamt':i['daybuyamt'],
                        'daysellamt':i['daysellamt']
                        })
            all_positions_df = pd.DataFrame(all_positions_data)
            
            if not all_positions_df.empty:
                # Calculate Total M2M
                closed_positions = all_positions_df[all_positions_df['buy_sell']=="NA"]
                closed_positions = closed_positions.copy() 
                if not closed_positions.empty:
                    closed_positions.loc[:,'totcfbuyamt'] = closed_positions.upldprc.astype(float)*closed_positions.cfbuyqty.astype(int)
                    closed_positions.loc[:,'totcfsellamt'] = closed_positions.upldprc.astype(float)*closed_positions.cfsellqty.astype(int)
                    closed_positions.loc[:,'netbuy']=closed_positions['daybuyamt'].astype(float)+closed_positions['totcfbuyamt']
                    closed_positions.loc[:,'netsell']=closed_positions['daysellamt'].astype(float)+closed_positions['totcfsellamt']
                    closed_positions.loc[:,'net_prft']=closed_positions['netsell']-closed_positions['netbuy']
                    closed_m2m = round(float(closed_positions['net_prft'].sum()),2)
                    print(f"Closed M2M: {closed_m2m}")
                    logger.info(format_line)
                    logger.info(f"<<<TODAY'S CLOSED POSITION : {closed_m2m} | ADJUST TARGET PROFIT>>>")
                    logger.info(format_line)
                    del closed_positions

                open_positions = all_positions_df[~(all_positions_df['buy_sell']=="NA")]
                open_positions = open_positions.copy() 

    if not open_positions.empty:
        open_positions['net_profit']=(open_positions['lp'].astype(float)-open_positions['netupldprc'].astype(float))*open_positions['qty'].astype(float)
        open_m2m =round(float(open_positions['net_profit'].sum()),2)
        logger.info(format_line)
        logger.info("<<<CURRENT POSITION>>>")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info("\n%s",open_positions[['buy_sell', 'tsym', 'qty', 'netupldprc', 'lp']])

    if global_vars is not None:
        # total_m2m = closed_m2m+open_m2m+float(global_vars.get('Past_M2M'))
        total_m2m = closed_m2m+open_m2m
    else:
        total_m2m = closed_m2m+open_m2m
        
    return open_positions, total_m2m, closed_m2m

def get_current_positions_new(api, logger, nfo_df, past_m2m, open_positions=None):
    global total_m2m, config,email_subject
    closed_m2m=0
    open_m2m=0
    if open_positions is not None:
        # Get current ltp of each position and update lp
        for index, row in open_positions.iterrows():
            row_token = nfo_df[(nfo_df.TradingSymbol==row['tsym'])].iloc[0].Token
            res=api.get_quotes(exchange="NFO", token=str(row_token))
            open_positions['lp'] = open_positions['lp'].astype(float)
            open_positions.at[index, 'lp'] = float(res['lp'])
    else:
        # Fetch the latest data (positions, LTP, etc.)
        all_positions_data=[]
        ret = api.get_positions()
        if ret is None:
            logger.info("Issue fetching Positions")
            email_subject = "Issue fetching Positions..."
            return None, 0, 0
        else:
            mtm = 0
            pnl = 0
            for i in ret:
                if i['tsym'][:5]=='NIFTY' or i['tsym'][:9] =='BANKNIFTY':
                    if int(i['netqty'])<0:
                        buy_sell = 'S'
                    elif int(i['netqty'])>0:
                        buy_sell = 'B'
                    elif int(i['netqty'])==0:
                        buy_sell = 'NA'
                    
                    # Error handling in case of wrong LTP value returned by API
                    if float(i['lp'])<1:
                        logger.info("Below 1 LP issue. Check manually")
                        email_subject = "Below 1 LP issue. Check manually"
                        return None, 0, 0

                    all_positions_data.append({
                        'buy_sell': buy_sell, 
                        'tsym':i['tsym'], 
                        'qty': i['netqty'], 
                        'remarks':'Existing Order', 
                        'upldprc': i['upldprc'], 
                        'netupldprc': i['netupldprc'], 
                        'lp':i['lp'], 
                        'ord_type':i['tsym'][12],
                        'rpnl':i['rpnl'],
                        'cfbuyqty': i['cfbuyqty'],
                        'cfsellqty': i['cfsellqty'],                
                        'daybuyamt':i['daybuyamt'],
                        'daysellamt':i['daysellamt']
                        })
            all_positions_df = pd.DataFrame(all_positions_data)
            
            if not all_positions_df.empty:
                # Calculate Total M2M
                closed_positions = all_positions_df[all_positions_df['buy_sell']=="NA"]
                closed_positions = closed_positions.copy() 
                if not closed_positions.empty:
                    closed_positions.loc[:,'totcfbuyamt'] = closed_positions.upldprc.astype(float)*closed_positions.cfbuyqty.astype(int)
                    closed_positions.loc[:,'totcfsellamt'] = closed_positions.upldprc.astype(float)*closed_positions.cfsellqty.astype(int)
                    closed_positions.loc[:,'netbuy']=closed_positions['daybuyamt'].astype(float)+closed_positions['totcfbuyamt']
                    closed_positions.loc[:,'netsell']=closed_positions['daysellamt'].astype(float)+closed_positions['totcfsellamt']
                    closed_positions.loc[:,'net_prft']=closed_positions['netsell']-closed_positions['netbuy']
                    closed_m2m = round(float(closed_positions['net_prft'].sum()),2)
                    print(f"Closed M2M: {closed_m2m}")
                    logger.info(format_line)
                    logger.info(f"<<<TODAY'S CLOSED POSITION : {closed_m2m} | ADJUST TARGET PROFIT>>>")
                    logger.info(format_line)
                    del closed_positions

                open_positions = all_positions_df[~(all_positions_df['buy_sell']=="NA")]
                open_positions = open_positions.copy() 

    if not open_positions.empty:
        open_positions['net_profit']=(open_positions['lp'].astype(float)-open_positions['netupldprc'].astype(float))*open_positions['qty'].astype(float)
        open_m2m =round(float(open_positions['net_profit'].sum()),2)
        logger.info(format_line)
        logger.info("<<<CURRENT POSITION>>>")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info("\n%s",open_positions[['buy_sell', 'tsym', 'qty', 'netupldprc', 'lp']])

    total_m2m = closed_m2m+open_m2m+past_m2m
        
    return open_positions, total_m2m, closed_m2m

def get_revised_position(api, logger,Symbol, Past_M2M):
    # Publish new Positions after 5 second wait
    time.sleep(10)
    rev_position, rev_m2m, closed_m2m = get_current_positions(api, logger,Symbol, Past_M2M)
    logger.info(format_line)
    logger.info("<<<REVISED POSITIONS>>>")
    if rev_position is None or rev_position.empty:
        logger.info("NO POSITIONS CREATED")
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info("\n%s",rev_position[['buy_sell', 'tsym', 'qty', 'upldprc', 'lp']])
        logger.info(format_line)
        logger.info(f"<<<REVISED M2M: {rev_m2m}>>>")
        
    logger.info(format_line)

def execute_basket(logger, global_vars, api, orders_df):
    # {'buy_sell':'B', 'tsym': pe_hedge, 'qty': lots*lot_size, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
    # place_order(buy_sell, tsym, qty, remarks="regular order")
    for i, order in orders_df.iterrows():
        place_order(logger, global_vars, api, order['buy_sell'], order['tsym'], order['qty'], order['remarks'])

def exit_positions(logger, global_vars, api, orders_df):
    # {'buy_sell':'B', 'tsym': pe_hedge, 'qty': lots*lot_size, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
    # place_order(buy_sell, tsym, qty, remarks="regular order")
    for i, order in orders_df.iterrows():
        rev_buy_sell = {"B": "S", "S": "B"}.get(order['buy_sell'])
        place_order(logger, global_vars, api, rev_buy_sell, order['tsym'], abs(int(order['qty'])), order['remarks'])

def auto_exit(logger, api, global_vars, max_profit, strategy, m2m, positions_df):
    # Reducing max_profit_percent by 3% for every adjustment
    auto_target = max_profit * (int(global_vars.get('percent_of_max_profit'))-3*global_vars.get('num_adjustments'))/100
    # Divide 1.5 targets for Iron Fly 
    if strategy =="IF":
        auto_target=auto_target/1.5
    logger.info(f"Max profit = {max_profit} | Auto target = {auto_target}")
    logger.info(f"if {m2m} > {auto_target} or {m2m}> {global_vars.get('target_profit')}:")
    logger.info(f"if {m2m} < -1 * {auto_target} or {m2m} < {global_vars.get('stop_loss')}")
    if m2m > auto_target or m2m> global_vars.get('target_profit'):
        logger.info(format_line)
        logger.info("Target Profit Acheived. Exit Trade")
        email_subject = f'<<< TARGET PROFIT ACHIEVED. EXIT TRADE | M2M: {m2m} >>>'   
        exit_positions(logger, global_vars, api, positions_df[['buy_sell','tsym','qty','remarks']])
        logger.info(format_line)
        return True
    elif m2m < -1 * auto_target or m2m < global_vars.get('stop_loss'):
        # Implement traling stop loss
        logger.info(format_line)
        logger.info("Stop Loss hit. Exit Trade")
        email_subject = f'<<< STOP LOSS HIT. EXIT TRADE | M2M: {m2m} >>>'
        exit_positions(logger, global_vars, api, positions_df[['buy_sell','tsym','qty','remarks']])
        logger.info(format_line)
        return True
    else:
        return False

def get_Option_Chain(api, logger, global_vars, symbol, expiry, minsp,maxsp, type):

    symbolDf = pd.read_csv(global_vars.get('nfo_sym_path'))
    symbolDf['hundred_strikes'] = symbolDf['TradingSymbol'].apply(lambda x: x[-2:])

    df=symbolDf[
        (symbolDf.OptionType==type) 
        & (symbolDf.Symbol==symbol)
        & (symbolDf.Expiry==expiry) 
        # & (symbolDf['hundred_strikes']=="00") # Keeping it safe
        & (symbolDf['StrikePrice']>minsp) & (symbolDf['StrikePrice']<maxsp)
        ]
    
    OList=[]
    for i in df.index:
        strikeInfo = df.loc[i]
        res=api.get_quotes(exchange="NFO", token=str(strikeInfo.Token))
        res={'oi':res.get('oi',0), 'tsym':res['tsym'], 'lp':float(res['lp']), 'lotSize':strikeInfo.LotSize, 'token':res['token'], 'StrikePrice':int(float(res['strprc']))}
        OList.append(res)
    Ostrikedf = pd.DataFrame(OList)
    return Ostrikedf

def get_Option_Chain_new(api, symbol, expiry, symbolDf, optype, min_strike_price=24000, max_strike_price=27000):
    print("Getting Option Chain...")
    # symbolDf['hundred_strikes'] = symbolDf['TradingSymbol'].apply(lambda x: x[-2:])
    
    df=symbolDf[
        (symbolDf.OptionType==optype) 
        & (symbolDf.Symbol==symbol)
        & (symbolDf.Expiry==expiry) 
        # & (symbolDf['hundred_strikes']=="00") # Keeping it safe
        & (symbolDf['StrikePrice']>min_strike_price)
        & (symbolDf['StrikePrice']<max_strike_price)
        ]
    
    OList=[]
    for i in df.index:
        strikeInfo = df.loc[i]
        res=api.get_quotes(exchange="NFO", token=str(strikeInfo.Token))
        res={'oi':res.get('oi',0), 'tsym':res['tsym'], 'lp':float(res['lp']), 'lotSize':strikeInfo.LotSize, 'token':res['token'], 'StrikePrice':int(float(res['strprc']))}
        OList.append(res)
    Ostrikedf = pd.DataFrame(OList)
    print("Completed fetching Option Chain data.")
    return Ostrikedf

def get_nearest_price_strike(df, ltp):
    df['price_diff']=abs(df['lp']-ltp)
    df.sort_values(by='price_diff', inplace=True)
    return df.iloc[0]['tsym'], df.iloc[0]['lp']

def get_nearest_strike_strike(df, strike):
    df['strike_diff']=abs(df['StrikePrice']-strike)
    df.sort_values(by='strike_diff', inplace=True)
    return df.iloc[0]['tsym'], df.iloc[0]['lp']

def get_support_resistence_atm(cedf,pedf):
    cedf['oi'] = pd.to_numeric(cedf['oi'], errors='coerce')
    cedf.dropna(subset=['oi'], inplace=True)
    cedf.sort_values(by='oi', ascending=False, inplace=True)

    pedf['oi'] = pd.to_numeric(pedf['oi'], errors='coerce')
    pedf.dropna(subset=['oi'], inplace=True)
    pedf.sort_values(by='oi', ascending=False, inplace=True)

    support = int(pedf.iloc[0]['tsym'][-5:])
    support_oi = pedf.iloc[0]['oi']
    resistance = int(cedf.iloc[0]['tsym'][-5:])
    resistance_oi = cedf.iloc[0]['oi']
    if (support+resistance)/2 % 50 == 0.0:
        atm = (support+resistance)/2
    elif support_oi>resistance_oi:
        atm = (support+resistance-50)/2
    else:
        atm = (support+resistance+50)/2
    return atm

def calculate_initial_positions(global_vars, logger, api, base_strike_price, CEOptdf, PEOptdf):
    ce_sell, ce_premium = get_nearest_strike_strike(CEOptdf, base_strike_price)
    pe_sell, pe_premium = get_nearest_strike_strike(PEOptdf, base_strike_price)
    tot_premium=round(pe_premium+ce_premium,2)
    pe_breakeven = base_strike_price - tot_premium
    ce_breakeven = base_strike_price + tot_premium
    ce_hedge, ce_hedge_premium = get_nearest_strike_strike(CEOptdf, ce_breakeven)
    pe_hedge, pe_hedge_premium = get_nearest_strike_strike(PEOptdf,pe_breakeven)
    qty = global_vars.get('lots')*global_vars.get('lot_size')
    Expiry= global_vars.get('Expiry')
    orders_df = pd.DataFrame([ 
        {'buy_sell':'B', 'tsym': pe_hedge, 'qty': qty, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
        {'buy_sell':'S', 'tsym': pe_sell,  'qty': qty, 'remarks':f'Initial PE Sell with premium: {pe_premium}'},
        {'buy_sell':'S', 'tsym': ce_sell,  'qty': qty, 'remarks':f'Initial CE Sell with premium: {ce_premium}'},
        {'buy_sell':'B', 'tsym': ce_hedge, 'qty': qty, 'remarks':f'Initial CE Hedg with premium: {ce_hedge_premium}'},
    ])
    span_res = api.span_calculator(userid,[
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"PE","strprc":str(pe_hedge[-5:])+".00","buyqty":str(qty),"sellqty":"0","netqty":"0"},
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"CE","strprc":str(ce_hedge[-5:])+".00","buyqty":str(qty),"sellqty":"0","netqty":"0"},
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"PE","strprc":str(pe_sell[-5:])+".00","buyqty":"0","sellqty":str(qty),"netqty":"0"},
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"CE","strprc":str(ce_sell[-5:])+".00","buyqty":"0","sellqty":str(qty),"netqty":"0"}
    ])
    trade_margin=float(span_res['span_trade']) + float(span_res['expo_trade'])
    cash_margin = (pe_hedge_premium+ ce_hedge_premium)*qty
    net_margin=trade_margin-cash_margin
    logger.info(format_line)
    print(f"Margin Requirements: Total: {trade_margin} | Equity Collateral: {round(net_margin/2,2)} | Cash Collateral: {round(net_margin/2,2)} | Cash: {cash_margin}")
    logger.info(f"Margin Requirements: Total: {trade_margin} | Equity Collateral: {round(net_margin/2,2)} | Cash Collateral: {round(net_margin/2,2)} | Cash: {cash_margin}")
    print(orders_df)
    logger.info(orders_df)
    logger.info(format_line)
    net_premium = round(tot_premium-ce_hedge_premium-pe_hedge_premium,2)
    logger.info(f"Sell premium = {tot_premium} | Net premium = {net_premium}")
    print(f"Net premium = {net_premium}")

    return orders_df, net_premium, trade_margin

def enter_trade_ironfly(api, logger,global_vars):
    global email_subject
    # global CEstrikedf, PEstrikedf
    print("Getting CE Option Chain...")
    CEOptdf=get_Option_Chain(api, logger, global_vars, "CE")
    print("Getting PE Option Chain...")
    PEOptdf=get_Option_Chain(api, logger, global_vars, "PE")

    # Get initial trade basis future price
    max_net_premium=0
    best_entry=None
    best_ord_df=None
    auto_margin = 0

    # Get initial trade basis oi
    print("Getting positions based on oi support/resistance")
    atm = get_support_resistence_atm(CEOptdf,PEOptdf)
    logger.info(format_line)
    logger.info("Positions based on Support/Resistance")
    Oi_ord_df, oi_net_premium, oi_margin = calculate_initial_positions(global_vars, logger, api, atm, CEOptdf, PEOptdf)
    Oi_ord_df.sort_values(by='buy_sell', inplace=True)

    print("Getting Future Price")
    res=api.get_quotes(exchange="NFO", token=str(global_vars.get('future_price_token')))
    future_strike = float(res['lp'])
    print("Positions based on Future Price")
    logger.info(format_line)
    logger.info("Positions based on Future Price")
    Fut_ord_df, f_net_premium, f_margin = calculate_initial_positions(global_vars, logger, api, future_strike, CEOptdf, PEOptdf)
    Fut_ord_df.sort_values(by='buy_sell', inplace=True)
    if f_net_premium>max_net_premium:
        max_net_premium= f_net_premium
        best_entry="FUTURE"
        best_ord_df=Fut_ord_df
        auto_margin = f_margin

    # Get initial trade basis current price
    print("Getting Current Price")
    res=api.get_quotes(exchange="NSE", token=str(global_vars.get('nifty_nse_token')))
    current_strike = float(res['lp'])
    logger.info(format_line)
    print("Positions based on Current Price")
    logger.info("Positions based on Current Price")
    Curr_ord_df, curr_net_premium, curr_margin = calculate_initial_positions(global_vars, logger, api, current_strike, CEOptdf, PEOptdf)
    Curr_ord_df.sort_values(by='buy_sell', inplace=True)
    if curr_net_premium>max_net_premium:
        max_net_premium= curr_net_premium
        best_entry="CURRENT"
        best_ord_df=Curr_ord_df
        auto_margin = curr_margin

    # Get initial trade basis delta
    logger.info(format_line)
    print("Positions based on Delta")
    logger.info("Positions based on Delta")
    delta_oc = CEOptdf.merge(PEOptdf, on = 'StrikePrice', how = 'left')
    delta_oc['delta_diff'] = abs(float(delta_oc[(delta_oc['lp_x']>0) &(delta_oc['lp_y']>0)]['lp_x']-delta_oc[(delta_oc['lp_x']>0) &(delta_oc['lp_y']>0)]['lp_y']))
    delta_oc.sort_values(by='delta_diff', inplace=True)
    delta_atm = delta_oc['StrikePrice'].iloc[0]
    Delta_ord_df, d_net_premium, d_margin = calculate_initial_positions(global_vars, logger, api, delta_atm, CEOptdf, PEOptdf)
    Delta_ord_df.sort_values(by='buy_sell', inplace=True)
    if d_net_premium>max_net_premium:
        max_net_premium= d_net_premium
        best_entry="DELTA"
        best_ord_df=Delta_ord_df
        auto_margin = d_margin
    
    # Get initial trade basis combination
    print("Getting combined position")
    comb_atm = round((4*delta_atm+2*current_strike+future_strike+atm)/800,0)*100
    logger.info(format_line)
    logger.info("Combined Positions")
    Comb_ord_df, comb_net_premium, comb_margin = calculate_initial_positions(global_vars, logger, api, comb_atm, CEOptdf, PEOptdf)
    Comb_ord_df.sort_values(by='buy_sell', inplace=True)
    if comb_margin>max_net_premium:
        max_net_premium= comb_net_premium
        best_entry="COMBINED"
        best_ord_df=Comb_ord_df
        auto_margin = comb_margin

    # Execute trade:
    logger.info(format_line)
    if global_vars.get('EntryType')=="AUTO":
        logger.info(f"AUTO: Placing Order as per {best_entry}")
        print(f"AUTO: Placing Order as per {best_entry}")
        execute_basket(logger, global_vars, api, best_ord_df)
    elif global_vars.get('EntryType')== "CURRENT":
        logger.info("Placing Order as per Current Values")
        execute_basket(logger, global_vars, api, Curr_ord_df)
    elif global_vars.get('EntryType')== "FUTURE":
        logger.info("Placing Order as per Future Values")
        execute_basket(logger, global_vars, api, Fut_ord_df)
    elif global_vars.get('EntryType')== "COMBINED":
        logger.info("Placing Order as per Combined Values")
        execute_basket(logger, global_vars, api, Comb_ord_df)
    elif global_vars.get('EntryType')== "DELTA":
        logger.info("Placing Order as per Delta Neutral")
        execute_basket(logger, global_vars, api, Delta_ord_df)
    elif global_vars.get('EntryType')== "OI":
        logger.info("Placing Order as per OI")
        execute_basket(logger, global_vars, api, Oi_ord_df)

    # email_subject = '<<<<<<<< ENTRY MADE >>>>>>>>>>>>'
    # current_time = datetime.now()
    # updated_time = current_time + timedelta(hours=5, minutes=30)
    # extracted_date = updated_time.date()
    # config['Entry_Date']=str(extracted_date)
    # save_config()
    # clear_state('state.csv')

    return

def enter_trade_manual(api, logger,global_vars):
    enter_trade_pd = pd.read_csv('entry_trade.csv')
    print(enter_trade_pd)

def count_working_days(global_vars):
    # Generate a range of dates from start_date to end_date
    if global_vars.get('Entry_Date')==0:
        Entry_Date= datetime.now()+ timedelta(hours=5, minutes=30)
    else:
        Entry_Date =global_vars.get('Entry_Date')
    current_time = datetime.now()
    updated_time = current_time + timedelta(hours=5, minutes=30)
    date_range = pd.date_range(start=Entry_Date, end=updated_time.date(),freq='B')  # 'B' for business days
    return len(date_range)

def place_order_new(logger, api, order_df, lot_size,live=0,remarks="regular order"):
    prd_type = 'M'
    exchange = 'NFO' 
    disclosed_qty= lot_size
    price_type = 'MKT'
    price=0
    trigger_price = None
    retention='DAY'
    for i, row in order_df.iterrows():
        if live:
            ret = api.place_order(buy_or_sell=row['buy_sell'], product_type=prd_type, exchange=exchange, 
                                  tradingsymbol=row['tsym'], quantity=abs(int(row['qty'])), 
                                  discloseqty=disclosed_qty,price_type=price_type, price=price,
                                  trigger_price=trigger_price, retention=retention, remarks=remarks)
            # print(buy_sell,"|", prd_type,"|", exchange,"|", tsym,"|", qty,"|", disclosed_qty,"|", price_type,"|", price,"|", trigger_price,"|", retention,"|", remarks)
            print(ret)
            if ret['stat']=="Ok":
                logger.info(f"Order successsful, Order No: {ret['norenordno']}") # Add reject reason
            else:
                logger.info(f"Order failed, Error: {ret['emsg']}")
        else:
            logger.info(f"TEST ORDER PLACEMENT: {row['buy_sell']}, {row['tsym']}, {abs(int(row['qty']))}, {remarks}")
            print((f"TEST ORDER PLACEMENT : {row['buy_sell']}, {row['tsym']}, {abs(int(row['qty']))}, {remarks}"))

def place_order(logger, global_vars, api, buy_sell, tsym, qty, remarks="regular order"):
    prd_type = 'M'
    exchange = 'NFO' 
    disclosed_qty= global_vars.get('lot_size')
    price_type = 'MKT'
    price=0
    trigger_price = None
    retention='DAY'
    if global_vars.get('live'):
        ret = api.place_order(buy_or_sell=buy_sell, product_type=prd_type, exchange=exchange, tradingsymbol=tsym, quantity=qty, discloseqty=disclosed_qty,
                              price_type=price_type, price=price,trigger_price=trigger_price, retention=retention, remarks=remarks)
        # print(buy_sell,"|", prd_type,"|", exchange,"|", tsym,"|", qty,"|", disclosed_qty,"|", price_type,"|", price,"|", trigger_price,"|", retention,"|", remarks)
        print(ret)
        if ret['stat']=="Ok":
            logger.info(f"Order successsful, Order No: {ret['norenordno']}") # Add reject reason
        else:
            logger.info(f"Order failed, Error: {ret['emsg']}")
    else:
        logger.info(f"TEST ORDER PLACEMENT: {buy_sell}, {tsym}, {qty}, {remarks}")
        print((f"TEST ORDER PLACEMENT : {buy_sell}, {tsym}, {qty}, {remarks}"))

def calculate_delta(logger, global_vars, api, df, current_strike):
    # Verify that there are only 2 records
    put_order = df[(df.buy_sell=="S")&(df.ord_type=="P")]
    call_order = df[(df.buy_sell=="S")&(df.ord_type=="C")] 

    pltp= float(put_order.iloc[0]['lp'])
    cltp= float(call_order.iloc[0]['lp'])
    delta = round(100*abs(pltp-cltp)/(pltp+cltp),2)

    pdiff = float(put_order.iloc[0]['upldprc'])-pltp
    cdiff = float(call_order.iloc[0]['upldprc'])-cltp
    profit_leg = "C" if  cdiff > pdiff else "P"
    loss_leg = "P" if  cdiff > pdiff else "C"

    pstrike = int(put_order.iloc[0]['tsym'][-5:])
    cstrike = int(call_order.iloc[0]['tsym'][-5:])

# if distance between put and call strike is less than min(dist between put and current_strike , dist between call and current_strike) consider strategy as IF
    print("Calculating Delta .. Getting Current Price")
    

    strategy = 'IF' if abs(pstrike-cstrike) < min(abs(current_strike-pstrike), abs(current_strike-cstrike)) else 'IC'
    # Special condition
    if pstrike==cstrike and cstrike==current_strike:
        strategy = 'IF'
    
    if abs(pstrike-cstrike)>1000:
        strategy="SAFE"

    put_hedge = df[(df.buy_sell=="B")&(df.ord_type=="P")]
    call_hedge = df[(df.buy_sell=="B")&(df.ord_type=="C")] 

    if len(put_hedge)>0:
        p_hedge_strike = int(put_hedge.iloc[0]['tsym'][-5:])
        pe_hedge_diff= pstrike-p_hedge_strike
    else:
        p_hedge_strike = None
        pe_hedge_diff= None

    if len(call_hedge)>0:
        c_hedge_strike = int(call_hedge.iloc[0]['tsym'][-5:])
        ce_hedge_diff= c_hedge_strike-cstrike
    else:
        c_hedge_strike = None
        ce_hedge_diff= None

    return delta, pltp, cltp, profit_leg, loss_leg, strategy, pe_hedge_diff, ce_hedge_diff, current_strike, pstrike, cstrike

def upstox_calculate_delta(df, current_strike):
    print(df)
    # Verify that there are only 2 records
    put_order = df[(df.buy_sell=="S")&(df.ord_type=="P")]
    call_order = df[(df.buy_sell=="S")&(df.ord_type=="C")] 

    pltp= float(put_order.iloc[0]['lp'])
    cltp= float(call_order.iloc[0]['lp'])
    delta = round(100*abs(pltp-cltp)/(pltp+cltp),2)

    pdiff = float(put_order.iloc[0]['upldprc'])-pltp
    cdiff = float(call_order.iloc[0]['upldprc'])-cltp
    profit_leg = "C" if  cdiff > pdiff else "P"
    loss_leg = "P" if  cdiff > pdiff else "C"

    pstrike = int(put_order.iloc[0]['tsym'][-7:-2])
    cstrike = int(call_order.iloc[0]['tsym'][-7:-2])

# if distance between put and call strike is less than min(dist between put and current_strike , dist between call and current_strike) consider strategy as IF
    print("Calculating Delta .. Getting Current Price")
    

    strategy = 'IF' if abs(pstrike-cstrike) < min(abs(current_strike-pstrike), abs(current_strike-cstrike)) else 'IC'
    # Special condition
    if pstrike==cstrike and cstrike==current_strike:
        strategy = 'IF'

    put_hedge = df[(df.buy_sell=="B")&(df.ord_type=="P")]
    call_hedge = df[(df.buy_sell=="B")&(df.ord_type=="C")] 

    if len(put_hedge)>0:
        p_hedge_strike = int(put_hedge.iloc[0]['tsym'][-7:-2])
        pe_hedge_diff= pstrike-p_hedge_strike
    else:
        p_hedge_strike = None
        pe_hedge_diff= None

    if len(call_hedge)>0:
        c_hedge_strike = int(call_hedge.iloc[0]['tsym'][-7:-2])
        ce_hedge_diff= c_hedge_strike-cstrike
    else:
        c_hedge_strike = None
        ce_hedge_diff= None

    return delta, pltp, cltp, profit_leg, loss_leg, strategy, pe_hedge_diff, ce_hedge_diff, current_strike, pstrike, cstrike


def calculate_metrics(logger, df):
    #Hedges
    puth_order = df[(df.buy_sell=="B")&(df.ord_type=="P")] 
    if len(puth_order)>0:
        phentry = float(puth_order.iloc[0]['netupldprc'])
        phltp= float(puth_order.iloc[0]['lp'])
    else:
        phentry=0
        phltp= 0

    callh_order = df[(df.buy_sell=="B")&(df.ord_type=="C")] 
    if len(callh_order)>0:
        chentry = float(callh_order.iloc[0]['netupldprc'])
        chltp= float(callh_order.iloc[0]['lp'])
    else:
        chentry = 0
        chltp= 0

    #Put and Call legs
    put_order = df[(df.buy_sell=="S")&(df.ord_type=="P")]
    call_order = df[(df.buy_sell=="S")&(df.ord_type=="C")] 
    pentry = float(put_order.iloc[0]['netupldprc'])
    centry = float(call_order.iloc[0]['netupldprc'])
    pltp= float(put_order.iloc[0]['lp'])
    cltp= float(call_order.iloc[0]['lp'])

    # CE: 100* (centry-clp)/centry << ++ is good
    ce_premium_movement = round(100* (centry-cltp)/centry,2)
    # PE: 100* (pentry-plp)/pentry << ++ is good
    pe_premium_movement = round(100* (pentry-pltp)/pentry,2)

    combined_premiums = round(100*((centry+pentry)-(cltp-pltp))/(centry+pentry),2)

    # Cobimed Delta: 100*((centry+pentry-chentry-phentry)-(clp+plp-chlp-plp))/(centry+pentry-chentry-phentry) << ++ is good
    net_delta_drift = round(100*((centry+pentry-chentry-phentry)-(cltp+pltp-chltp-pltp))/(centry+pentry-chentry-phentry),2)

    # CE/HCE: 100 * chltp-chentry)/(centry-cltp) << ++ is good
    ce_short_long = round(100 * (chltp-chentry)/(centry-cltp),2)

    # PE/HPE: 100 * (phlp-phentry)/(pentry-pltp) << ++ is good
    pe_short_long = round(100 * (phltp-phentry)/(pentry-pltp),2)

    logger.info("_______________________METRICS______________________")
    logger.info(f"1. CE Premium Movement: {ce_premium_movement} ")
    logger.info(f"2. PE Premium Movement: {pe_premium_movement}")
    logger.info(f"3. Combined Premium Movement: {combined_premiums}")
    logger.info(f"4. Net Delta Drift: {net_delta_drift}")
    logger.info(f"5. CE Long/Short Ratio: {ce_short_long}")
    logger.info(f"6. PE Long/Short Ratio: {pe_short_long}")
    logger.info(format_line)

def friday_till_expiry(date_str):
    # Parse the target date from string
    target_date = datetime.strptime(date_str, "%d-%b-%Y")
    
    # Get today's date
    today = datetime.today()
    
    # Initialize a counter for Fridays
    friday_count = 0
    
    # Loop from today to the target date
    current_date = today
    while current_date <= target_date:
        # Check if the current date is a Friday (weekday 4)
        if current_date.weekday() == 4:
            friday_count += 1
        # Move to the next day
        current_date += timedelta(days=1)
    
    return friday_count

def get_delta_thresholds(logger, global_vars):
    global Expiry
    friday_count = friday_till_expiry(global_vars.get('Expiry'))
    ICT = 43
    IFT= 50
    # if friday_count > 3:
    #     ICT = 50
    #     IFT=60
    # elif friday_count > 2:
    #     ICT = 40
    #     IFT=50
    # elif friday_count > 1:
    #     ICT = 36
    #     IFT=50
    # else:
    #     ICT = 33.33
    #     IFT=50
    # logger.info(f"ICT: {ICT} | IFT: {IFT}")
    return ICT, IFT

def require_safe_adjustments(logger, api, global_vars, strategy, delta, positions_df, profit_leg, loss_leg, pltp, cltp, symbol, expiry, minsp,maxsp, IC_delta_threshold, IF_delta_threshold, current_strike ):

    # IC_delta_threshold, IF_delta_threshold = get_delta_thresholds(logger, global_vars)
    ce_leg = positions_df[(positions_df['buy_sell']=='S')&(positions_df['ord_type']=='C')]
    pe_leg = positions_df[(positions_df['buy_sell']=='S')&(positions_df['ord_type']=='P')]
    ce_hedge = positions_df[(positions_df['buy_sell']=='B')&(positions_df['ord_type']=='C')]
    pe_hedge = positions_df[(positions_df['buy_sell']=='B')&(positions_df['ord_type']=='P')]

    qty = abs(int(positions_df.iloc[0]['qty']))

    is_ce_hedge = True if len(ce_hedge)==1 else False
    is_pe_hedge = True if len(pe_hedge)==1 else False

    if is_ce_hedge:
        ce_hedge_diff = abs(int(ce_leg['tsym'].iloc[0][-5:])-int(ce_hedge['tsym'].iloc[0][-5:]))
    if is_pe_hedge:
        pe_hedge_diff = abs(int(pe_leg['tsym'].iloc[0][-5:])-int(pe_hedge['tsym'].iloc[0][-5:]))
    
    H_strike=None
    exit_order_df=None
    new_delta = None

    if strategy=="SAFE":
        p_diff = current_strike-int(pe_leg['tsym'].iloc[0][-5:])
        c_diff = current_strike-int(ce_leg['tsym'].iloc[0][-5:])
        if p_diff<50:
            # Find lp diff between pleg and ce_sp_lp and ce_sp_plus_50_lp            
            ce_sp_lp = api.searchscrip(exchange='NSE', searchtext=ce_leg['tsym'].iloc[0][:-5]+(current_strike))['values'][0]
            ce_sp_plus_50_lp = api.searchscrip(exchange='NSE', searchtext=ce_leg['tsym'].iloc[0][:-5]+(current_strike+50)
            abs(pltp-ce_sp_lp) < abs(pltp-ce_sp_plus_50_lp)

            #If new Call SP and Loss leg SP is same then exit profit leg and make IF
            new_strike_price = int(L_tsym[-5:])
            if new_strike_price-int(pe_leg['tsym'].iloc[0][-5:])<=0:
                adjust=True
                exit_order_df = positions_df[positions_df.ord_type==profit_leg][['buy_sell','tsym','qty','remarks','netupldprc']]
                exit_order_df['buy_sell']=exit_order_df['buy_sell'].apply(lambda x: "S" if x=="B" else "B")
                new_order = {'buy_sell':"S",'tsym':L_tsym,'qty':qty,'remarks':"Adjustment Order", 'netupldprc':"0"}
                exit_order_df.loc[len(exit_order_df)+1] = new_order
        elif c_diff>-50:
            #Find the Put with lp closest to the loss leg
            search_ltp = cltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            #If new Put SP and Loss leg SP is same then exit profit leg and make IF
            new_strike_price = int(L_tsym[-5:])
            if new_strike_price-int(ce_leg['tsym'].iloc[0][-5:])>=0:
                adjust=True
                exit_order_df = positions_df[positions_df.ord_type==profit_leg][['buy_sell','tsym','qty','remarks','netupldprc']]
                exit_order_df['buy_sell']=exit_order_df['buy_sell'].apply(lambda x: "S" if x=="B" else "B")
                new_order = {'buy_sell':"S",'tsym':L_tsym,'qty':qty,'remarks':"Adjustment Order", 'netupldprc':"0"}
                exit_order_df.loc[len(exit_order_df)+1] = new_order

    elif strategy=="IF" and delta > IF_delta_threshold: 
        # Exit the loss making leg
        adjust=True
        exit_order_df = positions_df[positions_df.ord_type==loss_leg][['buy_sell','tsym','qty','remarks', 'netupldprc']]
        exit_order_df['buy_sell']=exit_order_df['buy_sell'].apply(lambda x: "S" if x=="B" else "B")
        #Find new legs
        if loss_leg=="C":
            search_ltp = pltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "CE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            if is_ce_hedge:
                H_strike = int(L_tsym[-5:])+ce_hedge_diff 
            new_delta = round(100*abs(float((L_lp-pltp)/(L_lp+pltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")
            rev_cstrike = int(L_tsym[-5:])
        else:
            search_ltp = cltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            if is_pe_hedge:
                H_strike = int(L_tsym[-5:])-pe_hedge_diff
            new_delta = round(100*abs(float((L_lp-cltp)/(L_lp+cltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")
            rev_pstrike = int(L_tsym[-5:])

        if new_delta < IC_delta_threshold:
            new_order = {'buy_sell':"S",'tsym':L_tsym,'qty':qty,'remarks':"Adjustment Order", 'netupldprc':"0"}
            exit_order_df.loc[len(exit_order_df)+1] = new_order
            if H_strike is not None:
                H_tsym, H_lp = get_nearest_strike_strike(odf, H_strike)
                new_order = {'buy_sell':"B",'tsym':H_tsym,'qty':qty,'remarks':"Adjustment Hedge Order", 'netupldprc':"0"}
                exit_order_df.loc[len(exit_order_df)+1] = new_order
        else:
            logger.info("EXIT TRADE: NEW DELTA > IC_DELTA_THRESHOLD")
            print("EXIT TRADE")

    elif strategy=="IC" and delta> IC_delta_threshold:
        #Exit Profit making leg
        adjust=True
        exit_order_df = positions_df[positions_df.ord_type==profit_leg][['buy_sell','tsym','qty','remarks','netupldprc']]
        exit_order_df['buy_sell']=exit_order_df['buy_sell'].apply(lambda x: "S" if x=="B" else "B")

        #Find new legs
        if profit_leg=="C":
            search_ltp = pltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "CE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp) 
            # Find loss_leg ATM
            loss_atm = int(positions_df[(positions_df.ord_type==loss_leg) & (positions_df.buy_sell=="S")].iloc[0]['tsym'][-5:]) 
            
            if int(L_tsym[-5:]) < loss_atm:
                new_strike_price =  loss_atm
            else:
                new_strike_price = int(L_tsym[-5:])
            
            L_tsym, L_lp = get_nearest_strike_strike(odf, new_strike_price)
            if is_ce_hedge:
                H_strike = new_strike_price+ce_hedge_diff

            rev_cstrike = int(L_tsym[-5:])

            new_delta = round(100*abs(float((L_lp-pltp)/(L_lp+pltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")
        else:
            search_ltp = cltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            # Find loss_leg ATM
            loss_atm = int(positions_df[(positions_df.ord_type==loss_leg) & (positions_df.buy_sell=="S")].iloc[0]['tsym'][-5:]) 

            if int(L_tsym[-5:]) > loss_atm:
                new_strike_price = loss_atm
            else:
                new_strike_price = int(L_tsym[-5:])

            L_tsym, L_lp = get_nearest_strike_strike(odf, new_strike_price)

            if is_pe_hedge:            
                H_strike = new_strike_price-pe_hedge_diff

            rev_pstrike = int(L_tsym[-5:])
            new_delta = round(100*abs(float((L_lp-cltp)/(L_lp+cltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")

            
        if new_delta < IF_delta_threshold:
            new_order = {'buy_sell':"S",'tsym':L_tsym,'qty':qty,'remarks':"Adjustment Order", 'netupldprc':"0"}
            exit_order_df.loc[len(exit_order_df)+1] = new_order
            if H_strike is not None:
                H_tsym, H_lp = get_nearest_strike_strike(odf, H_strike)
                new_order = {'buy_sell':"B",'tsym':H_tsym,'qty':qty,'remarks':"Adjustment Hedge Order", 'netupldprc':"0"}
                exit_order_df.loc[len(exit_order_df)+1] = new_order
        else:
            logger.info("EXIT TRADE: NEW DELTA > IF_DELTA_THRESHOLD")
            print("EXIT TRADE")

    return exit_order_df, new_delta


def require_adjustments(logger, api, global_vars, strategy, delta, positions_df, profit_leg, loss_leg, pltp, cltp, symbol, expiry, minsp,maxsp, IC_delta_threshold, IF_delta_threshold ):

    # IC_delta_threshold, IF_delta_threshold = get_delta_thresholds(logger, global_vars)
    ce_leg = positions_df[(positions_df['buy_sell']=='S')&(positions_df['ord_type']=='C')]
    pe_leg = positions_df[(positions_df['buy_sell']=='S')&(positions_df['ord_type']=='P')]
    ce_hedge = positions_df[(positions_df['buy_sell']=='B')&(positions_df['ord_type']=='C')]
    pe_hedge = positions_df[(positions_df['buy_sell']=='B')&(positions_df['ord_type']=='P')]

    qty = abs(int(positions_df.iloc[0]['qty']))

    is_ce_hedge = True if len(ce_hedge)==1 else False
    is_pe_hedge = True if len(pe_hedge)==1 else False

    if is_ce_hedge:
        ce_hedge_diff = abs(int(ce_leg['tsym'].iloc[0][-5:])-int(ce_hedge['tsym'].iloc[0][-5:]))
    if is_pe_hedge:
        pe_hedge_diff = abs(int(pe_leg['tsym'].iloc[0][-5:])-int(pe_hedge['tsym'].iloc[0][-5:]))
    
    H_strike=None
    exit_order_df=None
    new_delta = None

    if strategy=="IF" and delta > IF_delta_threshold: 
        # Exit the loss making leg
        adjust=True
        exit_order_df = positions_df[positions_df.ord_type==loss_leg][['buy_sell','tsym','qty','remarks', 'netupldprc']]
        exit_order_df['buy_sell']=exit_order_df['buy_sell'].apply(lambda x: "S" if x=="B" else "B")
        #Find new legs
        if loss_leg=="C":
            search_ltp = pltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "CE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            if is_ce_hedge:
                H_strike = int(L_tsym[-5:])+ce_hedge_diff 
            new_delta = round(100*abs(float((L_lp-pltp)/(L_lp+pltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")
            rev_cstrike = int(L_tsym[-5:])
        else:
            search_ltp = cltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            if is_pe_hedge:
                H_strike = int(L_tsym[-5:])-pe_hedge_diff
            new_delta = round(100*abs(float((L_lp-cltp)/(L_lp+cltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")
            rev_pstrike = int(L_tsym[-5:])

        if new_delta < IC_delta_threshold:
            new_order = {'buy_sell':"S",'tsym':L_tsym,'qty':qty,'remarks':"Adjustment Order", 'netupldprc':"0"}
            exit_order_df.loc[len(exit_order_df)+1] = new_order
            if H_strike is not None:
                H_tsym, H_lp = get_nearest_strike_strike(odf, H_strike)
                new_order = {'buy_sell':"B",'tsym':H_tsym,'qty':qty,'remarks':"Adjustment Hedge Order", 'netupldprc':"0"}
                exit_order_df.loc[len(exit_order_df)+1] = new_order
        else:
            logger.info("EXIT TRADE: NEW DELTA > IC_DELTA_THRESHOLD")
            print("EXIT TRADE")

    elif strategy=="IC" and delta> IC_delta_threshold:
        #Exit Profit making leg
        adjust=True
        exit_order_df = positions_df[positions_df.ord_type==profit_leg][['buy_sell','tsym','qty','remarks','netupldprc']]
        exit_order_df['buy_sell']=exit_order_df['buy_sell'].apply(lambda x: "S" if x=="B" else "B")

        #Find new legs
        if profit_leg=="C":
            search_ltp = pltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "CE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp) 
            # Find loss_leg ATM
            loss_atm = int(positions_df[(positions_df.ord_type==loss_leg) & (positions_df.buy_sell=="S")].iloc[0]['tsym'][-5:]) 
            
            if int(L_tsym[-5:]) < loss_atm:
                new_strike_price =  loss_atm
            else:
                new_strike_price = int(L_tsym[-5:])
            
            L_tsym, L_lp = get_nearest_strike_strike(odf, new_strike_price)
            if is_ce_hedge:
                H_strike = new_strike_price+ce_hedge_diff

            rev_cstrike = int(L_tsym[-5:])

            new_delta = round(100*abs(float((L_lp-pltp)/(L_lp+pltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")
        else:
            search_ltp = cltp
            odf = get_Option_Chain(api,logger, global_vars, symbol, expiry, minsp,maxsp, "PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            # Find loss_leg ATM
            loss_atm = int(positions_df[(positions_df.ord_type==loss_leg) & (positions_df.buy_sell=="S")].iloc[0]['tsym'][-5:]) 

            if int(L_tsym[-5:]) > loss_atm:
                new_strike_price = loss_atm
            else:
                new_strike_price = int(L_tsym[-5:])

            L_tsym, L_lp = get_nearest_strike_strike(odf, new_strike_price)

            if is_pe_hedge:            
                H_strike = new_strike_price-pe_hedge_diff

            rev_pstrike = int(L_tsym[-5:])
            new_delta = round(100*abs(float((L_lp-cltp)/(L_lp+cltp))),2)
            logger.info(f"New Delta of adjusted trade: {new_delta}")

            
        if new_delta < IF_delta_threshold:
            new_order = {'buy_sell':"S",'tsym':L_tsym,'qty':qty,'remarks':"Adjustment Order", 'netupldprc':"0"}
            exit_order_df.loc[len(exit_order_df)+1] = new_order
            if H_strike is not None:
                H_tsym, H_lp = get_nearest_strike_strike(odf, H_strike)
                new_order = {'buy_sell':"B",'tsym':H_tsym,'qty':qty,'remarks':"Adjustment Hedge Order", 'netupldprc':"0"}
                exit_order_df.loc[len(exit_order_df)+1] = new_order
        else:
            logger.info("EXIT TRADE: NEW DELTA > IF_DELTA_THRESHOLD")
            print("EXIT TRADE")

    return exit_order_df, new_delta

def past_time(t):
    # Get the current time
    now= datetime.now()
    delta = (t-now).total_seconds()
    # logger.info(format_line)
    # logger.info(f"Current Time: {now} | Checked for time: {t}")
    # logger.info(format_line)
    # Check if the current time is greater than t
    if delta< 0:
        return True
    else:
        return False

def find_best_position_to_enter(config):
    """Identify the best position to enter based on configuration."""
    # Analyze market data and strategy to determine entry point
    pass

def place_trade(order_details, live=True):
    """Place trade and confirm order."""
    if live:
        # Use Upstox API to place an order
        # Poll the trade book until confirmation
        pass
    else:
        # Send trade recommendation via email
        send_email("Trade Recommendation", f"Order Details: {order_details}")

def exit_position(current_position, profit_target, stop_loss):
    """Exit position if profit target or stop loss is hit."""
    # Check conditions and use Upstox API to exit position
    pass

def gather_metrics(current_position):
    """Gather relevant metrics like Delta, etc."""
    # Fetch and calculate metrics
    pass

def check_adjustment_needed(metrics, strategy, thresholds):
    """Determine if an adjustment is required."""
    # Logic to decide if an adjustment is feasible
    pass

def perform_adjustment(order_details):
    """Place adjustment order and confirm it."""
    # Use Upstox API for order adjustment
    pass

def upstox_calculate_breakevens(df, global_vars):
    # Calculate breakevens
    # qty = global_vars.get('lot_size')*global_vars.get('lots')
    df['total_credit'] = df['netupldprc'].astype(float).astype(int)
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    adj = global_vars.get('Past_M2M')/df['qty'].abs().mean()
    net_credit = round(float(df['total_credit'].sum()),2)
    higher_be = float(df[(df['ord_type']=="P")&(df['buy_sell']=="S")]['tsym'].iloc[0][-7:-2])+net_credit -adj
    lower_be = float(df[(df['ord_type']=="C")&(df['buy_sell']=="S")]['tsym'].iloc[0][-7:-2])-net_credit +adj
    return round(lower_be,0), round(higher_be,0)

def calculate_breakevens(df, global_vars):
    # Calculate breakevens
    qty = global_vars.get('lot_size')*global_vars.get('lots')
    df['total_credit'] = df['netupldprc'].astype(float).astype(int)
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    adj = global_vars.get('Past_M2M')/df['qty'].abs().mean()
    net_credit = round(float(df['total_credit'].sum()),2)
    higher_be = float(df[(df['ord_type']=="P")&(df['buy_sell']=="S")]['tsym'].iloc[0][-5:])+net_credit #-adj
    lower_be = float(df[(df['ord_type']=="C")&(df['buy_sell']=="S")]['tsym'].iloc[0][-5:])-net_credit #+adj
    return round(lower_be,0), round(higher_be,0)

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
    elif is_within_timeframe("07:00", "10:10"):
        return {"session": "session2","start_time": "07:00", "end_time": "10:10"}
    return None