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

live=True
# times=1.75
# stop_loss_per=0.5
exit_params = {
    day: {"distance_from_breakeven": dist, "loss_multiple": profit}
    for day, dist, profit in [
        (30, 3.28, 0.00),
        (29, 3.05, 0.00),
        (28, 2.83, 0.00),
        (27, 2.61, 0.00),
        (26, 2.41, 0.00),
        (25, 2.21, 0.02),
        (24, 2.02, 0.15),
        (23, 1.84, 0.27),
        (22, 1.66, 0.39),
        (21, 1.50, 0.50),
        (20, 1.34, 0.61),
        (19, 1.20, 0.72),
        (18, 1.06, 0.82),
        (17, 0.93, 0.92),
        (16, 0.81, 1.01),
        (15, 0.69, 1.10),
        (14, 0.59, 1.19),
        (13, 0.49, 1.27),
        (12, 0.40, 1.35),
        (11, 0.32, 1.19),
        (10, 0.25, 1.01),
        (9, 0.19, 0.92),
        (8, 0.13, 0.82),
        (7, 0.09, 0.61),
        (6, 0.05, 0.39),
        (5, 0.02, 0.15),
        (4, 0.00, 0.00),
        (3, 0.00, 0.00),
        (2, 0.00, 0.00),
        (1, 0.00, 0.00),
        (0, 0.00, 0.00),
    ]
}

def stop_loss_order(pos_df, api, live, sender_email, receiver_email, email_password):
    for i,pos in pos_df.iterrows():
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
    
    return userid, password, vendor_code, api_secret, imei, TOKEN, sender_email, receiver_email, email_password

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
    elif is_within_timeframe("07:00", "10:00"):
        return {"session": "session2","start_time": "07:00", "end_time": "10:00"}
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

def dict_to_table_manual(data):
    """Converts a dictionary to a formatted table string without external libraries."""
    max_key_length = max(len(str(k)) for k in data.keys())
    max_value_length = max(len(str(v)) for v in data.values())

    table = f"{'Key'.ljust(max_key_length)} | {'Value'.ljust(max_value_length)}\n"
    table += "-" * (max_key_length + max_value_length + 3) + "\n"

    for k, v in data.items():
        table += f"{str(k).ljust(max_key_length)} | {str(v).ljust(max_value_length)}\n"

    return table

import pandas as pd

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
        pos_df['expiry'] = pd.to_datetime(pos_df['expiry'])
        current_date = pd.Timestamp.today().normalize()
        pos_df['Days_to_Expiry'] = pos_df['expiry'].apply(lambda x: (x - current_date).days)
        pos_df['exit_breakeven_per']= pos_df.apply(lambda x: exit_params[x['Days_to_Expiry']]['distance_from_breakeven'],axis=1)
        pos_df['exit_loss_per']= pos_df.apply(lambda x: exit_params[x['Days_to_Expiry']]['loss_multiple'],axis=1)
        return pos_df
    except Exception as e:
        return None

def monitor_trade(api, sender_email, receiver_email, email_password):
    pos_df = get_positions(api)
    if pos_df is None:
        return {'get_pos Error':"Error getting position Info"} 
    total_pnl=0
    metrics = {"Total_PNL": total_pnl}
    expiry_metrics = {}
    current_index_price = float(api.get_quotes(exchange="NSE", token=str(26000))['lp'])
    
    for expiry, group in pos_df.groupby("expiry"):
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
        expiry_date_str = expiry.strftime('%Y-%m-%d')
        trade_hist_df = trade_hist_df[trade_hist_df['expiry']==expiry_date_str]
        # realized_qty = float(((trade_hist_df['daybuyqty'].astype(int)+trade_hist_df['cfsellqty'].astype(int))/2).sum())
        # realized_premium = float((trade_hist_df['upldprc'].astype(float)-trade_hist_df['totbuyavgprc'].astype(float)).sum())*realized_qty
        realized_premium = (trade_hist_df['upldprc'].astype(float)*trade_hist_df['cfsellqty'].astype(int)-trade_hist_df['totbuyavgprc'].astype(float)*trade_hist_df['daybuyqty'].astype(int)).sum()

        total_premium_collected_per_option = (current_premium + realized_premium) /current_qty
        current_pnl+=realized_premium
        max_profit+=realized_premium

        ce_rows = group[group["type"] == "CE"]
        pe_rows = group[group["type"] == "PE"]
        
        # if not ce_rows.empty and not pe_rows.empty:
        stop_loss_per = group['exit_loss_per'].mean().astype(float)
        max_loss = float(-1 * stop_loss_per * max_profit)
        if ce_rows.empty:
            ce_strike = 0
            upper_breakeven=999999
        else:
            ce_breakeven_factor = current_qty/(-1*ce_rows["netqty"].sum())
            ce_strike = float((ce_rows["sp"].astype(float) * ce_rows["netqty"].abs()).sum() / ce_rows["netqty"].abs().sum())
            upper_breakeven = float(ce_strike + total_premium_collected_per_option*ce_breakeven_factor - current_index_price * ce_rows['exit_breakeven_per'].mean().astype(float)/ 100)

        if pe_rows.empty:
            pe_strike = 0
            lower_breakeven = 0
        else:
            pe_breakeven_factor = current_qty/(-1*pe_rows["netqty"].sum())
            pe_strike = float((pe_rows["sp"].astype(float) * pe_rows["netqty"].abs()).sum() / pe_rows["netqty"].abs().sum())
            lower_breakeven = float(pe_strike - total_premium_collected_per_option*pe_breakeven_factor + current_index_price * pe_rows['exit_breakeven_per'].mean().astype(float)/ 100)
        
        if ce_strike!=0 and pe_strike!=0:
            breakeven_range = upper_breakeven - lower_breakeven
            near_breakeven = min(100 * (current_index_price - lower_breakeven) / current_index_price,  
                                100 * (upper_breakeven - current_index_price) / current_index_price)
        else:
            breakeven_range = 0
            near_breakeven = 0

        expiry_metrics[expiry] = {
            "PNL": round(current_pnl, 2),
            "CE_Strike": round(ce_strike, 2),
            "PE_Strike": round(pe_strike, 2),
            "Current_Index_Price": current_index_price,
            "Lower_Breakeven": round(lower_breakeven, 2),
            "Upper_Breakeven": round(upper_breakeven, 2),
            "Breakeven_Range": round(breakeven_range, 2),
            "Breakeven_Range_Per": round(100 * breakeven_range / current_index_price, 2),
            "Near_Breakeven": round(near_breakeven, 2),
            "Max_Profit": round(max_profit, 2),
            "Max_Loss": round(max_loss, 2)
        }
        total_pnl+=current_pnl
        stop_loss_condition = (current_index_price < lower_breakeven or current_index_price > upper_breakeven) and total_pnl < max_loss
        if stop_loss_condition:
            stop_loss_order(group, api, sender_email, receiver_email, email_password,live)
            expiry_metrics[expiry] = {
            "PNL": round(current_pnl, 2),
            "CE_Strike": round(ce_strike, 2),
            "PE_Strike": round(pe_strike, 2),
            "Current_Index_Price": current_index_price,
            "Lower_Breakeven": "STOP_LOSS",
            "Upper_Breakeven": "STOP_LOSS",
            "Breakeven_Range": "STOP_LOSS",
            "Breakeven_Range_Per": "STOP_LOSS",
            "Near_Breakeven": round(near_breakeven, 2),
            "Max_Profit": round(max_profit, 2),
            "Max_Loss": round(max_loss, 2)
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
                details.get("Lower_Breakeven", "N/A"), details.get("Upper_Breakeven", "N/A"),
                details.get("Breakeven_Range", "N/A"), details.get("Breakeven_Range_Per", "N/A"),
                details.get("Near_Breakeven", "N/A"), details.get("Max_Profit", "N/A"), details.get("Max_Loss", "N/A")
            ])
    
    df = pd.DataFrame(data, columns=[
        "Expiry", "PNL", "CE Strike", "PE Strike", "Current Index Price", "Lower Breakeven", 
        "Upper Breakeven", "Breakeven Range", "Breakeven %", "Near Breakeven", "Max Profit", "Max Loss"
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



def main():
    session = identify_session()
    if not session:
        print("No active trading session.")
        return
    
    # Login
    userid, password, vendor_code, api_secret, imei, TOKEN, sender_email, receiver_email, email_password = init_creds()
    api = login(userid, password, vendor_code, api_secret, imei, TOKEN)

    while is_within_timeframe("03:00", "03:45"):
        print("Initializing")
        sleep_time.sleep(60)

    counter=0

    # Start Monitoring
    while is_within_timeframe(session.get('start_time'), session.get('end_time')):
        metrics = monitor_trade(api, sender_email, receiver_email, email_password)
        
        if metrics =="STOP_LOSS":
            send_email(sender_email, receiver_email, email_password, "STOP LOSS HIT - QUIT", "STOP LOSS HIT")
        else:
        #     subject = f"FINVASIA: MTM:{metrics['Total_PNL']} | NEAR_BE:{metrics['Near_Breakeven']} | RANGE:{metrics['Breakeven_Range_Per']}| MAX_PROFIT:{metrics['Max_Profit']} | MAX_LOSS: {metrics['Max_Loss']}"
            if counter % 10 == 0:
                subject = "FINVASIA STATUS"
                metrics["INDIA_VIX"] = get_india_vix(api)
                email_body = format_trade_metrics(metrics)
                send_email(sender_email, receiver_email, email_password, subject, email_body)
            counter+=1
        sleep_time.sleep(60)
  
    # Logout
    api.logout()

if __name__ =="__main__":
    main()