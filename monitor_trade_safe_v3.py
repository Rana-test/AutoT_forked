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
        (11, 0.32, 1.43),
        (10, 0.25, 1.50),
        (9, 0.19, 1.57),
        (8, 0.13, 1.63),
        (7, 0.09, 1.69),
        (6, 0.05, 1.75),
        (5, 0.02, 1.80),
        (4, 0.00, 1.85),
        (3, 0.00, 1.89),
        (2, 0.00, 1.93),
        (1, 0.00, 1.97),
        (0, 0.00, 2.00),
    ]
}

def stop_loss_order(pos_df, api, live=False):
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

def get_india_vix():
    try:
        # Define the ticker symbol for India VIX
        ticker_symbol = '^INDIAVIX'

        # Fetch the data for India VIX
        india_vix = yf.Ticker(ticker_symbol)

        # Get the latest market data
        vix_data = india_vix.history(period='1d')

        # Extract and print the closing price, which represents the latest value
        if not vix_data.empty:
            current_vix_value = vix_data['Close'].iloc[-1]
            # print(f"Current India VIX value: {current_vix_value}")
        else:
            # print("Failed to retrieve India VIX data.")
            current_vix_value=0
        return current_vix_value
    except Exception as e:
        print(f"Error fetching India VIX data: {e}")
        return 0

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

def monitor_trade(api):
    pos_df = get_positions(api)
    if pos_df is None:
        return {'get_pos Error':"Error getting position Info"} 
    
    total_pnl = round(float(pos_df["PnL"].sum()),2)
    metrics = {"Total_PNL": total_pnl}
    
    expiry_metrics = {}
    current_index_price = float(api.get_quotes(exchange="NSE", token=str(26000))['lp'])
    
    for expiry, group in pos_df.groupby("expiry"):
        current_pnl = float((-1 * (group["netupldprc"].astype(float)-group["lp"].astype(float)) * group["netqty"].astype(float)).sum())
        max_profit = float((-1 * group["netupldprc"].astype(float) * group["netqty"].astype(float)).sum())
        # total_premium_collected = (group["totsellamt"] / abs(group["netqty"])).sum()
        total_premium_collected = group["totsellamt"].sum() / group["netqty"].abs().sum() if group["netqty"].abs().sum() else 0
        
        ce_rows = group[group["type"] == "CE"]
        pe_rows = group[group["type"] == "PE"]
        
        if not ce_rows.empty and not pe_rows.empty:
            stop_loss_per = group['exit_loss_per'].mean().astype(float)
            max_loss = float(-1 * stop_loss_per * max_profit)
            ce_strike = float((ce_rows["sp"].astype(float) * ce_rows["netqty"].abs()).sum() / ce_rows["netqty"].abs().sum())
            pe_strike = float((pe_rows["sp"].astype(float) * pe_rows["netqty"].abs()).sum() / pe_rows["netqty"].abs().sum())
            
            total_lots = ce_rows["netqty"].abs().sum() + pe_rows["netqty"].abs().sum()
            avg_premium_per_lot = total_premium_collected / total_lots if total_lots else 0
            
            upper_breakeven = float(ce_strike + avg_premium_per_lot - current_index_price * ce_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            lower_breakeven = float(pe_strike - avg_premium_per_lot + current_index_price * pe_rows['exit_breakeven_per'].mean().astype(float)/ 100)
            breakeven_range = upper_breakeven - lower_breakeven
            near_breakeven = min(100 * (current_index_price - lower_breakeven) / current_index_price,  
                                 100 * (upper_breakeven - current_index_price) / current_index_price)
            
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
            
            stop_loss_condition = (current_index_price < lower_breakeven or current_index_price > upper_breakeven) and total_pnl < max_loss
            if stop_loss_condition:
                stop_loss_order(group, api, live)
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
        
        else:
            expiry_metrics[expiry] = {"Error": "Incomplete CE or PE data for this expiry"}
    
    metrics["Expiry_Details"] = expiry_metrics
    
    
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
        <h2>Options Trading Metrics</h2>
        <p><strong>Total PNL:</strong> {total_pnl}</p>
        <p><strong>INDIA VIX:</strong> {india_vix}</p>
        {table_html}
    </body>
    </html>
    """
    
    return email_body

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
        metrics = monitor_trade(api)
        
        if metrics =="STOP_LOSS":
            send_email(sender_email, receiver_email, email_password, "STOP LOSS HIT - QUIT", "STOP LOSS HIT")
        else:
        #     subject = f"FINVASIA: MTM:{metrics['Total_PNL']} | NEAR_BE:{metrics['Near_Breakeven']} | RANGE:{metrics['Breakeven_Range_Per']}| MAX_PROFIT:{metrics['Max_Profit']} | MAX_LOSS: {metrics['Max_Loss']}"
            if counter % 10 == 0:
                subject = "FINVASIA STATUS"
                metrics["INDIA_VIX"] = round(get_india_vix(), 2)
                email_body = format_trade_metrics(metrics)
                send_email(sender_email, receiver_email, email_password, subject, email_body)
            counter+=1
        sleep_time.sleep(60)
  
    # Logout
    api.logout()

if __name__ =="__main__":
    main()