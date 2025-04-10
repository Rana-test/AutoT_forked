import logging
from logging.handlers import RotatingFileHandler
from api_helper import ShoonyaApiPy
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import pyotp
import pandas as pd
from datetime import datetime, timedelta, date
import requests
import zipfile
import time
### Add condition for exception, when index moves way beyond adjustment
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

live = config['live']
target_profit = config['target_profit']
stop_loss = config['stop_loss']
enter_today = config['enter_today']
EntryType= config['EntryType']
nse_sym_path = config['nse_sym_path']
nfo_sym_path = config['nfo_sym_path']
Expiry = config['Expiry']
Symbol = config['Symbol']
min_strike_price = config['min_strike_price']
max_strike_price = config['max_strike_price']
lot_size = config['lot_size']
lots = config['lots']
Past_M2M = config['Past_M2M']
interval = config['interval']
num_adjustments=config['num_adjustments']
Entry_Date= config['Entry_Date']


# config['Update_EOD']

def save_config():
    global config
    # Update config
    with open("config.yml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

userid=None
# Global variables
#  Initialize the API
api = ShoonyaApiPy()
state = pd.read_csv('state.csv')
symbolDf= None
total_m2m = 0
delta=0
edate = Expiry.split("-")
tsym_prefix= Symbol+edate[0]+edate[1]+edate[2][-2:]
email_subject = "Trade Analytics: NO ACTION"
format_line = "--------------------------------------------------------"

# Logging Setup
logger = logging.getLogger('Auto_Trader')
logger.setLevel(logging.DEBUG)  # Set the logging level for the logger
# Rotating file handler (file size limit of 1 MB, keeps 5 backup files)
file_handler = RotatingFileHandler('logs/app.log', maxBytes=1_000_000, backupCount=5)
# Create a logging format
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

def send_custom_email(subject, body):
    # Email configuration
    sender_email = os.getenv("EMAIL_USER")
    receiver_email = os.getenv("EMAIL_TO")
    password = os.getenv("EMAIL_PASS")

    # Create email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Add body to email
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(sender_email, password)  # Login to your email account
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)  # Send the email
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        server.quit()

def clear_state(csv_file):
    columns = ['sno', 'm2m']
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file, index=False)

def load_state(csv_file):
    """Load state from CSV if it exists, otherwise initialize defaults."""
    global  target_profit
    if not os.path.exists(csv_file):
        columns = ['sno', 'm2m']
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
    df = pd.read_csv(csv_file)
    if len(df)>0:
        m_profit = df['m2m'].max()  # Max profit based on m2m column
        print(f"Loaded state: Max profit: {m_profit}")
        logger.info(f"Loaded state: Max profit: {m_profit}")
        return m_profit,  df
    else:
        # Initial state if CSV doesn't have any entry
        return 0, target_profit, False, df

def save_state(csv_file, df):
    """Save the current state to CSV using pandas."""
    new_row={'sno':len(df)+1,'m2m':total_m2m}
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(csv_file, index=False)

def calculate_breakevens(df, closed_m2m):
            # Calculate breakevens
        df['total_credit'] = df['netupldprc'].astype(float)*df['qty'].astype(int)/(lot_size*lots)
        adj = closed_m2m/(lot_size*lots)
        net_credit = round(float(df['total_credit'].sum()),2)
        lower_be = float(df[(df['ord_type']=="P")&(df['buy_sell']=="S")]['tsym'].iloc[0][13:])+net_credit-adj
        higher_be = float(df[(df['ord_type']=="C")&(df['buy_sell']=="S")]['tsym'].iloc[0][13:])-net_credit+adj
        return lower_be, higher_be

# Step 1: Preprocess data and extract necessary information
def get_current_positions():
    global total_m2m, config,email_subject
    # Fetch the latest data (positions, LTP, etc.)
    open_pos_data=[]
    ret = api.get_positions()
    if ret is None:
        logger.info("Issue fetching Positions")
        email_subject = "Issue fetching Positions..."
        return None, 0, 0
    else:
        mtm = 0
        pnl = 0
        for i in ret:
            # mtm += float(i['urmtom'])
            # pnl += float(i['rpnl'])
            # day_m2m = mtm + pnl
            # {'buy_sell':'B', 'tsym': pe_hedge, 'qty': lots*lot_size, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
            # rev_buy_sell = {"B": "C", "C": "B"}.get(i['buy_sell'])
            if i['tsym'][:5]==Symbol:
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

                open_pos_data.append({
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
        positions_df = pd.DataFrame(open_pos_data)
        
        if not positions_df.empty:
            # Calculate Total M2M
            closed_positions = positions_df[positions_df['buy_sell']=="NA"]
            closed_positions = closed_positions.copy() 
            closed_m2m=0
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

            open_positions = positions_df[~(positions_df['buy_sell']=="NA")]
            open_positions = open_positions.copy() 
            open_m2m=0
            if not open_positions.empty:
                open_positions['net_profit']=(open_positions['lp'].astype(float)-open_positions['netupldprc'].astype(float))*open_positions['qty'].astype(float)
                open_m2m =round(float(open_positions['net_profit'].sum()),2)
                logger.info(format_line)
                logger.info("<<<CURRENT POSITION>>>")
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    logger.info("\n%s",open_positions[['buy_sell', 'tsym', 'qty', 'netupldprc', 'lp']])

            total_m2m = closed_m2m+open_m2m+Past_M2M
            # eod = datetime.strptime("10:00:00", "%H:%M:%S").time()
            eod = datetime.combine(thread_start_time.date(), datetime.strptime(config['end_time'], "%H:%M:%S").time())
            eod_minus_one = eod - timedelta(minutes=1)
            if past_time(eod_minus_one):
                if config['Update_EOD']==0:
                    # Update Settled Amount
                    config['Update_EOD']=1
                    config['Past_M2M']=config['Past_M2M']+closed_m2m
                    exit(0)
            else:
                config['Update_EOD']=0
            save_config()
            return open_positions, total_m2m, closed_m2m
        else:
            return None, 0, 0

def get_revised_position():
    # Publish new Positions after 5 second wait
    time.sleep(10)
    rev_position, rev_m2m, closed_m2m = get_current_positions()
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

def execute_basket(orders_df):
    # {'buy_sell':'B', 'tsym': pe_hedge, 'qty': lots*lot_size, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
    # place_order(buy_sell, tsym, qty, remarks="regular order")
    for i, order in orders_df.iterrows():
        place_order(order['buy_sell'], order['tsym'], order['qty'], order['remarks'])


def exit_positions(orders_df):
    # {'buy_sell':'B', 'tsym': pe_hedge, 'qty': lots*lot_size, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
    # place_order(buy_sell, tsym, qty, remarks="regular order")
    for i, order in orders_df.iterrows():
        rev_buy_sell = {"B": "S", "S": "B"}.get(order['buy_sell'])
        place_order(rev_buy_sell, order['tsym'], abs(int(order['qty'])), order['remarks'])

def get_Option_Chain(type):
    global symbolDf
    if symbolDf is None:
        symbolDf = pd.read_csv(nfo_sym_path)
        symbolDf['hundred_strikes'] = symbolDf['TradingSymbol'].apply(lambda x: x[-2:])
    
    df=symbolDf[
        (symbolDf.OptionType==type) 
        & (symbolDf.Symbol==Symbol)
        & (symbolDf.Expiry==Expiry) 
        & (symbolDf['hundred_strikes']=="00") # Keeping it safe
        & (symbolDf['StrikePrice']>min_strike_price) & (symbolDf['StrikePrice']<max_strike_price)
        ]
    
    OList=[]
    for i in df.index:
        strikeInfo = df.loc[i]
        res=api.get_quotes(exchange="NFO", token=str(strikeInfo.Token))
        res={'oi':res.get('oi',0), 'tsym':res['tsym'], 'lp':float(res['lp']), 'lotSize':strikeInfo.LotSize, 'token':res['token'], 'StrikePrice':int(float(res['strprc']))}
        OList.append(res)
    Ostrikedf = pd.DataFrame(OList)
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

    support = int(pedf.iloc[0]['tsym'][13:])
    support_oi = pedf.iloc[0]['oi']
    resistance = int(cedf.iloc[0]['tsym'][13:])
    resistance_oi = cedf.iloc[0]['oi']
    if (support+resistance)/2 % 50 == 0.0:
        atm = (support+resistance)/2
    elif support_oi>resistance_oi:
        atm = (support+resistance-50)/2
    else:
        atm = (support+resistance+50)/2
    return atm

def calculate_initial_positions(base_strike_price, CEOptdf, PEOptdf):
    ce_sell, ce_premium = get_nearest_strike_strike(CEOptdf, base_strike_price)
    pe_sell, pe_premium = get_nearest_strike_strike(PEOptdf, base_strike_price)
    tot_premium=round(pe_premium+ce_premium,2)
    pe_breakeven = base_strike_price - tot_premium
    ce_breakeven = base_strike_price + tot_premium
    ce_hedge, ce_hedge_premium = get_nearest_strike_strike(CEOptdf, ce_breakeven)
    pe_hedge, pe_hedge_premium = get_nearest_strike_strike(PEOptdf,pe_breakeven)
    orders_df = pd.DataFrame([ 
        {'buy_sell':'B', 'tsym': pe_hedge, 'qty': lots*lot_size, 'remarks':f'Initial PE Hedg with premium: {pe_hedge_premium}'},
        {'buy_sell':'S', 'tsym': pe_sell,  'qty': lots*lot_size, 'remarks':f'Initial PE Sell with premium: {pe_premium}'},
        {'buy_sell':'S', 'tsym': ce_sell,  'qty': lots*lot_size, 'remarks':f'Initial CE Sell with premium: {ce_premium}'},
        {'buy_sell':'B', 'tsym': ce_hedge, 'qty': lots*lot_size, 'remarks':f'Initial CE Hedg with premium: {ce_hedge_premium}'},
    ])
    span_res = api.span_calculator(userid,[
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"PE","strprc":str(pe_hedge[13:])+".00","buyqty":str(lots*lot_size),"sellqty":"0","netqty":"0"},
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"CE","strprc":str(ce_hedge[13:])+".00","buyqty":str(lots*lot_size),"sellqty":"0","netqty":"0"},
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"PE","strprc":str(pe_sell[13:])+".00","buyqty":"0","sellqty":str(lots*lot_size),"netqty":"0"},
        {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":Expiry,"optt":"CE","strprc":str(ce_sell[13:])+".00","buyqty":"0","sellqty":str(lots*lot_size),"netqty":"0"}
    ])
    trade_margin=float(span_res['span_trade']) + float(span_res['expo_trade'])
    cash_margin = (pe_hedge_premium+ ce_hedge_premium)*lots*lot_size
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
    

def enter_trade():
    global email_subject
    # global CEstrikedf, PEstrikedf
    email_subject="<<ENTERING TRADE>>"
    print("Getting CE Option Chain...")
    CEOptdf=get_Option_Chain("CE")
    print("Getting PE Option Chain...")
    PEOptdf=get_Option_Chain("PE")

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
    Oi_ord_df, oi_net_premium, oi_margin = calculate_initial_positions(atm, CEOptdf, PEOptdf)
    Oi_ord_df.sort_values(by='buy_sell', inplace=True)
    # if net_premium>max_net_premium:
    #     max_net_premium= net_premium
    #     best_entry="OI"
    #     best_ord_df=Oi_ord_df

    print("Getting Future Price")
    future_price_df = symbolDf[(symbolDf.Symbol=="NIFTY")&(symbolDf.Expiry==Expiry) &(symbolDf['OptionType']=="XX")]
    res=api.get_quotes(exchange="NFO", token=str(future_price_df.iloc[0].Token))
    future_strike = float(res['lp'])
    print("Positions based on Future Price")
    logger.info(format_line)
    logger.info("Positions based on Future Price")
    Fut_ord_df, f_net_premium, f_margin = calculate_initial_positions(future_strike, CEOptdf, PEOptdf)
    Fut_ord_df.sort_values(by='buy_sell', inplace=True)
    if f_net_premium>max_net_premium:
        max_net_premium= f_net_premium
        best_entry="FUTURE"
        best_ord_df=Fut_ord_df
        auto_margin = f_margin

    # Get initial trade basis current price
    print("Getting Current Price")
    nse_df = pd.read_csv(nse_sym_path)
    nifty_nse_token = nse_df[(nse_df.Symbol=="Nifty 50")&(nse_df.Instrument=="INDEX")].iloc[0]['Token']
    res=api.get_quotes(exchange="NSE", token=str(nifty_nse_token))
    current_strike = float(res['lp'])
    logger.info(format_line)
    print("Positions based on Current Price")
    logger.info("Positions based on Current Price")
    Curr_ord_df, curr_net_premium, curr_margin = calculate_initial_positions(current_strike, CEOptdf, PEOptdf)
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
    delta_oc['delta_diff'] = abs(delta_oc[(delta_oc['lp_x']>0) &(delta_oc['lp_y']>0)]['lp_x']-delta_oc[(delta_oc['lp_x']>0) &(delta_oc['lp_y']>0)]['lp_y'])
    delta_oc.sort_values(by='delta_diff', inplace=True)
    delta_atm = delta_oc['StrikePrice'].iloc[0]
    Delta_ord_df, d_net_premium, d_margin = calculate_initial_positions(delta_atm, CEOptdf, PEOptdf)
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
    Comb_ord_df, comb_net_premium, comb_margin = calculate_initial_positions(comb_atm, CEOptdf, PEOptdf)
    Comb_ord_df.sort_values(by='buy_sell', inplace=True)
    if comb_margin>max_net_premium:
        max_net_premium= comb_net_premium
        best_entry="COMBINED"
        best_ord_df=Comb_ord_df
        auto_margin = comb_margin

    # Execute trade:
    logger.info(format_line)
    if EntryType=="AUTO":
        logger.info(f"AUTO: Placing Order as per {best_entry}")
        print(f"AUTO: Placing Order as per {best_entry}")
        execute_basket(best_ord_df)
    elif EntryType== "CURRENT":
        logger.info("Placing Order as per Current Values")
        execute_basket(Curr_ord_df)
    elif EntryType== "FUTURE":
        logger.info("Placing Order as per Future Values")
        execute_basket(Fut_ord_df)
    elif EntryType== "COMBINED":
        logger.info("Placing Order as per Combined Values")
        execute_basket(Comb_ord_df)
    elif EntryType== "DELTA":
        logger.info("Placing Order as per Delta Neutral")
        execute_basket(Delta_ord_df)
    elif EntryType== "OI":
        logger.info("Placing Order as per OI")
        execute_basket(Oi_ord_df)

    email_subject = '<<<<<<<< ENTRY MADE >>>>>>>>>>>>'
    current_time = datetime.now()
    updated_time = current_time + timedelta(hours=5, minutes=30)
    extracted_date = updated_time.date()
    config['Entry_Date']=str(extracted_date)
    save_config()
    clear_state('state.csv')

    return

def count_working_days():
    global Entry_Date
    # Generate a range of dates from start_date to end_date
    if Entry_Date==0:
        Entry_Date= datetime.now()+ timedelta(hours=5, minutes=30)
    current_time = datetime.now()
    updated_time = current_time + timedelta(hours=5, minutes=30)
    date_range = pd.date_range(start=Entry_Date, end=updated_time.date(),freq='B')  # 'B' for business days
    return len(date_range)

def place_order(buy_sell, tsym, qty, remarks="regular order"):
    prd_type = 'M'
    exchange = 'NFO' 
    disclosed_qty= lot_size
    price_type = 'MKT'
    price=0
    trigger_price = None
    retention='DAY'
    if live:
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

def calculate_delta(df):
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

    pstrike = int(put_order.iloc[0]['tsym'][13:])
    cstrike = int(call_order.iloc[0]['tsym'][13:])

# if distance between put and call strike is less than min(dist between put and current_strike , dist between call and current_strike) consider strategy as IF
    print("Getting Current Price")
    nse_df = pd.read_csv(nse_sym_path)
    nifty_nse_token = nse_df[(nse_df.Symbol=="Nifty 50")&(nse_df.Instrument=="INDEX")].iloc[0]['Token']
    res=api.get_quotes(exchange="NSE", token=str(nifty_nse_token))
    current_strike = float(res['lp'])

    strategy = 'IF' if abs(pstrike-cstrike) < min(abs(current_strike-pstrike), abs(current_strike-cstrike)) else 'IC'
    # Special condition
    if pstrike==cstrike and cstrike==current_strike:
        strategy = 'IF'

    put_hedge = df[(df.buy_sell=="B")&(df.ord_type=="P")]
    call_hedge = df[(df.buy_sell=="B")&(df.ord_type=="C")] 

    p_hedge_strike = int(put_hedge.iloc[0]['tsym'][13:])
    c_hedge_strike = int(call_hedge.iloc[0]['tsym'][13:])
    pe_hedge_diff= pstrike-p_hedge_strike
    ce_hedge_diff= c_hedge_strike-cstrike

    return delta, pltp, cltp, profit_leg, loss_leg, strategy, pe_hedge_diff, ce_hedge_diff, current_strike, pstrike, cstrike

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

# Main monitoring and trading function
def monitor_and_execute_trades():
    global delta
    global email_subject
    global num_adjustments
    logger.info(f"### Day #: {count_working_days()} ###")
    # Capture days into the trade and exit in the expiry week
    # Capture total number of adjustments and exit if counter exceeds some value

    # Get Positions
    positions_df, m2m, closed_m2m = get_current_positions()

    # Step 1: Create positions on Day 1
    if positions_df is None or positions_df.empty:
        # noon = datetime.strptime("06:30:00", "%H:%M:%S").time()
        # if (check_day_after_last_thursday() and past_time(noon)) or enter_today :
        # Disabling auto entry
        if enter_today :
            enter_trade()
            # Publish new Positions after 5 second wait
            get_revised_position()
        return
    elif len(positions_df)!=4:
        email_subject = f'!!!! POSITIONS ERROR: Found {len(positions_df)} positions !!!!'
        logger.info(format_line)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info("\n%s",positions_df[['buy_sell', 'tsym', 'qty', 'upldprc', 'lp']])
        logger.info(f'M2M: {m2m}')
        return
    
    # Get breakevens
    lower_be, higher_be = calculate_breakevens(positions_df, closed_m2m)
    logger.info(f"LOWER BE: {lower_be}, HIGHER BE: {higher_be}")

    # Step 2: Check adjustment signal
    delta, pltp, cltp, profit_leg, loss_leg, strategy, pe_hedge_diff, ce_hedge_diff, current_strike, pstrike, cstrike = calculate_delta(positions_df)

    print(delta, pltp, cltp, profit_leg, loss_leg, strategy, pe_hedge_diff, ce_hedge_diff, current_strike, pstrike, cstrike)

    calculate_metrics(positions_df)

    # Calculate max_profit and exit if condition met
    max_profit = float((positions_df['qty'].astype(int)*positions_df['netupldprc'].astype(float)).sum()) * -1 + closed_m2m
    if auto_exit(max_profit, strategy, m2m, positions_df):
        return
    
    email_subject = f'DELTA: {delta}% | M2M: {m2m} | SP: {current_strike} | Strategy: {strategy}'

    adjust=False
    L_tsym=None
    H_tsym=None
    L_lp = None
    H_lp = None
    rev_pstrike = pstrike
    rev_cstrike = cstrike
    IC_delta_threshold, IF_delta_threshold = get_delta_thresholds()

    if strategy=="IF" and delta > IF_delta_threshold: 
        # Exit the loss making leg
        adjust=True
        exit_order_df = positions_df[positions_df.ord_type==loss_leg][['buy_sell','tsym','qty','remarks', 'netupldprc']]
        #Find new legs
        if loss_leg=="C":
            search_ltp = pltp
            odf = get_Option_Chain("CE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            H_strike = int(L_tsym[13:])+ce_hedge_diff 
            new_delta = round(100*abs(L_lp-pltp)/(L_lp+pltp),2)
            rev_cstrike = int(L_tsym[13:])
        else:
            search_ltp = cltp
            odf = get_Option_Chain("PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            H_strike = int(L_tsym[13:])-pe_hedge_diff
            new_delta = round(100*abs(L_lp-cltp)/(L_lp+cltp),2)
            rev_pstrike = int(L_tsym[13:])

        H_tsym, H_lp = get_nearest_strike_strike(odf, H_strike)

    elif strategy=="IC" and delta> IC_delta_threshold:
        #Exit Profit making leg
        adjust=True
        exit_order_df = positions_df[positions_df.ord_type==profit_leg][['buy_sell','tsym','qty','remarks','netupldprc']]

        #Find new legs
        if profit_leg=="C":
            search_ltp = pltp
            odf = get_Option_Chain("CE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp) 
            # Find loss_leg ATM
            loss_atm = int(positions_df[(positions_df.ord_type==loss_leg) & (positions_df.buy_sell=="S")].iloc[0]['tsym'][13:]) 
            
            if int(L_tsym[13:])> loss_atm:
                new_strike_price = int(L_tsym[13:])  
            else:
                new_strike_price = loss_atm
            
            L_tsym, L_lp = get_nearest_strike_strike(odf, new_strike_price)
            H_strike = new_strike_price+ce_hedge_diff

            rev_cstrike = int(L_tsym[13:])

            new_delta = round(100*abs(L_lp-pltp)/(L_lp+pltp),2)
        else:
            search_ltp = cltp
            odf = get_Option_Chain("PE")
            L_tsym, L_lp = get_nearest_price_strike(odf, search_ltp)
            # Find loss_leg ATM
            loss_atm = int(positions_df[(positions_df.ord_type==loss_leg) & (positions_df.buy_sell=="S")].iloc[0]['tsym'][13:]) 

            if int(L_tsym[13:]) < loss_atm:
                new_strike_price = int(L_tsym[13:])  
            else:
                new_strike_price = loss_atm

            L_tsym, L_lp = get_nearest_strike_strike(odf, new_strike_price)
            H_strike = new_strike_price-pe_hedge_diff

            rev_pstrike = int(L_tsym[13:])

            new_delta = round(100*abs(L_lp-cltp)/(L_lp+cltp),2)

        H_tsym, H_lp = get_nearest_strike_strike(odf, H_strike)

    if adjust:
        logger.info(format_line)
        logger.info("RECOMMENDED ADJUSTMENT:")
        for i, order in exit_order_df.iterrows():
            rev_buy_sell = {"B": "S", "S": "B"}.get(order['buy_sell'])
            logger.info(f"{rev_buy_sell} | {order['tsym']} | {lots*lot_size} | Original Order")
        logger.info(f"B | {H_tsym} | {lots*lot_size} | {H_lp} | Adjustment Hedge order")
        logger.info(f"S | {L_tsym} | {lots*lot_size} | {L_lp} | Adjustment Sell order")

        logger.info(f"ORIGINAL DELTA: {delta}%")
        logger.info(f"REVISED DELTA: {new_delta}%")
        logger.info(format_line)
        #Check if new adjustment reduces the max_profit and hence triggers an early exit before making adjustments

        # Initial Max Profit
        max_profit = float((positions_df['qty'].astype(int)*positions_df['netupldprc'].astype(float)).sum()) * -1 + closed_m2m
        # Less exit order sum
        exit_order_sum = float((exit_order_df['qty'].astype(int)*exit_order_df['netupldprc'].astype(float)).sum()) * -1
        # Plus new order sum
        new_order_sum = (L_lp - H_lp) *lots*lot_size
        revised_max_profit = max_profit - exit_order_sum + new_order_sum
        logger.info(f" Initial Max Profit: {max_profit} | Exit Order Sum: {exit_order_sum} | New Order Sum: {new_order_sum} | Revised Max Profit: {revised_max_profit}")
        #find new strategy
        revised_strategy = 'IF' if abs(rev_pstrike-rev_cstrike) < min(abs(current_strike-rev_pstrike), abs(current_strike-rev_cstrike)) else 'IC'
        # Special condition
        if rev_pstrike==rev_cstrike and rev_cstrike==current_strike:
            revised_strategy = 'IF'
        if auto_exit(revised_max_profit, revised_strategy, m2m, positions_df):
            return

        # Check if adjustment is not possible
        # If new Delta is higher than delta threshhold or new adjstment same as existing positions
        if (strategy =="IC" and new_delta > IF_delta_threshold) or (strategy =="IF" and new_delta > IC_delta_threshold) or (L_tsym == exit_order_df[exit_order_df['buy_sell']=='B'].iloc[0]['tsym']):
            logger.info(format_line)
            logger.info("Cannot find better adjustment.. Exiting Trade")
            email_subject = f'<<< CANNOT ADJUST. EXIT TRADE | M2M: {m2m} >>>'
            exit_positions(positions_df[['buy_sell','tsym','qty','remarks']])
            logger.info(format_line)
            return
        # Check revised max profit after adjustment and exit if target profit/stop loss hit 
        else:
            # Exit and create adjustment Legs:
            email_subject = f"*ADJUSTMENT* | DELTA: {delta}% | M2M: {m2m} | Revised DELTA: {new_delta}% "
            logger.info(format_line)
            logger.info("<<<ADJUSTMENTS>>>")
            exit_positions(exit_order_df)
            # Place leg and hedge orders
            place_order("B", H_tsym, lots*lot_size, remarks="Adjustment Hedge order")
            place_order("S", L_tsym, lots*lot_size, remarks="Adjustment Sell order")
            logger.info(f"REVISED DELTA: {new_delta}%")
            # Update num adjustments
            num_adjustments+=1
            config['num_adjustments']=num_adjustments
            save_config()
            # Publish new Positions after 5 second wait
            get_revised_position()
            logger.info(format_line)
    

# Function to check if today is the first day of the month (example)
def get_last_thursday(year, month):
    # Get the last day of the current month
    next_month = month % 12 + 1
    next_year = year + (month // 12)
    last_day_of_month = date(next_year if next_month == 1 else year, next_month, 1) - timedelta(days=1)
    
    # Find the last Thursday of the month
    while last_day_of_month.weekday() != 3:  # Thursday is weekday 3
        last_day_of_month -= timedelta(days=1)
    
    return last_day_of_month

def check_day_after_last_thursday():
    today = date.today()
    
    # Get the last Thursday of the current month
    last_thursday_current_month = get_last_thursday(today.year, today.month)

    # If the last Thursday is in the future, get the last Thursday of the previous month
    if last_thursday_current_month > today:
        last_thursday_current_month = get_last_thursday(today.year, today.month - 1)
    
    # Check if today is one day after the last Thursday
    return today == last_thursday_current_month + timedelta(days=1)

def auto_exit(max_profit, strategy, m2m, positions_df):
    global num_adjustments, email_subject 
    # Reducing max_profit_percent by 3% for every adjustment
    auto_target = max_profit * (int(config['percent_of_max_profit'])-3*num_adjustments)/100
    # Divide 1.5 targets for Iron Fly 
    if strategy =="IF":
        auto_target=auto_target/1.5
    logger.info(f"Max profit = {max_profit} | Auto target = {auto_target}")
    logger.info(f"if {m2m} > {auto_target} or {m2m}> {target_profit}:")
    logger.info(f"if {m2m} < -1 * {auto_target} or {m2m} < {stop_loss}")
    if m2m > auto_target or m2m> target_profit:
        logger.info(format_line)
        logger.info("Target Profit Acheived. Exit Trade")
        email_subject = f'<<< TARGET PROFIT ACHIEVED. EXIT TRADE | M2M: {m2m} >>>'   
        exit_positions(positions_df[['buy_sell','tsym','qty','remarks']])
        get_revised_position()
        logger.info(format_line)
        return True
    elif m2m < -1 * auto_target or m2m < stop_loss:
        # Implement traling stop loss
        logger.info(format_line)
        logger.info("Stop Loss hit. Exit Trade")
        email_subject = f'<<< STOP LOSS HIT. EXIT TRADE | M2M: {m2m} >>>'
        exit_positions(positions_df[['buy_sell','tsym','qty','remarks']])
        logger.info(format_line)
        return True
    else:
        return False


def login():
    global userid
    # TOKEN = os.getenv("TOKEN")
    # userid=os.getenv("userid")
    # password=os.getenv("password")
    # vendor_code=os.getenv("vendor_code")
    # api_secret=os.getenv("api_secret")
    # imei=os.getenv("imei")
    
    TOKEN = 'HWWYD7R4EPSW4I23H4634R336XALMED6'
    userid='FA417461' 
    password='NewIden@123' 
    vendor_code='FA417461_U'
    api_secret='456cdec44eae982782376e77101a6698'
    imei='abc1234'

    twoFA = pyotp.TOTP(TOKEN).now()

    login_response = api.login(userid=userid, password=password, twoFA=twoFA, vendor_code=vendor_code, api_secret=api_secret, imei=imei)   
    if login_response['stat'] == 'Ok':
        print('Logged in sucessfully')
    else:
        logger.info(f"Login failed: {login_response.get('emsg', 'Unknown error')}")
        print('Logged in failed')

def monitor_loop():
    global email_subject
    open('logs/app.log', 'w').close()
    email_subject ="Initializing..."
    monitor_and_execute_trades()
    # Send mail with log information
    with open('logs/app.log', 'r') as f:
        body = f.read() 
        # Send the email
        if live:
            send_custom_email('|||LIVE|||'+ email_subject, body)
        else:
            send_custom_email('|||DUMMY|||'+ email_subject, body)
            
    # Clear Logs
    open('logs/app.log', 'w').close()

def get_delta_thresholds():
    global Expiry
    friday_count = friday_till_expiry(Expiry)
    ICT = 36
    IFT=50
    if friday_count > 3:
        ICT = 50
        IFT=60
    elif friday_count > 2:
        ICT = 40
        IFT=50
    elif friday_count > 1:
        ICT = 36
        IFT=50
    else:
        ICT = 33.33
        IFT=50
    logger.info(f"ICT: {ICT} | IFT: {IFT}")
    return ICT, IFT

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

def calculate_metrics(df):
    puth_order = df[(df.buy_sell=="B")&(df.ord_type=="P")] 
    callh_order = df[(df.buy_sell=="B")&(df.ord_type=="C")] 

    phentry = float(puth_order.iloc[0]['netupldprc'])
    chentry = float(callh_order.iloc[0]['netupldprc'])

    phltp= float(puth_order.iloc[0]['lp'])
    chltp= float(callh_order.iloc[0]['lp'])

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

    # CE/HCE: 100 * (centry-clp)/(chlp-chentry) << ++ is good
    ce_short_long = round(100 * (centry-cltp)/(chltp-chentry),2)

    # PE/HPE: 100 * (pentry-plp)/(phlp-phentry) << ++ is good
    pe_short_long = round(100 * (pentry-pltp)/(phltp-phentry),2)

    logger.info("_______________________METRICS______________________")
    logger.info(f"1. CE Premium Movement: {ce_premium_movement} ")
    logger.info(f"2. PE Premium Movement: {pe_premium_movement}")
    logger.info(f"3. Combined Premium Movement: {combined_premiums}")
    logger.info(f"4. Net Delta Drift: {net_delta_drift}")
    logger.info(f"5. CE Short/Long Ratio: {ce_short_long}")
    logger.info(f"6. PE Short/Long Ratio: {pe_short_long}")
    logger.info(format_line)



# Call the main function periodically to monitor and execute trades
if __name__=="__main__":
    thread_start_time = datetime.now()
    # Define start and end times as datetime objects on today's date
    start_time = datetime.combine(thread_start_time.date(), datetime.strptime(config['start_time'], "%H:%M:%S").time())
    mid_time = datetime.combine(thread_start_time.date(), datetime.strptime('06:44:00', "%H:%M:%S").time())
    end_time = datetime.combine(thread_start_time.date(), datetime.strptime(config['end_time'], "%H:%M:%S").time())
    # start_time = datetime.strptime(config['start_time'], "%H:%M:%S")
    # end_time = datetime.strptime(config['end_time'], "%H:%M:%S")

    if end_time<mid_time:
        end_time = mid_time

    print(start_time, end_time, thread_start_time,past_time(start_time),past_time(end_time))

    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Login to Shoonya app
    print('Logging in ...')
    login()
    # monitor_loop() # For single execution
    # exit(0) # For single execution
    try:
        while past_time(start_time) and not past_time(end_time):
            monitor_loop()
            time.sleep(interval)
    except Exception as e:
        print(e)
    finally:
        api.logout()
        print("logged out")
