import streamlit as st
import boto3
import time
import io
import pandas as pd  #DateField
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
import runpy
import streamlit as st
from datetime import datetime, timedelta
import calendar


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FPLP-Active", layout="wide")


##############################################################################################################################
################################## ----- Setting up the Database Connection ----- ############################################
##############################################################################################################################

# ✅ Access from secrets
aws_config = st.secrets["aws"]

# ✅ Extract values
aws_region = aws_config["region_name"]
aws_access_key = aws_config["aws_access_key_id"]
aws_secret_key = aws_config["aws_secret_access_key"]
aws_session_token = aws_config["aws_session_token"]

# ✅ Create boto3 session
session = boto3.Session(
    aws_access_key_id=aws_config["aws_access_key_id"],
    aws_secret_access_key=aws_config["aws_secret_access_key"],
    aws_session_token=aws_config["aws_session_token"],
    region_name=aws_config["region_name"]
)

@st.cache_data
def load_data(query):
    try:
        start_time = time.time()
        # Create an Athena client
        athena_client = boto3.client(
            "athena",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
            aws_session_token=aws_session_token
        )
        print("Query started")
        # Start the query execution
        query_response = athena_client.start_query_execution(
            QueryString=query,
            ResultConfiguration={
                "OutputLocation": 's3://kb-kbnbfc-reports/output/'
            },
            WorkGroup="Business-team"
        )
        query_execution_id = query_response["QueryExecutionId"]
        print(f"Query Execution ID: {query_execution_id}")
        # Wait for the query to finish
        get_object = False
        timeout = time.time() + 60 * 30  # 30 mins timeout
        # Wait for query to complete
        while not get_object:
            try:
                if time.time() > timeout:
                    raise TimeoutError("Query execution exceeded timeout")
                # Check query status
                query_status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                status = query_status["QueryExecution"]["Status"]["State"]
                if status == "SUCCEEDED":
                    print("Query succeeded.")
                    get_object = True
                elif status == "FAILED":
                    raise Exception(f"Query failed: {query_status['QueryExecution']['Status']['StateChangeReason']}")
                elif status == "CANCELLED":
                    raise Exception("Query was cancelled.")
                time.sleep(5)  # Pause for 5 seconds before checking again
            except Exception as e:
                print(f"Error while checking query status: {e}")
                break
        # Fetch the result from S3 once the query is finished
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
            aws_session_token=aws_session_token
        )
        try:
            obj = s3_client.get_object(Bucket='kb-kbnbfc-reports', Key=f'output/{query_execution_id}.csv')
            print(f"Fetched data for query execution: {query_execution_id}")
        except Exception as e:
            print(f"Error fetching object from S3: {e}")
            return pd.DataFrame()         # Return an empty DataFrame in case of error
        # Load the data into a Pandas DataFrame
        my_data = pd.read_csv(io.BytesIO(obj['Body'].read()))
        if my_data.shape[0] > 0:
            print(f"SQL fetch size is - {len(my_data.index)}")
            print(my_data.head())
            print("The head of the data frame is printed here")
        else:
            my_data = pd.DataFrame()
            print("SQL fetch size = 0")
        # Print elapsed time
        end_time = time.time() - start_time 
        print(f"Elapsed time: {end_time / 60} minutes") 
        print("Time has elapsed for the time being") 
        print("Data fetched from athena, waiting for it to get into the wordpress")
        print("")
        print("The end time is ",end_time)
        print("The total elapsed time is ",end_time+1) 
        return my_data
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    

#############################################################################################################################
################################################ INPUTS REQUIRED ############################################################
#############################################################################################################################

# ongoing_month="2025-08"
# last_month="2025-07"
# lm_name = "July-25"
# om_name = "Aug-25"
# om_days=list(range(1,5)) # 1 MORE THAN THE TILL DATE 
# om_actual_days= list(range(1,31))
# lm_days = list(range(1,32)) # 1 MORE THAN THE COUNT OF DAYS IN THE LAST MONTH 

# Get today's date
today = datetime.today()

# Ongoing month = current month
ongoing_month = today.strftime("%Y-%m")

# Last month = first day of this month - 1 day → gives last month's date
first_day_this_month = today.replace(day=1)
last_month_date = first_day_this_month - timedelta(days=1)
last_month = last_month_date.strftime("%Y-%m")

# Friendly names
lm_name = last_month_date.strftime("%b-%y")   # e.g., "Jul-25"
om_name = today.strftime("%b-%y")             # e.g., "Aug-25"

# Day ranges
# om_days: 1 to (today + 1)
om_days = list(range(1, today.day + 1 ))

# om_actual_days: 1 to total days in ongoing month
om_days_in_month = calendar.monthrange(today.year, today.month)[1]
om_actual_days = list(range(1, om_days_in_month ))

# lm_days: 1 to (last month's total days + 1)
lm_days_in_month = calendar.monthrange(last_month_date.year, last_month_date.month)[1]
lm_days = list(range(1, lm_days_in_month + 1 ))

# Output check
print(f"ongoing_month = {ongoing_month}")
print(f"last_month = {last_month}")
print(f"lm_name = {lm_name}")
print(f"om_name = {om_name}")
print(f"om_days = {om_days}")
print(f"om_actual_days = {om_actual_days}")
print(f"lm_days = {lm_days}")

###########################################################################################################################
############################ ----- Query to get the last 3 month's data ( Inc current month ) ----- #######################
###########################################################################################################################

print("FPLP query is getting started !!! ")

#DateField inside the query 
query_FPLP = f"""with base_emi as 
(select userid,
loan_id,
product_name,
loan_number,
date_format(disbursed_date,'%Y-%m-%d') as disb_date,
date_format(disbursed_date,'%Y-%m') as disb_month, 
date_format(emi_due_date,'%Y-%m') as due_month, 
date_format(emi_due_date,'%Y-%m-%d') as due_date,
date_format(closedon,'%Y-%m') as closed_month,
date_format(closedon,'%Y-%m-%d') as closed_date, 
producttenure,
installment_number,
rank() over(partition by loan_id order by installment_id desc) as installment_remaining
from kreditbee_bi_dw_iceberg.yp_emi_data_tbl a 
where date_format(emi_due_date,'%Y-%m') IN ('{ongoing_month}','{last_month}')
and product_name IN ('MLA-15K','MLA-20K','MLA-30K','MLA-40K','MLA-50K')),
state_cte as(select * from (
select be.loan_id, state, changedon, rank() over(partition by be.userid,be.loan_id order by b.id desc) as state_rnk 
from base_emi be  join yp_iceberg.yp_user_state b on be.userid = b.uid 
where date_format(b.changedon,'%Y-%m-%d') <= DATE_FORMAT(DATE_ADD('day', 1, date(closed_date)), '%Y-%m-%d')
) where state_rnk = 1
group by 1,2,3,4),
ug_cte as (
select * from (
select be.loan_id,oldband,oldsubband, newband, newsubband, changedon, rank() over(partition by be.userid,be.loan_id order by b.id desc) as band_rnk 
from base_emi be  join yp_iceberg.yp_log_user_band_change b on be.userid = b.uid 
where date_format(b.changedon,'%Y-%m-%d') <= DATE_FORMAT(DATE_ADD('day', 1, date(closed_date)), '%Y-%m-%d')
and date_format(b.changedon,'%Y-%m-%d') >disb_date
) where band_rnk = 1
group by 1,2,3,4,5,6,7),
next_loan_cte as(select * from (
select l.userid,
be.loan_id,
l.id as loan_id_next,
l.productname as product_next,
date_format(l.disbursedoN,'%Y-%m-%d') as disb_next_date,
date_format(l.disbursedoN,'%Y-%m') as disb_next_month,
date_diff('day',date(be.closed_date),date(date_format(l.disbursedoN,'%Y-%m-%d'))) as disb_gap_closed,
principaldue,
rank() over(partition by be.userid,be.loan_id order by l.id) as lrnk 
from yp_iceberg.yp_loan l join base_emi be on l.userid = be.userid and l.id > be.loan_id
where l.state in (47,71)
) where lrnk = 1
group by 1,2,3,4,5,6,7,8,9)
select due_month,due_date,closed_date,disb_next_date,disb_next_month,closed_month,producttenure,installment_number,product_name,state,newband,product_next,
count(distinct a.loan_id) as cnt_loans from base_emi a left join state_cte sc on sc.loan_id = a.loan_id
left join ug_cte uc on uc.loan_id = a.loan_id
left join next_loan_cte nlc on nlc.loan_id = a.loan_id
group by 1,2,3,4,5,6,7,8,9,10,11,12 ;
"""
df=load_data(query_FPLP)

print(df)
# df contains the last 3 months data of df_active
# df_active : active df ( closed_date is either empty or is the current month )
# df_inactive : Inactice df ( closed date is prev month or before ) 

current_month = pd.Period(ongoing_month, freq='M')

df['closed_month'] = pd.to_datetime(df['closed_month'], errors='coerce').dt.to_period('M')

# Active woh hain jinke closed month empty ho ya isi month close huye ho
df_active = df[(df['closed_month'].isna()) | (df['closed_month'] == current_month)]
df_inactive = df[df['closed_month'] < current_month]
# May mein jo active hain !!
df_active_may = df[(df['closed_month'].isna()) | (df['closed_month'] >= pd.Period(last_month, freq='M'))] #DateField
df_active_may['due_month'] = pd.to_datetime(df_active_may['due_month'], errors='coerce').dt.to_period('M')
df_active_may = df_active_may[df_active_may['due_month'] == pd.Period(last_month, freq='M')] #DateField
# df_active_may.to_csv("df_active_may.csv")

df_active_june = df[(df['closed_month'].isna()) | (df['closed_month'] >= pd.Period(ongoing_month, freq='M'))] #DateField
df_active_june['due_month'] = pd.to_datetime(df_active_june['due_month'], errors='coerce').dt.to_period('M')
df_active_june = df_active_june[df_active_june['due_month'] == pd.Period(ongoing_month, freq='M')] #DateField

# print("InActive closed_months:", df_inactive['closed_month'].unique())
# print("Active closed_months June:", df_active_june['closed_month'].unique())
# print("Active closed_months May:", df_active_may['closed_month'].unique())

########################################################################################################################################
###### Now we are having active and inactive datframes separately to be handled. #######################################################
########################################################################################################################################

st.title("📈 Active Base ( FPLP )")
# ---------- INPUT DF: df_active must be preloaded and contain columns: due_date, closed_date, cnt_loans ----------
# Convert dates
for col in ['due_date', 'closed_date','disb_next_date']:
    df_active[col] = pd.to_datetime(df_active[col], errors='coerce')
    df_active_june[col] = pd.to_datetime(df_active_june[col], errors='coerce')
    df_active_may[col] = pd.to_datetime(df_active_may[col], errors='coerce')

# Convert due_date to period for filtering
df_active['due_month'] = df_active['due_date'].dt.to_period('M')
df_active['closed_day'] = df_active['closed_date'].dt.day

df_active_june['due_month'] = df_active_june['due_date'].dt.to_period('M')
df_active_june['closed_day'] = df_active_june['closed_date'].dt.day
# print(df_active_june)

df_active_may['due_month'] = df_active_may['due_date'].dt.to_period('M')
df_active_may['closed_day'] = df_active_may['closed_date'].dt.day

df_active['due_day'] = df_active['due_date'].dt.day
df_active_may['due_day'] = df_active_may['due_date'].dt.day
df_active_june['due_day'] = df_active_june['due_date'].dt.day

# For may & june stacked bar
df_active_may['month_label'] = 'May'
df_active_june['month_label'] = 'June'
############ CHECK ###########

# Current and previous month
pd.Period(ongoing_month, freq='M')
previous_month = current_month - 1 # dynamic 

######################################### BAND MAP CHANGE ###################################################
import numpy as np

band_map = {
    2275: "MLA-50K",
    2001: "MLA-40K",
    621: "MLA-30K",
    620: "MLA-20K",
    619: "MLA-15K"
}

# --- For June ---
df_active_june['conf_product'] = np.where(
    (df_active_june['state'] == 53) & (df_active_june['newband'].isna()),
    df_active_june['product_name'],
    df_active_june['newband'].map(band_map)
)

# --- For May ---
df_active_may['conf_product'] = np.where(
    (df_active_may['state'] == 53) & (df_active_may['newband'].isna()),
    df_active_may['product_name'],
    df_active_may['newband'].map(band_map)
)

# Filter for current and previous month
df_june = df_active_june[df_active_june['due_month'] == current_month].copy()
df_may = df_active_may[df_active_may['due_month'] == previous_month].copy()

################################ UPGRADE / NON UPGRADE CLASSIFICATION #######################

def is_upgrade(row):
    # Define mapping from product names to band codes (current)
    product_to_band = {
        2275: "MLA-50K",
    2001: "MLA-40K",
    621: "MLA-30K",
    620: "MLA-20K",
    619: "MLA-15K"
    }
    # Assign priority rank to bands
    band_order = {619:1,620:2,621:3,2001:4,2775:5}
    curr_band_id = product_to_band.get(row['product_name'], 0)
    curr_rank = band_order.get(curr_band_id, 0) 
    new_rank = band_order.get(row['newband'], 0) 
    state = row['state']
    # Upgrade if moved to higher-ranked band
    if state == 53 and new_rank > curr_rank:
        return 'Upgrade'
    # Non-Upgrade if no band or same/lower
    elif (state == 53 and pd.isna(row['newband'])) or (state == 53 and new_rank<=curr_rank):
        return 'Non-Upgrade'
    return 'Other'

df_active_june['upgrade_flag'] = df_active_june.apply(is_upgrade, axis=1)
df_active_may['upgrade_flag'] = df_active_may.apply(is_upgrade, axis=1)
# df_may.to_csv("df_may_closure.csv")


################################## SAME/FUTURE MONTH CLASSIFICATION ###########################
def is_sm(row):
    if row['producttenure'] == row['installment_number']:
        return 'SM'
    else:
        return 'FM'
    
df_active_june['sm_flag'] = df_active_june.apply(is_sm, axis=1)
df_active_may['sm_flag'] = df_active_may.apply(is_sm, axis=1)

# df_active_june.to_csv("df_june_for_validation.csv")

############################################## SAME/FUTURE M & UPGRADE / NON- UPGRADE ######################################

def is_sm_ug(row):
    if row['sm_flag'] == "SM" and row['upgrade_flag']== "Upgrade":
        return 'SM_UG'
    elif row['sm_flag'] == "SM" and row['upgrade_flag']== "Non-Upgrade":
        return 'SM_NUG'
    elif row['sm_flag'] == "FM" and row['upgrade_flag']== "Upgrade":
        return 'FM_UG'
    elif row['sm_flag'] == "FM" and row['upgrade_flag']== "Non-Upgrade":
        return 'FM_NUG'

df_active_june['sm_ug_flag'] = df_active_june.apply(is_sm_ug, axis=1)
df_active_may['sm_ug_flag'] = df_active_may.apply(is_sm_ug, axis=1)

##################################################################################################################################
###################################################### BAND & EMI WISE ###########################################################
##################################################################################################################################

# Add a hyperlink that opens `inactive.py` in a new tab
import streamlit as st

def active():
    st.markdown(
        "<a href='?view=inactive' target='_blank'><button>Open Inactive</button></a>",
        unsafe_allow_html=True
    )

def inactive():
    st.title("Inactive View")

view = st.query_params.get("view", ["active"])[0]

if view == "inactive":
    inactive()
else:
    active()

    
import streamlit as st
band_options = ['Overall', 'MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K', 'MLA-60K']
emi_options = ['Overall', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

band_list = []
emi_list = []


import streamlit as st
from streamlit_pills import pills

# Pills (inside sticky header)
selected_band = pills(
    "Select Band",
    ['Overall', 'MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K', 'MLA-60K'],
    key="band_filter"
)

selected_emi = pills(
    "Select EMI",
    ['Overall', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    key="emi_filter"
)


# Filter logic for df_june (current month)
if (selected_band == 'Overall' and selected_emi == 'Overall'):
    df_june = df_active_june[df_active_june['due_month'] == current_month].copy()
    band_list = ['Overall', 'MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K', 'MLA-60K']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['installment_number'] == selected_emi)].copy()
    band_list = ['Overall', 'MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K', 'MLA-60K']
    emi_list = [ selected_emi]

elif (selected_band != 'Overall' and selected_emi == 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['product_name'] == selected_band)].copy()
    band_list = [selected_band]
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

else:
    df_june = df_active_june[
        (df_active_june['due_month'] == current_month) &
        (df_active_june['product_name'] == selected_band) & (df_active_june['installment_number'] == selected_emi)].copy()
    band_list = [selected_band]
    emi_list = [selected_emi]


# (Optional) Filter df_may for previous month if needed MAY DAY MAY DAY 
if (selected_band == 'Overall' and selected_emi == 'Overall'):
    df_may = df_active_may[df_active_may['due_month'] == previous_month].copy()
    band_list = ['Overall', 'MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K', 'MLA-60K']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['installment_number'] == selected_emi)].copy()
    band_list = ['Overall', 'MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K', 'MLA-60K']
    emi_list = [ selected_emi]

elif (selected_band != 'Overall' and selected_emi == 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['product_name'] == selected_band)].copy()  
    band_list = [selected_band]
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

else:
    df_may = df_active_may[
        (df_active_may['due_month'] == previous_month) &
        (df_active_may['product_name'] == selected_band) & (df_active_may['installment_number'] == selected_emi)].copy()
    band_list = [selected_band]
    emi_list = [selected_emi]

# from here df_may and df_june are used !!!!

######################################### CLOSURE KAHANI STARTS HERE #############################

# ----- Daily EMI Due Count (Bar) -----
daily_emi_june = df_june.groupby('due_day')['cnt_loans'].sum().reindex(range(1,32), fill_value=0)
daily_emi_may = df_may.groupby('due_day')['cnt_loans'].sum().reindex(range(1,32), fill_value=0)
# ----- Daily Closure for Current Month -----
closed_curr = df_june[df_june['closed_date'].dt.to_period('M') == current_month].groupby('closed_day')['cnt_loans'].sum().reindex(range(1,32), fill_value=0).cumsum()
# ----- Daily Closure for Previous Month -----
closed_prev = df_may[df_may['closed_date'].dt.to_period('M') == previous_month].groupby('closed_day')['cnt_loans'].sum().reindex(range(1,32), fill_value=0).cumsum()
# -------------------------- PLOT ---------------------
fig = go.Figure()
# Key annotation days
annot_days = [ 5, 10, 15, 20, 25]
# Daily EMI Due – May
fig.add_trace(go.Scatter(
    x=lm_days,
    y=daily_emi_may,
    name=f'{lm_name} LineUP EMI (Cumulative)',
    mode='lines+markers', line=dict(color='red', dash='dash'),
    fill='tozeroy',
    text=[f"{v:.0f}" if day in annot_days else "" for day, v in zip(range(1,32), daily_emi_may)],
    textposition="top center"
))
# Daily EMI Due – June
fig.add_trace(go.Scatter(
    x=om_days,
    y=daily_emi_june,
    name=f'{om_name} LineUP EMI Due',
    mode='lines+markers+text',
    line=dict(color='lightblue'),
    fill='tozeroy',
    opacity=0.5,
    text=[f"{v:.0f}" if day in annot_days else "" for day, v in zip(range(1,32), daily_emi_june)],
    textposition="top center"
))
# Closures – Current Month
fig.add_trace(go.Scatter(
    x=om_days,
    y=closed_curr,
    name=f'Closures (Cumulative )  – {current_month.strftime("%b-%Y")}',
    mode='lines+markers+text',
    line=dict(color='green', width=2),
    text=[f"{v:.0f}" if day in annot_days else "" for day, v in zip(range(1,32), closed_curr)],
    textposition="bottom right"
))
# Closures – Previous Month
fig.add_trace(go.Scatter(
    x=lm_days,
    y=closed_prev,
    name=f'Closures ( Cumulative ) – {previous_month.strftime("%b-%Y")}',
    mode='lines+markers+text',
    line=dict(color='orange', dash='dash'),
    text=[f"{v:.0f}" if day in annot_days else "" for day, v in zip(range(1,32), closed_prev)],
    textposition="bottom right"
))
fig.update_layout(
    title="📊 Lined up EMI Due vs Closure MTD Trend",
    xaxis_title="Closure Date",
    yaxis_title="EMI Count",
    barmode='overlay',
    plot_bgcolor='white',
    height=500
)
st.plotly_chart(fig, use_container_width=True)

###################################################################################################################################
######################################### BOTH OF THE MONTH'S ACTIVE BASE #########################################################
###################################################################################################################################

# ---------- Combine Both Months`````````````````` ----------
df_combined = pd.concat([
    df_active_may[['due_day', 'installment_number', 'cnt_loans', 'month_label']],
    df_active_june[['due_day', 'installment_number', 'cnt_loans', 'month_label']]
])
# ---------- Group by day, EMI number, and month ----------
grouped = df_combined.groupby(['due_day', 'month_label', 'installment_number'])['cnt_loans'].sum().reset_index()
# ---------- Plot ----------
fig = px.bar(
    grouped,
    x='due_day',
    y='cnt_loans',
    color='installment_number',
    facet_col='month_label',
    labels={'due_day': 'Day of Month', 'cnt_loans': 'EMI Count'},
    title=f"📊 EMI Stack by Installment Number – May vs {om_name}",
    height=600
)
fig.update_layout(
    barmode='stack',
    plot_bgcolor='white',
    xaxis=dict(tickmode='linear', tick0=1, dtick=1)
)
with st.expander("📦 View EMI Stack Chart"):
    st.plotly_chart(fig, use_container_width=True)

############################################################################################################################
######################################### MTD TRIALS #######################################################################
############################################################################################################################

import calendar 
# Days in month
days = list(range(1,32))  #dynamic
# Define month references
current_month = pd.to_datetime(f"{ongoing_month}-01") #DateField
previous_month = pd.to_datetime(f"{last_month}-01") #DateField

# ------------------------------------------ MTD Closure % ---------------------------------------------
may_closure_mtd = []
june_closure_mtd = []
june_predicted_mtd = []
june_predicted_closure_count=[]
predicted_closure_matrix_daywise = {}

closure_ratio_df = pd.DataFrame({
    1: [5.31, 4.09, 3.11, 3.50, 3.79],
    2: [6.32, 4.29, 3.45, 4.01, 4.51],
    3: [7.50, 5.53, 5.13, 5.61, 5.95],
    4: [10.13, 8.25, 7.64, 7.49, 8.27],
    5: [17.99, 14.40, 12.23, 11.43, 12.12],
    6: [35.31, 22.72, 18.14, 15.26, 16.41],
    7: [83.42, 38.50, 27.93, 20.31, 21.13],
    8: [88.73, 79.39, 52.86, 29.28, 22.31],
    9: [50.00, 62.75, 83.97, 53.53, 38.21],
    10: [0.00, 100.00, 45.00, 82.57, 72.77],
    11: [0.00, 0.00, 75.00, 50.00, 100.00],
}, index=['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K'])

# Convert percentages to decimals
ratio_df = closure_ratio_df / 100

for d in lm_days:
    current_day = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    may_active = df_may[df_may['due_date'] <= current_day]['cnt_loans'].sum()
    print("may active count : ", may_active)
    may_closed = df_may[(df_may['closed_date'] <= current_day) & (df_may['closed_month']==pd.Period(last_month, freq='M'))]['cnt_loans'].sum()      #DateField
    print("may closed count : ", may_closed)
    may_ratio = (may_closed / may_active) * 100 if may_active else 0
    may_closure_mtd.append(may_ratio)
    print("ratio :", may_ratio)

for d in om_days:
    current_day_june = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    # FIX: Compare day to day
    june_active = df_june[df_june['due_day'] <= d]['cnt_loans'].sum() 
    june_closed = df_june[(df_june['closed_day'] <= d) & (df_june['closed_month']==pd.Period(ongoing_month, freq='M'))]['cnt_loans'].sum()     #DateField
    june_ratio = (june_closed / june_active) * 100 if june_active else 0    
    june_closure_mtd.append(june_ratio)
    print("june active count : ", june_active)
    print("june closed count : ", june_closed)
    print("ratio :", june_ratio)
    
for d in om_actual_days:
    ################################## PREDICTION ########################################
    # 🧠 PREDICTED MTD %
    df_mtd = df_june[df_june['due_day'] <= d]  # MTD snapshot
    # Build EMI x Product (installment_number x product) matrix
    lineup_matrix = df_mtd.pivot_table(index='product_name', columns='installment_number',
                                       values='cnt_loans', aggfunc='sum').fillna(0)
    # Align with ratio_df shape and order
    # Initialize predicted values
    predicted_mtd_count = 0
    total_lineup = 0

    lineup_matrix = lineup_matrix.reindex(ratio_df.index).fillna(0)
    print(" ********************************* ",lineup_matrix)
    lineup_matrix = lineup_matrix[[col for col in ratio_df.columns if col in lineup_matrix.columns]]

    if selected_band == 'Overall' and selected_emi == 'Overall':
        predicted_matrix = lineup_matrix * ratio_df
        predicted_mtd_count = predicted_matrix.values.sum()
        print("predicted mt countttttttttttttttttttt MFFFFF",predicted_mtd_count)
        total_lineup = lineup_matrix.values.sum()
        predicted_closure_matrix_daywise[d] = predicted_matrix.copy()


    elif selected_band == 'Overall' and selected_emi != 'Overall':
        if selected_emi not in lineup_matrix.columns:
            predicted_ratio = 0
            june_predicted_mtd.append(predicted_ratio)
            continue
            
        predicted_matrix = lineup_matrix[[selected_emi]] * ratio_df[[selected_emi]]
        predicted_mtd_count = predicted_matrix.values.sum()
        total_lineup = lineup_matrix[[selected_emi]].values.sum()
        predicted_closure_matrix_daywise[d] = predicted_matrix.copy()

    
    elif selected_band != 'Overall' and selected_emi == 'Overall':
        if selected_band not in lineup_matrix.index:
            predicted_ratio = 0
            june_predicted_mtd.append(predicted_ratio)
            continue
        predicted_matrix = lineup_matrix.loc[[selected_band]] * ratio_df.loc[[selected_band]]
        predicted_mtd_count = predicted_matrix.values.sum()
        total_lineup = lineup_matrix.loc[[selected_band]].values.sum()
        predicted_closure_matrix_daywise[d] = predicted_matrix.copy()

    
    else:
        if selected_band not in lineup_matrix.index or selected_emi not in lineup_matrix.columns:
            predicted_ratio = 0
            june_predicted_mtd.append(predicted_ratio)
            continue
            
        val = lineup_matrix.at[selected_band, selected_emi] * ratio_df.at[selected_band, selected_emi]
        predicted_mtd_count = val
        total_lineup = lineup_matrix.at[selected_band, selected_emi]
        predicted_matrix = pd.DataFrame( [[val]], index=[selected_band], columns=[selected_emi])
        predicted_closure_matrix_daywise[d] = predicted_matrix.copy()


    # predicted_mtd_count = (lineup_matrix * ratio_df).values.sum()
    # total_lineup = lineup_matrix.values.sum()
    june_predicted_closure_count.append(predicted_mtd_count)
    predicted_ratio = (predicted_mtd_count / total_lineup) * 100 if total_lineup else 0
    june_predicted_mtd.append(predicted_ratio)
    print(f"predicted ratio for day {d} : {predicted_ratio}")
    #######################################################################################


#######################################################################################################################################
############################################################## MTD PLOT #############################################################
#######################################################################################################################################

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_dual_axis = make_subplots(specs=[[{"secondary_y": True}]])

# --- Cumulative EMI Line-Up ---
fig_dual_axis.add_trace(go.Scatter(
    x=lm_days, y=daily_emi_may,  # 🔴 May - Cumulative Line
    name=f'{lm_name} EMI Line up',
    mode='lines+markers', 
    fill='tozeroy',
    line=dict(color='red', dash='dash')), secondary_y=False)
fig_dual_axis.add_trace(go.Scatter(
    x=om_days, y=daily_emi_june,  # 🔵 June - Cumulative Area
    name=f'{om_name} EMI Line up ',
    mode='lines',
    line=dict(color='lightblue'),
    fill='tozeroy',
    opacity=0.4
), secondary_y=False)
# --- MTD Closure % Curves ---
fig_dual_axis.add_trace(go.Scatter(
    x=lm_days, y=may_closure_mtd,  # 🟠 May - Closure %
    name=f"{lm_name} MTD Closure %",
    mode='lines+markers',
    line=dict(color='orange', dash='dash')
), secondary_y=True)

fig_dual_axis.add_trace(go.Scatter(
    x=om_days, y=june_closure_mtd,  # 🟢 June - Closure %
    name=f"{om_name} MTD Closure %",
    mode='lines+markers',
    line=dict(color='green')
), secondary_y=True)

# Predicted curve 
fig_dual_axis.add_trace(go.Scatter(
    x=om_actual_days,  # e.g., [1, 2, 3, ..., 31]
    y=june_predicted_mtd,  # 🔮 Predicted MTD % (already computed above)
    name='Predicted MTD Closure %',
    mode='lines+markers',
    line=dict(color='#9467bd')  # 
), secondary_y=True)

# --- Layout Settings ---
fig_dual_axis.update_layout(
    title="📊 Daily EMI Line-Up vs MTD Closure %",
    xaxis_title="Day of Month",
    plot_bgcolor='white',
    height=520,
    legend_title="Metrics"
)
# --- Y-Axis Customization ---
fig_dual_axis.update_yaxes(
    title_text="Daily EMI Due Count",
    secondary_y=False,
    showgrid=False,
    tickformat="~s",
    rangemode="tozero"
)
fig_dual_axis.update_yaxes(
    title_text="MTD Closure %",
    secondary_y=True,
    range=[0, 30],
    tickformat=".1f"
)
# --- Display in Streamlit ---
st.plotly_chart(fig_dual_axis, use_container_width=True)

##################################################################################################################################
###################################################### MTD CONFIRMATION ##########################################################
##################################################################################################################################


from plotly.subplots import make_subplots
import plotly.graph_objects as go

predicted_confirmation_matrix_daywise = {}
predicted_confirmation_daywise=[]

confirmation_ratio_df = pd.DataFrame({
    1: [94.29, 95.61, 96.95, 96.54, 97.39],
    2: [88.35, 90.24, 90.51, 92.31, 91.40],
    3: [84.83, 86.18, 88.86, 87.05, 88.21],
    4: [81.71, 82.32, 83.19, 81.19, 84.36],
    5: [78.26, 79.95, 79.33, 79.80, 81.65],
    6: [76.13, 76.40, 78.39, 77.93, 81.07],
    7: [73.52, 75.93, 75.64, 75.21, 76.78],
    8: [86.61, 72.70, 71.58, 72.78, 77.44],
    9: [50.00, 85.00, 80.16, 74.11, 76.52],
    10: [0.00, 100.00, 44.44, 80.83, 77.65],
    11: [0.00, 0.00, 100.00, 50.00, 100.00],
}, index=['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K'])

confirmation_ratio_df = confirmation_ratio_df.fillna(0)
confirmation_ratio_df = confirmation_ratio_df.astype(float) / 100
pred_confirmation_count=[]

def get_confirmation_data(days, df, month_str):
    daily_closures = []
    daily_confirm_pct = []
    predicted_confirm_pct = []

    for d in days:
        cutoff = pd.Timestamp(f"{month_str}-{d:02d}")
        closed = df[(df['closed_date'] <= cutoff)]
        total_closures = closed['cnt_loans'].sum()
        confirmed = closed[closed['state'] == 53]['cnt_loans'].sum()
        daily_closures.append(total_closures)
        daily_confirm_pct.append((confirmed / total_closures * 100) if total_closures else 0)

    for d in om_actual_days:
    # Get closure predicted matrix from previous day
        if (d - 1) not in predicted_closure_matrix_daywise:
            # Handle missing day (e.g. first day), fallback to zero matrix or lineup_matrix
            # For safety, maybe skip or initialize with zeros
            print(f"Warning: Closure matrix for day {d-1} missing!")
            continue
        
        lineup_matrix = predicted_closure_matrix_daywise[d]

        # Ensure ratio_aligned is aligned to lineup_matrix
        ratio_aligned = confirmation_ratio_df.reindex(index=lineup_matrix.index).fillna(0)
        ratio_aligned = ratio_aligned[[col for col in lineup_matrix.columns if col in ratio_aligned.columns]]

        # Initialize values
        predicted_mtd_count = 0
        total_lineup = 0

        # Apply EMI × Band Logic:
        if selected_band == 'Overall' and selected_emi == 'Overall':
            predicted_matrix = lineup_matrix * ratio_aligned
            predicted_mtd_count = predicted_matrix.values.sum()
            total_lineup = lineup_matrix.values.sum()
            predicted_confirmation_matrix_daywise[d] = predicted_matrix.copy()

        elif selected_band == 'Overall' and selected_emi != 'Overall':
            if selected_emi not in lineup_matrix.columns:
                predicted_confirm_pct.append(0)
                continue
            predicted_matrix = lineup_matrix[[selected_emi]] * ratio_aligned[[selected_emi]]
            predicted_mtd_count = predicted_matrix.values.sum()
            total_lineup = lineup_matrix[[selected_emi]].values.sum()
            predicted_confirmation_matrix_daywise[d] = predicted_matrix.copy()

        elif selected_band != 'Overall' and selected_emi == 'Overall':
            if selected_band not in lineup_matrix.index:
                predicted_confirm_pct.append(0)
                continue
            predicted_matrix = lineup_matrix.loc[[selected_band]] * ratio_aligned.loc[[selected_band]]
            predicted_mtd_count = predicted_matrix.values.sum()
            total_lineup = lineup_matrix.loc[[selected_band]].values.sum()
            predicted_confirmation_matrix_daywise[d] = predicted_matrix.copy()

        else:
            if selected_band not in lineup_matrix.index or selected_emi not in lineup_matrix.columns:
                predicted_confirm_pct.append(0)
                continue
            val = lineup_matrix.at[selected_band, selected_emi] * ratio_aligned.at[selected_band, selected_emi]
            predicted_mtd_count = val
            total_lineup = lineup_matrix.at[selected_band, selected_emi]
            predicted_matrix = pd.DataFrame([[val]], index=[selected_band], columns=[selected_emi])
            predicted_confirmation_matrix_daywise[d] = predicted_matrix.copy()

        # Now compute predicted ratio wrt closure counts (or total_lineup)
        predicted_ratio = (predicted_mtd_count / total_lineup) * 100 if total_lineup else 0
        predicted_confirm_pct.append(predicted_ratio)
        predicted_confirmation_daywise.append(predicted_mtd_count)

    return daily_closures, daily_confirm_pct, predicted_confirm_pct


# Get data
may_closures, may_confirm_pct,may_pred_conf = get_confirmation_data(lm_days,df_may, last_month) #DateField
june_closures, june_confirm_pct, predicted_confirm_pct = get_confirmation_data(om_days,df_june, ongoing_month) #DateField
days_may = list(range(1, 32))
days_june = list(range(1, 31))

# --- Plot ---
fig_conf = make_subplots(specs=[[{"secondary_y": True}]])

# May Area + Confirm %
fig_conf.add_trace(go.Scatter(
    x=lm_days, y=may_closures,
    name=f"{lm_name} Closures",
    mode='lines+markers', 
    fill='tozeroy',
    line=dict(color='red', dash='dash'),
    text=[str(int(x)) if day in [1, 5, 10, 15, 20, 25, 30] else "" for day, x in zip(days_may, may_closures)],
    textposition='top center'
), secondary_y=False)

fig_conf.add_trace(go.Scatter(
    x=lm_days, y=may_confirm_pct,
    name=f"{lm_name} Confirmation %",
    mode='lines+markers+text',
    line=dict(color='orange', dash='dash'),
    text=[f"{x:.1f}%" if day in [1, 5, 10, 15, 20, 25, 30] else "" for day, x in zip(days_may, may_confirm_pct)],
    textposition='top center'
), secondary_y=True)

# June Area + Confirm %
fig_conf.add_trace(go.Scatter(
    x=om_days, y=june_closures,
    name=f"{om_name} Closures",
    mode='lines+markers',
    fill='tozeroy',
    line=dict(color='lightblue'),
), secondary_y=False)

fig_conf.add_trace(go.Scatter(
    x=om_days, y=june_confirm_pct,
    name=f"{om_name} Confirmation %",
    mode='lines+markers',
    line=dict(color='green'),
), secondary_y=True)

fig_conf.add_trace(go.Scatter(
    x=om_actual_days,
    y=predicted_confirm_pct,
    name=f"{om_name} Predicted Confirmation %",
    mode='lines+markers',
    line=dict(color='#9467bd', dash='dash'),
    fill=None
), secondary_y=True)

fig_conf.update_layout(
    title=f"📋 Closure Count & Confirmation % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month",
    plot_bgcolor='white',
    height=500,
    legend_title="Metric"
)

fig_conf.update_yaxes(title_text="Closure Count", secondary_y=False, rangemode="tozero")
fig_conf.update_yaxes(title_text="Confirmation %", secondary_y=True, range=[0, 100])

# Show in Streamlit
st.plotly_chart(fig_conf, use_container_width=True)


#####################################################################################################################################
########################################################## KAHANI - PART 2 ##########################################################
#####################################################################################################################################

#######################################################################################################################################
########################################### MOVEMENT ##################################################################################
#######################################################################################################################################

############################################## TARGETABLE BASE - OUTFLOW ##################################

# --- Movement Classification --- 
FPLP_band_list = ["MLA-15K","MLA-20K","MLA-30K","MLA-40K", "MLA-50"]

def classify(row):
    conf_product = row['conf_product']
    state = row['state']
    newband=row['newband']
    if state == 34:
        return 'Reject'
    elif ((pd.isna(conf_product) and pd.isna(newband) and state == 53) or (conf_product in FPLP_band_list and state in [53])):
        return 'In FPLP'
    else:
        return 'Moved Out'

# Prepare closure data for movement classification
print("june closure before",df_active_june)
df_may = df_may[df_may['closed_date'].dt.to_period("M") == pd.Period(previous_month, freq='M')]
# df_june_closure = df_active_june[df_active_june['closed_date'].dt.to_period("M") == current_month].copy()
df_june = df_june[df_june['closed_date'].dt.to_period("M") == pd.Period(current_month, freq='M')]


print("June movement hai yeh new :::: ",df_june)

for df in [df_may, df_june]:
    df['day'] = df['closed_date'].dt.day
    df['movement'] = df.apply(classify, axis=1)

# df_june.to_csv("df_closed_june_testing_10_07.csv")

# --- Movement MTD % ---
def movement_mtd(days,df, movement):
    curve = []
    for d in days:
        till_day = df[df['day'] <= d]
        total = till_day[till_day['state'] == 53]['cnt_loans'].sum() # conf count 
        move = till_day[(till_day['movement'] == movement) & (till_day['state'] == 53)]['cnt_loans'].sum()
        perc = (move / total) * 100 if total else 0
        curve.append(perc)
    return curve

# --- Movement Cumulative Count ---
def movement_cumulative(days,df, movement):
    curve = []
    for d in days:
        till_day = df[df['day'] <= d]
        count = till_day[(till_day['movement'] == movement) & (till_day['state'] == 53)]['cnt_loans'].sum()
        curve.append(count)
    return curve

# % Curves
may_FPLP = movement_mtd(lm_days,df_may, 'In FPLP')
may_moved = movement_mtd(lm_days,df_may, 'Moved Out')
june_FPLP = movement_mtd(om_days,df_june, 'In FPLP')
june_moved = movement_mtd(om_days,df_june, 'Moved Out')
# Absolute Count Curves
may_FPLP_abs = movement_cumulative(lm_days,df_may, 'In FPLP')
may_moved_abs = movement_cumulative(lm_days,df_may, 'Moved Out')
june_FPLP_abs = movement_cumulative(om_days,df_june, 'In FPLP')
june_FPLP_abs = movement_cumulative(om_days,df_june, 'In FPLP')
june_FPLP_abs = movement_cumulative(om_days,df_june, 'In FPLP')
june_moved_abs = movement_cumulative(om_days,df_june, 'Moved Out')

############# TARGETABLE BASE ANALYSIS - INFLOW #######################
# TB mein kahan kahan se janta aayi !!! 
# Case 1 : Band Upgrdae ( Uograde ) - state =53 and band upg
# Case 2 : Band Downgrade ( Non-Upgrade ) - state = 53 and band down 


current_month = pd.Period(ongoing_month, freq='M')
# Current and previous month
pd.Period(ongoing_month, freq='M')
previous_month = current_month - 1 # dynamic 


# print("printing df_june after confirmation ",df_june)

#################################### Prediction ###########################################


may_conf_cumsum = []
june_conf_cumsum = []

predicted_inFPLP_pct = []
predicted_inFPLP_daywise=[]
predicted_inFPLP_matrix_daywise = {}

predicted_move_out_pct = []
predicted_outFPLP_daywise=[]
predicted_outFPLP_matrix_daywise = {}

##################################### In FPLP & Out FPLP prediction ##################################

outFPLP_data = {
    1:  [9.65,  8.37,  6.49,  9.92,  6.97],
    2:  [14.21, 10.42, 10.92, 8.48,  8.96],
    3:  [14.70, 11.95, 12.39, 13.78, 13.83],
    4:  [19.82, 16.39, 13.98, 15.58, 14.70],
    5:  [23.04, 18.00, 15.92, 16.14, 13.93],
    6:  [24.55, 18.93, 17.08, 16.18, 15.88],
    7:  [25.41, 19.30, 18.19, 16.53, 17.07],
    8:  [11.90, 20.76, 18.26, 19.67, 18.05],
    9:  [50.00, 0.00, 18.60, 20.16, 19.74],
    10: [100.00, 0.00, 50.00, 21.54, 19.01],
    11: [100.00, 100.00, 25.00, 50.00, 16.67]}

# Create DataFrame from raw data
inFPLP_ratio_df = pd.DataFrame(outFPLP_data,index=['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K'])
inFPLP_ratio_df = 100 - inFPLP_ratio_df
inFPLP_ratio_df = inFPLP_ratio_df / 100
inFPLP_ratio_df = inFPLP_ratio_df.astype(float).fillna(0)

outFPLP_ratio_df = pd.DataFrame(outFPLP_data,index=['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K'])
outFPLP_ratio_df = outFPLP_ratio_df / 100
outFPLP_ratio_df = outFPLP_ratio_df.astype(float).fillna(0)

for d in lm_days:
    # --- May ---
    current_day_may = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    may_closed = df_may[(df_may['closed_date'] <= current_day_may) & (df_may['state'] == 53)]['cnt_loans'].sum()
    may_conf_cumsum.append(may_closed)


for d in om_days:
    # --- June ---
    current_day_june = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    june_closed = df_june[(df_june['closed_date'] <= current_day_june) & (df_june['state'] == 53)]['cnt_loans'].sum()
    june_conf_cumsum.append(june_closed)


for d in om_actual_days:
    if (d - 1) not in predicted_confirmation_matrix_daywise:
            # Handle missing day (e.g. first day), fallback to zero matrix or lineup_matrix
            # For safety, maybe skip or initialize with zeros
            print(f"Warning: Closure matrix for day {d-1} missing!")
            continue
    lineup_matrix = predicted_confirmation_matrix_daywise[d]
    lineup_matrix_out = predicted_confirmation_matrix_daywise[d]

    # Ensure ratio_aligned is aligned to lineup_matrix
    ratio_aligned = inFPLP_ratio_df.reindex(index=lineup_matrix.index).fillna(0)
    ratio_aligned = ratio_aligned[[col for col in lineup_matrix.columns if col in ratio_aligned.columns]]

    ratio_aligned_out = outFPLP_ratio_df.reindex(index=lineup_matrix_out.index).fillna(0)
    ratio_aligned_out = ratio_aligned_out[[col for col in lineup_matrix_out.columns if col in ratio_aligned.columns]]

    predicted_mtd_count = 0
    predicted_mtd_count_out =0 
    total_lineup = 0
    total_lineup_out = 0 

    # Apply EMI × Band Logic:
    if selected_band == 'Overall' and selected_emi == 'Overall':
        predicted_matrix = lineup_matrix * ratio_aligned
        predicted_mtd_count = predicted_matrix.values.sum()

        predicted_matrix_out = lineup_matrix_out * ratio_aligned_out
        predicted_mtd_count_out = predicted_matrix_out.values.sum()

        total_lineup = lineup_matrix.values.sum()
        total_lineup_out = lineup_matrix_out.values.sum()

        predicted_inFPLP_matrix_daywise[d] = predicted_matrix.copy()
        predicted_outFPLP_matrix_daywise[d] = predicted_matrix_out.copy()

    elif selected_band == 'Overall' and selected_emi != 'Overall':
        if selected_emi not in lineup_matrix.columns:
            predicted_inFPLP_pct.append(0)
            continue
        if selected_emi not in lineup_matrix_out.columns:
            predicted_move_out_pct.append(0)
            continue

        predicted_matrix = lineup_matrix[[selected_emi]] * ratio_aligned[[selected_emi]]
        predicted_mtd_count = predicted_matrix.values.sum()
        total_lineup = lineup_matrix[[selected_emi]].values.sum()
        predicted_inFPLP_matrix_daywise[d] = predicted_matrix.copy()

        predicted_matrix_out = lineup_matrix_out[[selected_emi]] * ratio_aligned_out[[selected_emi]]
        predicted_mtd_count_out = predicted_matrix_out.values.sum()
        total_lineup_out = lineup_matrix_out[[selected_emi]].values.sum()
        predicted_outFPLP_matrix_daywise[d] = predicted_matrix_out.copy()

    elif selected_band != 'Overall' and selected_emi == 'Overall':
        if selected_band not in lineup_matrix.index:
            predicted_inFPLP_pct.append(0)
            continue

        if selected_band not in lineup_matrix_out.index:
            predicted_move_out_pct.append(0)
            continue

        predicted_matrix = lineup_matrix.loc[[selected_band]] * ratio_aligned.loc[[selected_band]]
        predicted_mtd_count = predicted_matrix.values.sum()
        total_lineup = lineup_matrix.loc[[selected_band]].values.sum()
        predicted_inFPLP_matrix_daywise[d] = predicted_matrix.copy()

        predicted_matrix_out = lineup_matrix_out.loc[[selected_band]] * ratio_aligned_out.loc[[selected_band]]
        predicted_mtd_count_out = predicted_matrix_out.values.sum()
        total_lineup_out = lineup_matrix_out.loc[[selected_band]].values.sum()
        predicted_outFPLP_matrix_daywise[d] = predicted_matrix_out.copy()

    else:
        if selected_band not in lineup_matrix.index or selected_emi not in lineup_matrix.columns:
            predicted_inFPLP_pct.append(0)
            continue
        if selected_band not in lineup_matrix_out.index or selected_emi not in lineup_matrix_out.columns:
            predicted_move_out_pct.append(0)
            continue

        val = lineup_matrix.at[selected_band, selected_emi] * ratio_aligned.at[selected_band, selected_emi]
        predicted_mtd_count = val
        total_lineup = lineup_matrix.at[selected_band, selected_emi]
        predicted_matrix = pd.DataFrame([[val]], index=[selected_band], columns=[selected_emi])
        predicted_inFPLP_matrix_daywise[d] = predicted_matrix.copy()

        val_out = lineup_matrix_out.at[selected_band, selected_emi] * ratio_aligned_out.at[selected_band, selected_emi]
        predicted_mtd_count_out = val_out
        total_lineup_out = lineup_matrix_out.at[selected_band, selected_emi]
        predicted_matrix_out = pd.DataFrame([[val]], index=[selected_band], columns=[selected_emi])
        predicted_outFPLP_matrix_daywise[d] = predicted_matrix_out.copy()

    # Now compute predicted ratio wrt closure counts (or total_lineup)
    predicted_ratio = (predicted_mtd_count / total_lineup) * 100 if total_lineup else 0
    predicted_inFPLP_pct.append(predicted_ratio)
    predicted_inFPLP_daywise.append(predicted_mtd_count)

    # Now compute predicted ratio wrt closure counts (or total_lineup)
    predicted_ratio_out = (predicted_mtd_count_out / total_lineup_out) * 100 if total_lineup else 0
    predicted_move_out_pct.append(predicted_ratio_out)
    predicted_outFPLP_daywise.append(predicted_mtd_count_out)

fig_combined_abs = make_subplots(specs=[[{"secondary_y": True}]])
# FPLP %
fig_combined_abs.add_trace(go.Scatter(
    x=lm_days, y=may_FPLP,
    name=f"{lm_name} In FPLP %",
    mode='lines+markers',
    line=dict(color="#F1993A", dash='dash')),
    secondary_y=False)
fig_combined_abs.add_trace(go.Scatter(
    x=om_days, y=june_FPLP,
    name=f"{om_name} In FPLP %",
    mode='lines+markers',
    line=dict(color='#388E3C')),
    secondary_y=False)

# Predicted Curves 
fig_combined_abs.add_trace(go.Scatter(
    x=om_actual_days, 
    y=predicted_inFPLP_pct,
    name=f"{om_name} Predicted In FPLP %",
    mode='lines+markers',
    line=dict(color='purple')
), secondary_y=False)

fig_combined_abs.add_trace(go.Scatter(
    x=om_actual_days, 
    y=predicted_move_out_pct,
    name=f"{om_name} Predicted Move Out % ",
    mode='lines+markers',
    line=dict(color='purple', dash='dash')
), 
secondary_y=False)

# Closure Counts (Secondary Y-axis)
fig_combined_abs.add_trace(go.Scatter(
    x=lm_days, y=may_conf_cumsum,
    name=f"{lm_name} Confirmed (Abs)",
    mode='lines+markers', 
    fill='tozeroy',
    line=dict(color='red', dash='dash')),
    secondary_y=True)
fig_combined_abs.add_trace(go.Scatter(
    x=om_days, y=june_conf_cumsum,
    name=f"{om_name} Confirmed (Abs)",
    mode='lines',
    fill='tozeroy',
    line=dict(color='lightblue'),
    opacity=0.35),
    secondary_y=True)
# -- Moved Out % (Primary Y-Axis)
fig_combined_abs.add_trace(go.Scatter(
    x=lm_days, y=may_moved,
    name=f"{lm_name} Moved Out %",
    mode='lines+markers',
    line=dict(color="#FFA000", dash='dot')),
    secondary_y=False)
fig_combined_abs.add_trace(go.Scatter(
    x=om_days, y=june_moved,
    name=f"{om_name} Moved Out %",
    mode='lines+markers',
    line=dict(color="#0E9302")),
    secondary_y=False)
# Layout
fig_combined_abs.update_layout(
    title=f"In FPLP % ( Targettable Base + MoveOut ) – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month",
    legend_title="Metric",
    plot_bgcolor='white',
    height=600,  # Slightly taller for clarity
)
# Y-Axis Customization
fig_combined_abs.update_yaxes(
    title_text="FPLP %", 
    range=[0, 80], 
    secondary_y=False, 
    tickformat=".1f"
)
max_val = max(max(may_conf_cumsum), max(june_conf_cumsum))
step = round(max_val / 5, -3)  # 10 steps, rounded to nearest 1000

fig_combined_abs.update_yaxes(
    title_text="Confirmation Count", 
    secondary_y=True, 
    rangemode="tozero", 
    showgrid=False, 
    tickformat="~s",
    side="right",
    dtick=20000 # or another appropriate value like 2500/10000 based on your data scale
)
st.plotly_chart(fig_combined_abs, use_container_width=True)

######################################################################################################################################
################################################### UPGRADES N NUG PERCENTAGES #######################################################
######################################################################################################################################

# --- Constants ---
days = list(range(1, 32))
# --- Initialize lists ---
may_FPLP_cumsum, june_FPLP_cumsum = [], []
may_upgrade_pct, june_upgrade_pct = [], []
may_nug_pct, june_nug_pct = [], []
predicted_ug_pct,predicted_nug_pct = [],[]

# yahan woh data hai jo in FPLP hain ( Targetable base )
# --- May Loop ---
for d in lm_days:
    cutoff = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    # newband
    may_FPLP_df = df_may[
        (df_may['closed_date'] <= cutoff) & ((df_may['newband'].isin([619,620,621,2001,2775])) | (df_may['newband'].isna())) & (df_may['state']==53)]
    may_FPLP_count=may_FPLP_df['cnt_loans'].sum()
    
    ug = may_FPLP_df[may_FPLP_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum() # UPGRADE COUNT 
    nug = may_FPLP_df[may_FPLP_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum() # NON-UPGRADE COUNT 

    may_FPLP_cumsum.append(may_FPLP_count)
    may_upgrade_pct.append((ug / may_FPLP_count * 100) if may_FPLP_count else 0)
    may_nug_pct.append((nug / may_FPLP_count * 100) if may_FPLP_count else 0)

# --- June Loop ---
for d in om_days:
    cutoff = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    june_FPLP_df = df_june[
        (df_june['closed_date'] <= cutoff) & ((df_june['newband'].isin([619,620,621,2001,2775])) | (df_june['newband'].isna())) & (df_june['state']==53)]
    june_FPLP_count=june_FPLP_df['cnt_loans'].sum()
    
    ug = june_FPLP_df[june_FPLP_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum()
    nug = june_FPLP_df[june_FPLP_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum()

    june_FPLP_cumsum.append(june_FPLP_count)
    june_upgrade_pct.append((ug / june_FPLP_count * 100) if june_FPLP_count else 0)
    june_nug_pct.append((nug / june_FPLP_count * 100) if june_FPLP_count else 0)

###########################################################################################################################################
########################################### AB YAHAN SE HUM LANDING BAND KI BAAT KRENGE ###################################################
###########################################################################################################################################

current_month = pd.Period(ongoing_month, freq='M')
# Current and previous month
pd.Period(ongoing_month, freq='M')
previous_month = current_month - 1 # dynamic 

# Filter logic for df_june (current month)
if (selected_band == 'Overall' and selected_emi == 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['state'] == 53)].copy()
    band_list = ['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['installment_number'] == selected_emi) & (df_active_june['state'] == 53)].copy()
    band_list = ['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K']
    emi_list = [ selected_emi]

elif (selected_band != 'Overall' and selected_emi == 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['conf_product'] == selected_band) & (df_active_june['state'] == 53)].copy()
    band_list = [selected_band]
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

else:
    df_june = df_active_june[
        (df_active_june['due_month'] == current_month) &
        (df_active_june['conf_product'] == selected_band) & (df_active_june['installment_number'] == selected_emi) & (df_active_june['state'] == 53)].copy()
    band_list = [selected_band]
    emi_list = [selected_emi]


# Filter df_may for previous month if needed MAY DAY MAY DAY 
if (selected_band == 'Overall' and selected_emi == 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['state'] == 53)].copy()
    band_list = ['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['installment_number'] == selected_emi) & (df_active_may['state'] == 53)].copy()
    band_list = ['MLA-15K', 'MLA-20K', 'MLA-30K', 'MLA-40K', 'MLA-50K']
    emi_list = [ selected_emi]

elif (selected_band != 'Overall' and selected_emi == 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['conf_product'] == selected_band) & (df_active_may['state'] == 53)].copy()  
    band_list = [selected_band]
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

else:
    df_may = df_active_may[
        (df_active_may['due_month'] == previous_month) &
        (df_active_may['conf_product'] == selected_band) & (df_active_may['installment_number'] == selected_emi) & (df_active_may['state'] == 53)].copy()
    band_list = [selected_band]
    emi_list = [selected_emi]

###################################### Ab yahan jo bhi df aayega woh conf product aur emi number se filtered hoga !!!!!! ################
###################################### Directly use df_may and df_june now they are having IN FPLP PRODUCTS ITSELF  #################################

# --- June Loop ---
for d in om_days:
    cutoff = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    june_FPLP_df = df_june[
        (df_june['closed_date'] <= cutoff) & ((df_june['newband'].isin([619,620,621,2001,2775])) | (df_june['newband'].isna())) & (df_june['state']==53)]
    june_FPLP_count=june_FPLP_df['cnt_loans'].sum()
    
    ug = june_FPLP_df[june_FPLP_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum()
    nug = june_FPLP_df[june_FPLP_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum()

    june_FPLP_cumsum.append(june_FPLP_count)
    june_upgrade_pct.append((ug / june_FPLP_count * 100) if june_FPLP_count else 0)
    june_nug_pct.append((nug / june_FPLP_count * 100) if june_FPLP_count else 0)


for d in lm_days:
    cutoff = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    # newband
    may_FPLP_df = df_may[
        (df_may['closed_date'] <= cutoff) & ((df_may['newband'].isin([619,620,621,2001,2775])) | (df_may['newband'].isna())) & (df_may['state']==53)]
    may_FPLP_count=may_FPLP_df['cnt_loans'].sum()
    
    ug = may_FPLP_df[may_FPLP_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum() # UPGRADE COUNT 
    nug = may_FPLP_df[may_FPLP_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum() # NON-UPGRADE COUNT 

    may_FPLP_cumsum.append(may_FPLP_count)
    may_upgrade_pct.append((ug / may_FPLP_count * 100) if may_FPLP_count else 0)
    may_nug_pct.append((nug / may_FPLP_count * 100) if may_FPLP_count else 0)


# --- PLOT ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Area chart – IN FPLP counts
fig.add_trace(go.Scatter(x=lm_days, y=may_FPLP_cumsum, name="In FPLP – May", fill='tozeroy',
                         line=dict(color='red')), secondary_y=False)
fig.add_trace(go.Scatter(x=om_days, y=june_FPLP_cumsum, name=f"{om_name} In FPLP", fill='tozeroy',
                         line=dict(color='lightblue')), secondary_y=False)

# Line chart – Upgrade & Non- Upgrade percentage % Month wise !!!! 
fig.add_trace(go.Scatter(x=lm_days, y=may_upgrade_pct, name=f"{lm_name} Upgrade %",
                         mode='lines+markers', line=dict(color='orange', dash='dot')), secondary_y=True)
fig.add_trace(go.Scatter(x=lm_days, y=may_nug_pct, name=f"{lm_name} Non-Upgrade %",
                         mode='lines+markers', line=dict(color='orange')), secondary_y=True)
fig.add_trace(go.Scatter(x=om_days, y=june_upgrade_pct, name=f"{om_name} Upgrade %",
                         mode='lines+markers', line=dict(color='green', dash='dot')), secondary_y=True)
fig.add_trace(go.Scatter(x=om_days, y=june_nug_pct, name=f"{om_name} Non-Upgrade %",
                         mode='lines+markers', line=dict(color='green')), secondary_y=True)
# PREDICTED #####
# --- Predicted Upgrade % Curve ---
# --- Predicted Upgrade % Curve ---
fig.add_trace(go.Scatter(
    x=om_days, y=predicted_ug_pct,
    name="Predicted Upgrade %",
    mode='lines+markers',
    line=dict(color='purple', dash='dot')  # 🔁 Purple dashed
), secondary_y=True)

# --- Predicted Non-Upgrade % Curve ---
fig.add_trace(go.Scatter(
    x=om_actual_days, y=predicted_nug_pct,
    name="Predicted Non-Upgrade %",
    mode='lines+markers',
    line=dict(color='purple')  # 🔁 Purple solid
), secondary_y=True)

# Layout
fig.update_layout(
    title=f"In FPLP ( Targetable Base) Upgrade/Non-Upgrade % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month",
    legend_title="Metric",
    height=550,
    plot_bgcolor='white')

# Y-axis configs
fig.update_yaxes(title_text="In FPLP Count", secondary_y=False, tickformat="~s", rangemode="tozero")
fig.update_yaxes(title_text="Upgrade/Non-Upgrade %", secondary_y=True, range=[0, 110], tickformat=".1f")
# --- Show in Streamlit ---
st.plotly_chart(fig, use_container_width=True)

########################################################################################################################################
########################################## DISBURSAL STAGE #############################################################################
########################################################################################################################################

import calendar 
# Lists to store data
may_FPLP_abs,june_FPLP_abs = [],[]
may_disb_pct,june_disb_pct = [],[]

df_may['disb_next_date'] = pd.to_datetime(df_may['disb_next_date'], errors='coerce')

for d in lm_days:
    # ------------------ MAY ------------------
    cutoff_may = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    may_FPLP_df = df_may[(df_may['closed_date'] <= cutoff_may) & ((df_may['newband'].isin([619,620,621,2001,2775])) | (df_may['newband'].isna())) & (df_may['state']==53)]
    may_FPLP_count=may_FPLP_df['cnt_loans'].sum()
    may_FPLP_abs.append(may_FPLP_count)

    may_disbursed_df = df_may[
        (df_may['disb_next_month'] == last_month) &  #DateField
        (df_may['closed_month'] == last_month) &  #DateField
        (df_may['disb_next_date'] <= cutoff_may) & 
        (df_may['product_next'].isin(band_list))]
    
    may_disbursed=may_disbursed_df['cnt_loans'].sum()
    may_disb_pct.append((may_disbursed / may_FPLP_count) * 100 )
    
for d in om_days:
    # ------------------ JUNE ------------------
    disb_cutoff_june = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    # FPLP Cumulative Count – June
    june_FPLP_df = df_june[(df_june['closed_date'] <= disb_cutoff_june) & ((df_june['newband'].isin([619,620,621,2001,2775])) | (df_june['newband'].isna())) & (df_june['state']==53)]
    june_FPLP_count=june_FPLP_df['cnt_loans'].sum()
    june_FPLP_abs.append(june_FPLP_count)
    # print(june_FPLP_count)
    # Disbursals till day d – June
    june_disbursed_df = df_june[
        (df_june['disb_next_month'] == ongoing_month) & #DateField
        (df_june['closed_month'] == ongoing_month) & #DateField
        (df_june['disb_next_date'] <= disb_cutoff_june) &
        (df_june['product_next'].isin(band_list)) ]
    
    june_disbursed = june_disbursed_df['cnt_loans'].sum()
    # print(june_disbursed)
    june_disb_pct.append((june_disbursed / june_FPLP_count) * 100)


################################ DISBURSAL PREDICTION ########################################################################
#################################################################################################################################


from plotly.subplots import make_subplots
# --- Combined Plot: Area chart (FPLP absolute) + Line chart (Disbursal %) ---
fig_disb = make_subplots(specs=[[{"secondary_y": True}]])
# Area Charts – FPLP Absolute
fig_disb.add_trace(go.Scatter(x=lm_days, y=may_FPLP_abs, name=f"{lm_name} In FPLP",
                              mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')),
                   secondary_y=False)
fig_disb.add_trace(go.Scatter(x=om_days, y=june_FPLP_abs, name=f"{om_name} In FPLP",
                              mode='lines', fill='tozeroy', line=dict(color='lightblue')),
                   secondary_y=False)
# Line Charts – Disbursal %
fig_disb.add_trace(go.Scatter(x=lm_days, y=may_disb_pct, name=f"{lm_name} Disbursal %",
                              mode='lines+markers', line=dict(color='orange', dash='dash')),
                   secondary_y=True)
fig_disb.add_trace(go.Scatter(x=om_days, y=june_disb_pct, name=f"{om_name} Disbursal %",
                              mode='lines+markers', line=dict(color='green')),
                   secondary_y=True)
# Layout
fig_disb.update_layout(
    title=f"In FPLP (Targetable Base) Disbursal % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month",
    legend_title="Metric",
    height=500,
    plot_bgcolor='white'
)
# Y-axis Configs
fig_disb.update_yaxes(title_text="In FPLP Count", secondary_y=False, rangemode="tozero", tickformat="~s")
fig_disb.update_yaxes(title_text="Disbursal %", secondary_y=True, range=[0, 80], tickformat=".1f")
# Display in Streamlit
st.plotly_chart(fig_disb, use_container_width=True)

##########################################################################################################################################
###################################################### UPGRADES - NON UPGRADES DISBURSALS ################################################
##########################################################################################################################################

def get_disbursal_curves(days,df, month_str, upgrade_flag):
    abs_FPLP, disb_pct = [], []
    for d in days:
        cutoff = pd.Timestamp(f"{month_str}-{d:02d}")
        FPLP_mask = (
            (df['closed_date'] <= cutoff) &
            (df['upgrade_flag'] == upgrade_flag) &
            ((df['newband'].isin([619,620,621,2001,2775])) | (df['newband'].isna())) & 
            (df['state']==53))
        FPLP_count = df.loc[FPLP_mask, 'cnt_loans'].sum()
        abs_FPLP.append(FPLP_count)

        disbursed = df[
            (df['disb_next_month'] == month_str) &
            (df['disb_next_date'] <= cutoff) &
            (df['upgrade_flag'] == upgrade_flag) & (df['product_next'].isin(band_list)) & (df['closed_month'] == month_str)
        ]['cnt_loans'].sum()

        disb_pct.append((disbursed / FPLP_count) * 100 if FPLP_count else 0)
    return abs_FPLP, disb_pct

may_upgrade_FPLP, may_upgrade_pct = get_disbursal_curves(lm_days,df_may, last_month, "Upgrade") #DateField
june_upgrade_FPLP, june_upgrade_pct = get_disbursal_curves(om_days,df_june, ongoing_month, "Upgrade") #DateField
may_nonup_FPLP, may_nonup_pct = get_disbursal_curves(lm_days,df_may, last_month, "Non-Upgrade") #DateField
june_nonup_FPLP, june_nonup_pct = get_disbursal_curves(om_days,df_june, ongoing_month, "Non-Upgrade") #DateField


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ========== UPGRADE PLOT ==========
fig_upgrade = make_subplots(specs=[[{"secondary_y": True}]])
fig_upgrade.add_trace(go.Scatter(x=lm_days, y=may_upgrade_FPLP,
    name=f"{lm_name} Upgrade", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
fig_upgrade.add_trace(go.Scatter(x=om_days, y=june_upgrade_FPLP,
    name=f"{om_name} Upgrade", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
fig_upgrade.add_trace(go.Scatter(x=lm_days, y=may_upgrade_pct,
    name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
fig_upgrade.add_trace(go.Scatter(x=om_days, y=june_upgrade_pct,
    name=f"{om_name} Disb % ", mode='lines+markers', line=dict(color='green')), secondary_y=True)
fig_upgrade.update_layout(
    title=f"Upgrades (TB) Disbursal % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month", legend_title="Metric", height=500, plot_bgcolor='white'
)
fig_upgrade.update_yaxes(title_text="In FPLP Count", secondary_y=False, rangemode="tozero", tickformat="~s")
fig_upgrade.update_yaxes(title_text="Disbursal %", secondary_y=True, range=[0, 120], tickformat=".1f")


# ========== NON-UPGRADE PLOT ==========
fig_nonup = make_subplots(specs=[[{"secondary_y": True}]])
fig_nonup.add_trace(go.Scatter(x=lm_days, y=may_nonup_FPLP,
    name=f"{lm_name} Non-Upgrade", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
fig_nonup.add_trace(go.Scatter(x=om_days, y=june_nonup_FPLP,
    name=f"{om_name} Non-Upgrade", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
fig_nonup.add_trace(go.Scatter(x=lm_days, y=may_nonup_pct,
    name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
fig_nonup.add_trace(go.Scatter(x=om_days, y=june_nonup_pct,
    name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
fig_nonup.update_layout(
    title=f"Non-Upgrades (TB) Disbursal % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month", legend_title="Metric", height=500, plot_bgcolor='white')

fig_nonup.update_yaxes(title_text="In FPLP Count", secondary_y=False, rangemode="tozero", tickformat="~s")
fig_nonup.update_yaxes(title_text="Disbursal %", secondary_y=True, range=[0, 120], tickformat=".1f")
# Show in Streamlit
colA,colB= st.columns(2)
with colA:
    st.plotly_chart(fig_upgrade, use_container_width=True)
with colB:
    st.plotly_chart(fig_nonup, use_container_width=True)

#################################################################################################################################
############################################ SM / FM DISBURSAL ############################################################################
##################################################################################################################################

days = list(range(1, calendar.monthrange(2025, 6)[1] + 1))  # Only 1–30 for June #DateField

def get_disbursal_curves(days,df, month_str, sm_flag, upgrade_flag):
    abs_FPLP = []
    disb_pct = []

    for d in days:
        cutoff = pd.Timestamp(f"{month_str}-{d:02d}")
        # Filter for relevant FPLP products
        valid_products = ['MLA-3K', 'MLA-5K', 'MLA-8K', 'MLA-10K', 
                          'MLA-5K-R', 'MLA-8K-R', 'MLA-10K-R']
        # Cumulative count in FPLPP till date (filtered by SM/FM + Upgrade/Non-Upgrade)
        FPLP_mask = (
            (df['closed_date'] <= cutoff) &
           ((df['newband'].isin([619,620,621,2001,2775])) | (df['newband'].isna())) &
            (df['sm_flag'] == sm_flag) &
            (df['upgrade_flag'] == upgrade_flag) &
            (df['state']==53)
        )
        FPLP_count = df.loc[FPLP_mask, 'cnt_loans'].sum()
        abs_FPLP.append(FPLP_count)
        # Disbursed count till date (same filters)
        disb_mask = (
            (df['closed_month'] == month_str)&
            (df['disb_next_month'] == month_str) &
            (df['disb_next_date'] <= cutoff) &
            (df['product_next'].isin(band_list)) &
            (df['sm_flag'] == sm_flag) &
            (df['upgrade_flag'] == upgrade_flag) & 
            (df['state']==53)
        )
        disbursed = df.loc[disb_mask, 'cnt_loans'].sum()
        disb_pct.append((disbursed / FPLP_count) * 100 if FPLP_count else 0)
    return abs_FPLP, disb_pct

with st.expander("📊 View All SM/FM × Upgrade/Non-Upgrade Disbursals", expanded=False):
    col1, col2 = st.columns(2)
    # SM + Upgrade
    with col1:
        may_sm_upg, may_pct_sm_upg = get_disbursal_curves(lm_days,df_may, last_month, "SM", "Upgrade") #DateField
        june_sm_upg, june_pct_sm_upg = get_disbursal_curves(om_days,df_june, ongoing_month, "SM", "Upgrade") #DateField
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=lm_days, y=may_sm_upg, name=f"{lm_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig1.add_trace(go.Scatter(x=om_days, y=june_sm_upg, name=f"{om_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig1.add_trace(go.Scatter(x=lm_days, y=may_pct_sm_upg, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig1.add_trace(go.Scatter(x=om_days, y=june_pct_sm_upg, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig1.update_layout(title="SM – Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig1.update_yaxes(title_text="FPLP Count", secondary_y=False)
        fig1.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig1, use_container_width=True)
    # SM + Non-Upgrade
    with col2:
        may_sm_non, may_pct_sm_non = get_disbursal_curves(lm_days,df_may, last_month, "SM", "Non-Upgrade") #DateField
        june_sm_non, june_pct_sm_non = get_disbursal_curves(om_days,df_june, ongoing_month, "SM", "Non-Upgrade") #DateField
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=lm_days, y=may_sm_non, name=f"{lm_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=om_days, y=june_sm_non, name=f"{om_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=lm_days, y=may_pct_sm_non, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig2.add_trace(go.Scatter(x=om_days, y=june_pct_sm_non, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig2.update_layout(title="SM – Non-Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig2.update_yaxes(title_text="FPLP Count", secondary_y=False)
        fig2.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig2, use_container_width=True)
    col3, col4 = st.columns(2)
    # FM + Upgrade
    with col3:
        may_fm_upg, may_pct_fm_upg = get_disbursal_curves(lm_days,df_may, last_month, "FM", "Upgrade") #DateField
        june_fm_upg, june_pct_fm_upg = get_disbursal_curves(om_days,df_june, ongoing_month, "FM", "Upgrade") #DateField
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=lm_days, y=may_fm_upg, name=f"{lm_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig3.add_trace(go.Scatter(x=om_days, y=june_fm_upg, name=f"{om_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig3.add_trace(go.Scatter(x=lm_days, y=may_pct_fm_upg, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig3.add_trace(go.Scatter(x=om_days, y=june_pct_fm_upg, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig3.update_layout(title="FM – Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig3.update_yaxes(title_text="FPLP Count", secondary_y=False)
        fig3.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig3, use_container_width=True)
    # FM + Non-Upgrade
    with col4:
        may_fm_non, may_pct_fm_non = get_disbursal_curves(lm_days,df_may,last_month, "FM", "Non-Upgrade") #DateField
        june_fm_non, june_pct_fm_non = get_disbursal_curves(om_days,df_june, ongoing_month, "FM", "Non-Upgrade") #DateField
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=lm_days, y=may_fm_non, name=f"{lm_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig4.add_trace(go.Scatter(x=om_days, y=june_fm_non, name=f"{om_name} FPLP", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig4.add_trace(go.Scatter(x=lm_days, y=may_pct_fm_non, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig4.add_trace(go.Scatter(x=om_days, y=june_pct_fm_non, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig4.update_layout(title="FM – Non-Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig4.update_yaxes(title_text="FPLP Count", secondary_y=False)
        fig4.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig4, use_container_width=True)
