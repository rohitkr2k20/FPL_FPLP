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

from dotenv import load_dotenv
import os
import boto3

# ✅ Load .env
load_dotenv()

# ✅ Get environment variables
aws_region = os.getenv("aws_region")
aws_access_key = os.getenv("aws_access_key")
aws_secret_key = os.getenv("secret_key")
aws_session_token = os.getenv("aws_session_token")

# ✅ Optional debug check
print("Loaded AWS region:", aws_region)

# ✅ Create boto3 session
session = boto3.Session(
aws_access_key_id=aws_access_key,
aws_secret_access_key=aws_secret_key,
aws_session_token=aws_session_token,
region_name=aws_region
)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FPL-Active", layout="wide")

# # Button to run your custom code and open second app
# if st.button("Inactive FPL"):
#     # Your functionality — put any logic here
#     st.success("✅ Code block executed successfully!")
#     exec(open("inactive.py").read())

##############################################################################################################################
################################## ----- Setting up the Database Connection ----- ############################################
##############################################################################################################################
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

ongoing_month="2025-08"
last_month="2025-07"
lm_name = "July-25"
om_name = "Aug-25"
om_days=list(range(1,5)) # 1 MORE THAN THE TILL DATE 
om_actual_days= list(range(1,31))
lm_days = list(range(1,32)) # 1 MORE THAN THE COUNT OF DAYS IN THE LAST MONTH 


###########################################################################################################################
############################ ----- Query to get the last 3 month's data ( Inc current month ) ----- #######################
###########################################################################################################################

print("query is getting started !!! ")
#DateField inside the query 
query_FPL = f"""
with base_emi as 
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
where 
date_format(emi_due_date,'%Y-%m') IN ('{ongoing_month}','{last_month}')  
and product_name in ('MLA-10K','MLA-10K-INACTIVE','MLA-10K-R','MLA-5K','MLA-5K-R','MLA-8K','MLA-8K-R'))
,
state_cte as(select * from (
select be.loan_id, state, changedon, rank() over(partition by be.userid,be.loan_id order by b.id desc) as state_rnk 
from base_emi be  join yp_iceberg.yp_user_state b on be.userid = b.uid 
where date_format(b.changedon,'%Y-%m-%d') <= DATE_FORMAT(DATE_ADD('day', 1, date(closed_date)), '%Y-%m-%d')
) where state_rnk = 1
group by 1,2,3,4) 
,
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
df=load_data(query_FPL)

print(df)
# df contains the last 3 months data of df_active
# df_active : active df ( closed_date is either empty or is the current month )
# df_inactive : Inactice df ( closed date is prev month or before ) 

current_month = pd.Period(ongoing_month, freq='M')
# Current and previous month
pd.Period(ongoing_month, freq='M')
previous_month = current_month - 1 # dynamic 

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

st.title("📈 Active Base ( FPL )")
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



######################################### BAND MAP CHANGE ###################################################
import numpy as np

band_map = {
    2250: "MLA-3K",
    1070: "MLA-5K",
    4341: "MLA-5K-R",
    1071: "MLA-8K",
    4425: "MLA-8K-R",
    1072: "MLA-10K",
    4457: "MLA-10K-R"
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
        'MLA-3K': 2250,
        'MLA-5K': 1070,
        'MLA-5K-R': 4341,
        'MLA-8K': 1071,
        'MLA-8K-R': 4425,
        'MLA-10K': 1072,
        'MLA-10K-R': 4457
    }
    # Assign priority rank to bands
    band_order = {2250: 1, 1070: 2, 4341: 3,1071: 4, 4425: 5, 1072: 6, 4457: 7}
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

def active():
    st.markdown("<a href='?view=inactive' target='_blank'><button>Open Inactive</button></a>",unsafe_allow_html=True)

def inactive():
    st.title("Inactive View")

view = st.query_params.get("view", ["active"])[0]

if view == "inactive":
    inactive()
else:
    active()

    
import streamlit as st
band_options = ['Overall', 'MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
emi_options = ['Overall', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

band_list = []
emi_list = []

# selected_band = st.selectbox("🎯 Select Product Band", band_options)
# selected_emi = st.selectbox("🎯 Select Product EMI Number", emi_options)
# print(selected_band) 
# print(selected_emi)

import streamlit as st
from streamlit_pills import pills

# Pills (inside sticky header)
selected_band = pills(
    "Select Band",
    ['Overall', 'MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R'],
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
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['installment_number'] == selected_emi)].copy()
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
    emi_list = [ selected_emi]

elif (selected_band != 'Overall' and selected_emi == 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['product_name'] == selected_band)].copy()
    band_list = [selected_band]
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

else:
    df_june = df_active_june[
        (df_active_june['due_month'] == current_month) &
        (df_active_june['product_name'] == selected_band) & (df_active_june['installment_number'] == selected_emi)].copy()
    band_list = [selected_band]
    emi_list = [selected_emi]


# (Optional) Filter df_may for previous month if needed MAY DAY MAY DAY 
if (selected_band == 'Overall' and selected_emi == 'Overall'):
    df_may = df_active_may[df_active_may['due_month'] == previous_month].copy()
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['installment_number'] == selected_emi)].copy()
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
    emi_list = [ selected_emi]

elif (selected_band != 'Overall' and selected_emi == 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['product_name'] == selected_band)].copy()  
    band_list = [selected_band]
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

else:
    df_may = df_active_may[
        (df_active_may['due_month'] == previous_month) &
        (df_active_may['product_name'] == selected_band) & (df_active_may['installment_number'] == selected_emi)].copy()
    band_list = [selected_band]
    emi_list = [selected_emi]

# from here df_may and df_june are used !!!!

######################################### CLOSURE KAHANI STARTS HERE #############################

# ----- Daily EMI Due Count (Bar) -----
daily_emi_june = df_june.groupby('due_day')['cnt_loans'].sum().reindex(range(1,32), fill_value=0).cumsum()
daily_emi_may = df_may.groupby('due_day')['cnt_loans'].sum().reindex(range(1,32), fill_value=0).cumsum()
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
    name=f'{lm_name} LineUP EMI (Non-Cumulative)',
    mode='lines+markers', line=dict(color='red', dash='dash'),
    fill='tozeroy',
    text=[f"{v:.0f}" if day in annot_days else "" for day, v in zip(range(1,32), daily_emi_may)],
    textposition="top center"
))
# Daily EMI Due – June
fig.add_trace(go.Scatter(
    x=om_days,
    y=daily_emi_june,
    name=f'{om_name} LineUP EMI Due ',
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
    title=f"📊 EMI Stack by Installment Number – {lm_name} vs {om_name}",
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
    1: [6.99, 5.33, 8.82, 5.85, 9.24, 5.47, 6.56],
    2: [7.47, 5.08, 8.26, 5.37, 7.36, 5.14, 6.17],
    3: [9.35, 6.87, 9.43, 7.20, 8.66, 7.15, 8.58],
    4: [13.47, 10.19, 13.68, 10.67, 11.71, 10.90, 13.08],
    5: [19.98, 14.99, 20.27, 15.35, 17.08, 15.66, 18.79],
    6: [33.22, 27.18, 33.72, 27.07, 28.78, 27.49, 32.99],
    7: [79.72, 80.77, 75.20, 76.41, 67.63, 75.20, 90.24]}, 
    index=[ "MLA-10K-R","MLA-10K","MLA-8K-R","MLA-8K","MLA-5K-R","MLA-5K","MLA-3K"])

# Convert percentages to decimals
ratio_df = closure_ratio_df / 100

for d in lm_days:
    current_day = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    may_active = df_may[df_may['due_date'] <= current_day]['cnt_loans'].sum()
    # print("may active count : ", may_active)
    may_closed = df_may[(df_may['closed_date'] <= current_day) & (df_may['closed_month']==pd.Period(last_month, freq='M'))]['cnt_loans'].sum()      #DateField
    # print("may closed count : ", may_closed)
    may_ratio = (may_closed / may_active) * 100 if may_active else 0
    may_closure_mtd.append(may_ratio)
    # print("ratio :", may_ratio)

for d in om_days:
    current_day_june = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    # FIX: Compare day to day
    june_active = df_june[df_june['due_day'] <= d]['cnt_loans'].sum() 
    june_closed = df_june[(df_june['closed_day'] <= d) & (df_june['closed_month']==pd.Period(ongoing_month, freq='M'))]['cnt_loans'].sum()     #DateField
    june_ratio = (june_closed / june_active) * 100 if june_active else 0    
    june_closure_mtd.append(june_ratio)
    # print("june active count : ", june_active)
    # print("june closed count : ", june_closed)
    # print("ratio :", june_ratio)
    
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
    # print(" ********************************* ",lineup_matrix)
    lineup_matrix = lineup_matrix[[col for col in ratio_df.columns if col in lineup_matrix.columns]]

    if selected_band == 'Overall' and selected_emi == 'Overall':
        predicted_matrix = lineup_matrix * ratio_df
        predicted_mtd_count = predicted_matrix.values.sum()
        # print("predicted mt countttttttttttttttttttt MFFFFF",predicted_mtd_count)
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
    #######################################################################################


#######################################################################################################################################
############################################################## MTD PLOT #############################################################
#######################################################################################################################################

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_dual_axis = make_subplots(specs=[[{"secondary_y": True}]])

# --- Cumulative EMI Line-Up ---
fig_dual_axis.add_trace(go.Scatter(
    x=days, y=daily_emi_may,  # 🔴 May - Cumulative Line
    name=f'{lm_name} EMI Line up',
    mode='lines+markers', 
    fill='tozeroy',
    line=dict(color='red', dash='dash')), secondary_y=False)
fig_dual_axis.add_trace(go.Scatter(
    x=days, y=daily_emi_june,  # 🔵 June - Cumulative Area
    name=f"{om_name} - EMI Line up",
    mode='lines',
    line=dict(color='lightblue'),
    fill='tozeroy',
    opacity=0.4
), secondary_y=False)
# --- MTD Closure % Curves ---
fig_dual_axis.add_trace(go.Scatter(
    x=days, y=may_closure_mtd,  # 🟠 May - Closure %
    name=f"{lm_name} MTD Closure %",
    mode='lines+markers',
    line=dict(color='orange', dash='dash')
), secondary_y=True)

fig_dual_axis.add_trace(go.Scatter(
    x=days, y=june_closure_mtd,  # 🟢 June - Closure %
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
    1: [92.64, 97.05, 91.01, 94.88, 74.42, 92.85, None],
    2: [86.19, 90.34, 83.35, 85.44, 67.53, 82.52, None],
    3: [83.66, 85.83, 81.98, 80.09, 69.14, 74.57, None],
    4: [82.31, 84.28, 80.93, 78.52, 67.04, 73.47, None],
    5: [77.88, 80.66, 76.67, 74.26, 61.09, 70.11, None],
    6: [75.60, 77.92, 73.70, 70.25, 53.80, 64.61, None],
    7: [69.11, 73.39, 65.57, 67.31, 46.37, 59.04, None]
}, index=["MLA-10K-R","MLA-10K","MLA-8K-R","MLA-8K","MLA-5K-R","MLA-5K","MLA-3K"])

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
    name=f"{lm_name}-Closures",
    mode='lines+markers', 
    fill='tozeroy',
    line=dict(color='red', dash='dash'),
    text=[str(int(x)) if day in [1, 5, 10, 15, 20, 25, 30] else "" for day, x in zip(days_may, may_closures)],
    textposition='top center'
), secondary_y=False)

fig_conf.add_trace(go.Scatter(
    x=lm_days, y=may_confirm_pct,
    name=f"{lm_name} - Confirmation %",
    mode='lines+markers+text',
    line=dict(color='orange', dash='dash'),
    text=[f"{x:.1f}%" if day in [1, 5, 10, 15, 20, 25, 30] else "" for day, x in zip(days_may, may_confirm_pct)],
    textposition='top center'
), secondary_y=True)

# June Area + Confirm %
fig_conf.add_trace(go.Scatter(
    x=om_days, y=june_closures,
    name=f"{om_name} - Closures",
    mode='lines+markers',
    fill='tozeroy',
    line=dict(color='lightblue'),
), secondary_y=False)

fig_conf.add_trace(go.Scatter(
    x=om_days, y=june_confirm_pct,
    name=f"{om_name} Confirmation % ",
    mode='lines+markers',
    line=dict(color='green'),
), secondary_y=True)

fig_conf.add_trace(go.Scatter(
    x=om_actual_days,
    y=predicted_confirm_pct,
    name=f"{om_name} Predicted Confirmation % ",
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
FPL_band_list = ["MLA-3K","MLA-5K","MLA-5K-R","MLA-8K", "MLA-8K-R","MLA-10K","MLA-10K-R"]

def classify(row):
    conf_product = row['conf_product']
    state = row['state']
    newband=row['newband']
    if state == 34:
        return 'Reject'
    elif ((pd.isna(conf_product) and pd.isna(newband) and state == 53) or (conf_product in FPL_band_list and state in [53])):
        return 'In FPL'
    else:
        return 'Moved Out'

# Prepare closure data for movement classification
# print("june closure before",df_active_june)
df_may = df_may[df_may['closed_date'].dt.to_period("M") == pd.Period(previous_month, freq='M')]
# df_june_closure = df_active_june[df_active_june['closed_date'].dt.to_period("M") == current_month].copy()
df_june = df_june[df_june['closed_date'].dt.to_period("M") == pd.Period(current_month, freq='M')]


# print("June movement hai yeh new :::: ",df_june)

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
may_FPL = movement_mtd(lm_days,df_may, 'In FPL')
may_moved = movement_mtd(lm_days,df_may, 'Moved Out')
june_FPL = movement_mtd(om_days,df_june, 'In FPL')
june_moved = movement_mtd(om_days,df_june, 'Moved Out')
# Absolute Count Curves
may_FPL_abs = movement_cumulative(lm_days,df_may, 'In FPL')
may_moved_abs = movement_cumulative(lm_days,df_may, 'Moved Out')
june_FPL_abs = movement_cumulative(om_days,df_june, 'In FPL')
june_FPL_abs = movement_cumulative(om_days,df_june, 'In FPL')
june_FPL_abs = movement_cumulative(om_days,df_june, 'In FPL')
june_moved_abs = movement_cumulative(om_days,df_june, 'Moved Out')

#######################################################################################################################################
############################################## TARGETABLE BASE ANALYSIS - INFLOW ######################################################
#######################################################################################################################################

# TB mein kahan kahan se janta aayi !!! 
# Case 1 : Band Upgrdae ( Uograde ) - state =53 and band upg
# Case 2 : Band Downgrade ( Non-Upgrade ) - state = 53 and band down 


current_month = pd.Period(ongoing_month, freq='M')
# Current and previous month
pd.Period(ongoing_month, freq='M')
previous_month = current_month - 1 # dynamic 

#################################### Prediction ###########################################

# Confirmation of past month and current month.
may_conf_cumsum = []
june_conf_cumsum = []

# Predicted In FPL list % and absolute numbers
predicted_infpl_pct = []
predicted_infpl_daywise=[] 

# Predicted MoveOut of FPL list % and absolute numbers
predicted_move_out_pct = []
predicted_outfpl_daywise=[]

# I want this in the form of band X is_sm_ug so that I can use it for further processing !!!! 
predicted_infpl_matrix_daywise = {}
predicted_outfpl_matrix_daywise = {}
predicted_confirmation_matrix_smug_daywise = {}


##################################### In FPL & Out FPL prediction ##################################

outfpl_data = {
    1: [24.01, 21.17, 9.02, 14.12, 16.31, 12.94, 11.64],
    2: [30.56, 36.65, 10.90, 14.44, 19.84, 14.03, 12.62],
    3: [39.44, 40.53, 13.06, 17.70, 23.24, 16.84, 15.15],
    4: [55.02, 52.87, 16.43, 18.25, 18.07, 16.58, 14.92],
    5: [61.80, 59.56, 17.21, 20.38, 17.47, 18.11, 16.29],
    6: [61.83, 56.42, 17.24, 24.57, 15.17, 20.39, 18.35],
    7: [69.24, 64.78, 16.84, 29.82, 11.36, 21.54, 19.38]}

# Create DataFrame from raw data
infpl_ratio_df = pd.DataFrame(outfpl_data,index=["MLA-10K-R", "MLA-10K", "MLA-8K-R", "MLA-8K", "MLA-5K-R", "MLA-5K", "MLA-3K"])
infpl_ratio_df = 100 - infpl_ratio_df
infpl_ratio_df = infpl_ratio_df / 100
infpl_ratio_df = infpl_ratio_df.astype(float).fillna(0)

outfpl_ratio_df = pd.DataFrame(outfpl_data,index=["MLA-10K-R", "MLA-10K", "MLA-8K-R", "MLA-8K", "MLA-5K-R", "MLA-5K", "MLA-3K"])
outfpl_ratio_df = outfpl_ratio_df / 100
outfpl_ratio_df = outfpl_ratio_df.astype(float).fillna(0)

########################################## UPGRADE ########################################################################

upgrade_distribution_ratio = pd.DataFrame({
    'SM_UG': [26.29, 15.64, 22.27, 2.10, 5.08, 0.00],
    'SM_NUG': [6.48, 8.38, 7.94, 16.67, 30.29, 0.00],
    'FM_UG': [47.16, 40.40, 52.90, 18.04, 20.66, 0.00],
    'FM_NUG': [20.07, 35.58, 16.88, 63.18, 43.97, 100.00]
}, index=["MLA-10K-R", "MLA-10K", "MLA-8K-R", "MLA-8K", "MLA-5K-R", "MLA-5K"])

# Convert percentages to decimals
upgrade_distribution_ratio = upgrade_distribution_ratio / 100
upgrade_pct_infpl_daywise = []

######################################### DISBURSAL #######################################################################

# Raw data from the image
disbursal_ratio_df = pd.DataFrame({
    'SM_UG': [48.83, 63.53, 49.42, 77.18, 55.56, 0.00],
    'SM_NUG': [59.30, 59.30, 59.30, 59.30, 59.30, 59.30],
    'FM_UG': [72.04, 80.04, 75.93, 77.10, 82.13, 0.00],
    'FM_NUG': [82.05, 61.44, 90.22, 65.11, 48.67, 100.00]
}, index=["MLA-10K-R", "MLA-10K", "MLA-8K-R", "MLA-8K", "MLA-5K-R", "MLA-5K"])

# Convert percentages to decimals
disbursal_ratio_df = disbursal_ratio_df / 100

# 2. INIT THE UPGRADE FLAG MATRIX (fixed values, used in % calculation)
upgrade_flag_matrix = pd.DataFrame({
    'SM_UG': 1,
    'SM_NUG': 0,
    'FM_UG': 1,
    'FM_NUG': 0
}, index=upgrade_distribution_ratio.index).T  # Transpose so it's sm_ug_flag × band

#################################################################################################################

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
    
    # Get the line up matrix for IN and OUT 
    lineup_matrix = predicted_confirmation_matrix_daywise[d]        
    lineup_matrix_out = predicted_confirmation_matrix_daywise[d]    

    # Ensure ratio_aligned is aligned to lineup_matrix
    ratio_aligned = infpl_ratio_df.reindex(index=lineup_matrix.index).fillna(0)
    ratio_aligned = ratio_aligned[[col for col in lineup_matrix.columns if col in ratio_aligned.columns]]

    ratio_aligned_out = outfpl_ratio_df.reindex(index=lineup_matrix_out.index).fillna(0)
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

        predicted_infpl_matrix_daywise[d] = predicted_matrix.copy()
        predicted_outfpl_matrix_daywise[d] = predicted_matrix_out.copy()


    elif selected_band == 'Overall' and selected_emi != 'Overall':
        if selected_emi not in lineup_matrix.columns:
            predicted_infpl_pct.append(0)
            continue
        if selected_emi not in lineup_matrix_out.columns:
            predicted_move_out_pct.append(0)
            continue

        predicted_matrix = lineup_matrix[[selected_emi]] * ratio_aligned[[selected_emi]]
        predicted_mtd_count = predicted_matrix.values.sum()
        total_lineup = lineup_matrix[[selected_emi]].values.sum()
        predicted_infpl_matrix_daywise[d] = predicted_matrix.copy()

        predicted_matrix_out = lineup_matrix_out[[selected_emi]] * ratio_aligned_out[[selected_emi]]
        predicted_mtd_count_out = predicted_matrix_out.values.sum()
        total_lineup_out = lineup_matrix_out[[selected_emi]].values.sum()
        predicted_outfpl_matrix_daywise[d] = predicted_matrix_out.copy()


    elif selected_band != 'Overall' and selected_emi == 'Overall':
        if selected_band not in lineup_matrix.index:
            predicted_infpl_pct.append(0)
            continue

        if selected_band not in lineup_matrix_out.index:
            predicted_move_out_pct.append(0)
            continue

        predicted_matrix = lineup_matrix.loc[[selected_band]] * ratio_aligned.loc[[selected_band]]
        predicted_mtd_count = predicted_matrix.values.sum()
        total_lineup = lineup_matrix.loc[[selected_band]].values.sum()
        predicted_infpl_matrix_daywise[d] = predicted_matrix.copy()

        predicted_matrix_out = lineup_matrix_out.loc[[selected_band]] * ratio_aligned_out.loc[[selected_band]]
        predicted_mtd_count_out = predicted_matrix_out.values.sum()
        total_lineup_out = lineup_matrix_out.loc[[selected_band]].values.sum()
        predicted_outfpl_matrix_daywise[d] = predicted_matrix_out.copy()


    else:
        if selected_band not in lineup_matrix.index or selected_emi not in lineup_matrix.columns:
            predicted_infpl_pct.append(0)
            continue
        if selected_band not in lineup_matrix_out.index or selected_emi not in lineup_matrix_out.columns:
            predicted_move_out_pct.append(0)
            continue

        val = lineup_matrix.at[selected_band, selected_emi] * ratio_aligned.at[selected_band, selected_emi]
        predicted_mtd_count = val
        total_lineup = lineup_matrix.at[selected_band, selected_emi]
        predicted_matrix = pd.DataFrame([[val]], index=[selected_band], columns=[is_sm_ug])
        predicted_infpl_matrix_daywise[d] = predicted_matrix.copy()

        val_out = lineup_matrix_out.at[selected_band, selected_emi] * ratio_aligned_out.at[selected_band, selected_emi]
        predicted_mtd_count_out = val_out
        total_lineup_out = lineup_matrix_out.at[selected_band, selected_emi]
        predicted_matrix_out = pd.DataFrame([[val]], index=[selected_band], columns=[is_sm_ug])
        predicted_outfpl_matrix_daywise[d] = predicted_matrix_out.copy()

fig_combined_abs = make_subplots(specs=[[{"secondary_y": True}]])
# FPL %
fig_combined_abs.add_trace(go.Scatter(
    x=lm_days, y=may_FPL,
    name=f"{lm_name} In FPL %",
    mode='lines+markers',
    line=dict(color="#F1993A", dash='dash')),
    secondary_y=False)
fig_combined_abs.add_trace(go.Scatter(
    x=om_days, y=june_FPL,
    name=f"{om_name} In FPL % ",
    mode='lines+markers',
    line=dict(color='#388E3C')),
    secondary_y=False)

# Predicted Curves 
fig_combined_abs.add_trace(go.Scatter(
    x=om_actual_days, 
    y=predicted_infpl_pct,
    name=f"{om_name} Predicted In FPL %",
    mode='lines+markers',
    line=dict(color='purple')
), secondary_y=False)

fig_combined_abs.add_trace(go.Scatter(
    x=om_actual_days, 
    y=upgrade_pct_infpl_daywise,
    name=f"{om_name} Predicted Upgrade FPL %",
    mode='lines+markers',
    line=dict(color='purple')
), secondary_y=False)


fig_combined_abs.add_trace(go.Scatter(
    x=om_actual_days, 
    y=predicted_move_out_pct,
    name=f"{om_name} Predicted Move Out %",
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
    name=f"{lm_name} - Moved Out %",
    mode='lines+markers',
    line=dict(color="#FFA000", dash='dot')),
    secondary_y=False)
fig_combined_abs.add_trace(go.Scatter(
    x=om_days, y=june_moved,
    name=f"{om_name} Moved Out % ",
    mode='lines+markers',
    line=dict(color="#0E9302")),
    secondary_y=False)
# Layout
fig_combined_abs.update_layout(
    title=f"In FPL % ( Targettable Base + MoveOut ) – {lm_name} vs  {om_name}", #DateField
    xaxis_title="Day of Month",
    legend_title="Metric",
    plot_bgcolor='white',
    height=600,  # Slightly taller for clarity
)
# Y-Axis Customization
fig_combined_abs.update_yaxes(
    title_text="FPL %", 
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
may_FPL_cumsum, june_FPL_cumsum = [], []
may_upgrade_pct, june_upgrade_pct = [], []
may_nug_pct, june_nug_pct = [], []
predicted_ug_pct,predicted_nug_pct = [],[]

# yahan woh data hai jo in fpl hain ( Targetable base )
# --- May Loop ---
for d in lm_days:
    cutoff = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    # newband
    may_FPL_df = df_may[
        (df_may['closed_date'] <= cutoff) & ((df_may['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_may['newband'].isna())) & (df_may['state']==53)]
    may_FPL_count=may_FPL_df['cnt_loans'].sum()
    
    ug = may_FPL_df[may_FPL_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum() # UPGRADE COUNT 
    nug = may_FPL_df[may_FPL_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum() # NON-UPGRADE COUNT 

    may_FPL_cumsum.append(may_FPL_count)
    may_upgrade_pct.append((ug / may_FPL_count * 100) if may_FPL_count else 0)
    may_nug_pct.append((nug / may_FPL_count * 100) if may_FPL_count else 0)

# --- June Loop ---
for d in om_days:
    cutoff = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    june_FPL_df = df_june[
        (df_june['closed_date'] <= cutoff) & ((df_june['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_june['newband'].isna())) & (df_june['state']==53)]
    june_FPL_count=june_FPL_df['cnt_loans'].sum()
    
    ug = june_FPL_df[june_FPL_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum()
    nug = june_FPL_df[june_FPL_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum()

    june_FPL_cumsum.append(june_FPL_count)
    june_upgrade_pct.append((ug / june_FPL_count * 100) if june_FPL_count else 0)
    june_nug_pct.append((nug / june_FPL_count * 100) if june_FPL_count else 0)

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
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_june = df_active_june[(df_active_june['due_month'] == current_month) & (df_active_june['installment_number'] == selected_emi) & (df_active_june['state'] == 53)].copy()
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
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
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
    emi_list = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12]

elif (selected_band == 'Overall' and selected_emi != 'Overall'):
    df_may = df_active_may[(df_active_may['due_month'] == previous_month) & (df_active_may['installment_number'] == selected_emi) & (df_active_may['state'] == 53)].copy()
    band_list = ['MLA-5K', 'MLA-5K-R', 'MLA-8K', 'MLA-8K-R', 'MLA-10K', 'MLA-10K-R']
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
###################################### Directly use df_may and df_june now they are having IN FPL PRODUCTS ITSELF  #################################

# --- June Loop ---
for d in om_days:
    cutoff = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    june_FPL_df = df_june[
        (df_june['closed_date'] <= cutoff) & ((df_june['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_june['newband'].isna())) & (df_june['state']==53)]
    june_FPL_count=june_FPL_df['cnt_loans'].sum()
    
    ug = june_FPL_df[june_FPL_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum()
    nug = june_FPL_df[june_FPL_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum()

    june_FPL_cumsum.append(june_FPL_count)
    june_upgrade_pct.append((ug / june_FPL_count * 100) if june_FPL_count else 0)
    june_nug_pct.append((nug / june_FPL_count * 100) if june_FPL_count else 0)


for d in lm_days:
    cutoff = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    # newband
    may_FPL_df = df_may[
        (df_may['closed_date'] <= cutoff) & ((df_may['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_may['newband'].isna())) & (df_may['state']==53)]
    may_FPL_count=may_FPL_df['cnt_loans'].sum()
    
    ug = may_FPL_df[may_FPL_df['upgrade_flag'] == 'Upgrade']['cnt_loans'].sum() # UPGRADE COUNT 
    nug = may_FPL_df[may_FPL_df['upgrade_flag'] == 'Non-Upgrade']['cnt_loans'].sum() # NON-UPGRADE COUNT 

    may_FPL_cumsum.append(may_FPL_count)
    may_upgrade_pct.append((ug / may_FPL_count * 100) if may_FPL_count else 0)
    may_nug_pct.append((nug / may_FPL_count * 100) if may_FPL_count else 0)


################################################################################################################################

# --- PLOT ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Area chart – IN FPL counts
fig.add_trace(go.Scatter(x=lm_days, y=may_FPL_cumsum, name=f"{lm_name} In FPL", fill='tozeroy',
                         line=dict(color='red')), secondary_y=False)
fig.add_trace(go.Scatter(x=om_days, y=june_FPL_cumsum, name=f"{om_name} In FPL", fill='tozeroy',
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
fig.add_trace(go.Scatter(
    x=days, y=upgrade_pct_infpl_daywise,
    name="Predicted Upgrade %",
    mode='lines+markers',
    line=dict(color='purple', dash='dot')  # 🔁 Purple dashed
), secondary_y=True)

# # --- Predicted Non-Upgrade % Curve ---
# fig.add_trace(go.Scatter(
#     x=days, y=predicted_nug_pct,
#     name="Predicted Non-Upgrade %",
#     mode='lines+markers',
#     line=dict(color='purple')  # 🔁 Purple solid
# ), secondary_y=True)

# Layout
fig.update_layout(
    title=f"In FPL ( Targetable Base) Upgrade/Non-Upgrade % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month",
    legend_title="Metric",
    height=550,
    plot_bgcolor='white')

# Y-axis configs
fig.update_yaxes(title_text="In FPL Count", secondary_y=False, tickformat="~s", rangemode="tozero")
fig.update_yaxes(title_text="Upgrade/Non-Upgrade %", secondary_y=True, range=[0, 110], tickformat=".1f")
# --- Show in Streamlit ---
st.plotly_chart(fig, use_container_width=True)

########################################################################################################################################
########################################## DISBURSAL STAGE #############################################################################
########################################################################################################################################

import calendar 
# Lists to store data
may_FPL_abs,june_FPL_abs = [],[]
may_disb_pct,june_disb_pct = [],[]

df_may['disb_next_date'] = pd.to_datetime(df_may['disb_next_date'], errors='coerce')

for d in lm_days:
    # ------------------ MAY ------------------
    cutoff_may = pd.Timestamp(f"{last_month}-{d:02d}") #DateField
    may_FPL_df = df_may[(df_may['closed_date'] <= cutoff_may) & ((df_may['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_may['newband'].isna())) & (df_may['state']==53)]
    may_FPL_count = may_FPL_df['cnt_loans'].sum()
    may_FPL_abs.append(may_FPL_count)
    print("may in fpl count :::",may_FPL_count)

    may_disbursed_df = df_may[
        (df_may['disb_next_month'] == last_month) &  #DateField
        (df_may['closed_month'] == last_month) &  #DateField
        (df_may['disb_next_date'] <= cutoff_may) & 
        # ((df_may['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_may['newband'].isna())) & (df_may['state']==53) &
        (df_may['product_next'].isin(band_list))]
    
    may_disbursed=may_disbursed_df['cnt_loans'].sum()
    print("may DISBURSED count :::",may_disbursed)
    may_disb_pct.append((may_disbursed / may_FPL_count) * 100 )
    
for d in om_days:
    # ------------------ JUNE ------------------
    disb_cutoff_june = pd.Timestamp(f"{ongoing_month}-{d:02d}") #DateField
    # FPL Cumulative Count – June
    june_FPL_df = df_june[(df_june['closed_date'] <= disb_cutoff_june) & ((df_june['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df_june['newband'].isna())) & (df_june['state']==53)]
    june_FPL_count=june_FPL_df['cnt_loans'].sum()
    june_FPL_abs.append(june_FPL_count)
    # print(june_FPL_count)
    # Disbursals till day d – June
    june_disbursed_df = df_june[
        (df_june['disb_next_month'] == ongoing_month) & #DateField
        (df_june['closed_month'] == ongoing_month) & #DateField
        (df_june['disb_next_date'] <= disb_cutoff_june) &
        (df_june['product_next'].isin(band_list)) ]
    
    june_disbursed = june_disbursed_df['cnt_loans'].sum()
    # print(june_disbursed)
    june_disb_pct.append((june_disbursed / june_FPL_count) * 100)


################################ DISBURSAL PREDICTION ########################################################################

# --- Combined Plot: Area chart (FPL absolute) + Line chart (Disbursal %) ---
fig_disb = make_subplots(specs=[[{"secondary_y": True}]])
# Area Charts – FPL Absolute
fig_disb.add_trace(go.Scatter(x=lm_days, y=may_FPL_abs, name=f"{lm_name} In FPL",
                              mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')),
                   secondary_y=False)
fig_disb.add_trace(go.Scatter(x=om_days, y=june_FPL_abs, name=f"{om_name} In FPL",
                              mode='lines', fill='tozeroy', line=dict(color='lightblue')),
                   secondary_y=False)
# Line Charts – Disbursal %
fig_disb.add_trace(go.Scatter(x=lm_days, y=may_disb_pct, name=f"{lm_name} Disbursal %",
                              mode='lines+markers', line=dict(color='orange', dash='dash')),
                   secondary_y=True)
fig_disb.add_trace(go.Scatter(x=om_days, y=june_disb_pct, name=f"{om_name} Disbursal %",
                              mode='lines+markers', line=dict(color='green')),
                   secondary_y=True)

################### Fake disbursals #####################
fk_disb=[53,53.5,54,54.6,54.8,55,55.2,55.6,56.2,56.7,57.1,57.5,57.9,58.2,58.6,59.01,59.3,59.8,60.1,60.31,60.45,60.63,60.91,62.1,62.8,63.2,63.9,64.3,63.9,64.1]
fig_disb.add_trace(go.Scatter(x=om_actual_days, y=fk_disb, name=f"{om_name} Disbursal %",
                              mode='lines+markers', line=dict(color='purple')),
                   secondary_y=True)
# Layout
fig_disb.update_layout(
    title=f"In FPL (Targetable Base) Disbursal % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month",
    legend_title="Metric",
    height=500,
    plot_bgcolor='white'
)
# Y-axis Configs
fig_disb.update_yaxes(title_text="In FPL Count", secondary_y=False, rangemode="tozero", tickformat="~s")
fig_disb.update_yaxes(title_text="Disbursal %", secondary_y=True, range=[0, 80], tickformat=".1f")
# Display in Streamlit
st.plotly_chart(fig_disb, use_container_width=True)

##########################################################################################################################################
###################################################### UPGRADES - NON UPGRADES DISBURSALS ################################################
##########################################################################################################################################

def get_disbursal_curves(days,df, month_str, upgrade_flag):
    abs_FPL, disb_pct = [], []
    for d in days:
        cutoff = pd.Timestamp(f"{month_str}-{d:02d}")
        FPL_mask = (
            (df['closed_date'] <= cutoff) &
            (df['upgrade_flag'] == upgrade_flag) &
            ((df['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df['newband'].isna())) & (df['state']==53))
        FPL_count = df.loc[FPL_mask, 'cnt_loans'].sum()
        abs_FPL.append(FPL_count)

        disbursed = df[
            (df['disb_next_month'] == month_str) &
            (df['closed_month'] == month_str) &
            (df['disb_next_date'] <= cutoff) &
            (df['upgrade_flag'] == upgrade_flag) & (df['product_next'].isin(band_list))
        ]['cnt_loans'].sum()

        disb_pct.append((disbursed / FPL_count) * 100 if FPL_count else 0)
    return abs_FPL, disb_pct

may_upgrade_FPL, may_upgrade_pct = get_disbursal_curves(lm_days,df_may, last_month, "Upgrade") #DateField
june_upgrade_FPL, june_upgrade_pct = get_disbursal_curves(om_days,df_june, ongoing_month, "Upgrade") #DateField
may_nonup_FPL, may_nonup_pct = get_disbursal_curves(lm_days,df_may, last_month, "Non-Upgrade") #DateField
june_nonup_FPL, june_nonup_pct = get_disbursal_curves(om_days,df_june, ongoing_month, "Non-Upgrade") #DateField


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ========== UPGRADE PLOT ==========
fig_upgrade = make_subplots(specs=[[{"secondary_y": True}]])
fig_upgrade.add_trace(go.Scatter(x=lm_days, y=may_upgrade_FPL,
    name=f"{lm_name} -Upgrade", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
fig_upgrade.add_trace(go.Scatter(x=om_days, y=june_upgrade_FPL,
    name=f"{om_name} Upgrade", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
fig_upgrade.add_trace(go.Scatter(x=lm_days, y=may_upgrade_pct,
    name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
fig_upgrade.add_trace(go.Scatter(x=om_days, y=june_upgrade_pct,
    name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
fig_upgrade.add_trace(go.Scatter(x=om_actual_days, y=fk_disb, name=f"{om_name} Disbursal %",
                              mode='lines+markers', line=dict(color='purple')),
                   secondary_y=True)
fig_upgrade.update_layout(
    title=f"Upgrades (TB) Disbursal % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month", legend_title="Metric", height=500, plot_bgcolor='white'
)
fig_upgrade.update_yaxes(title_text="In FPL Count", secondary_y=False, rangemode="tozero", tickformat="~s")
fig_upgrade.update_yaxes(title_text="Disbursal %", secondary_y=True, range=[0, 120], tickformat=".1f")



# ========== NON-UPGRADE PLOT ==========
fig_nonup = make_subplots(specs=[[{"secondary_y": True}]])
fig_nonup.add_trace(go.Scatter(x=lm_days, y=may_nonup_FPL,
    name=f"{lm_name} Non-Upgrade", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
fig_nonup.add_trace(go.Scatter(x=om_days, y=june_nonup_FPL,
    name=f"{om_name} Non-Upgrade", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
fig_nonup.add_trace(go.Scatter(x=lm_days, y=may_nonup_pct,
    name=f"{lm_name} Disb % ", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
fig_nonup.add_trace(go.Scatter(x=om_days, y=june_nonup_pct,
    name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
fig_nonup.add_trace(go.Scatter(x=om_actual_days, y=fk_disb, name=f"{om_name} Disbursal %",
                              mode='lines+markers', line=dict(color='purple')),
                   secondary_y=True)
fig_nonup.update_layout(
    title=f"Non-Upgrades (TB) Disbursal % – {lm_name} vs {om_name}", #DateField
    xaxis_title="Day of Month", legend_title="Metric", height=500, plot_bgcolor='white')

fig_nonup.update_yaxes(title_text="In FPL Count", secondary_y=False, rangemode="tozero", tickformat="~s")
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
    abs_FPL = []
    disb_pct = []

    for d in days:
        cutoff = pd.Timestamp(f"{month_str}-{d:02d}")
        # Filter for relevant FPL products
        valid_products = ['MLA-3K', 'MLA-5K', 'MLA-8K', 'MLA-10K', 
                          'MLA-5K-R', 'MLA-8K-R', 'MLA-10K-R']
        # Cumulative count in FPLP till date (filtered by SM/FM + Upgrade/Non-Upgrade)
        FPL_mask = (
            (df['closed_date'] <= cutoff) &
           ((df['newband'].isin([2250,1070,4341,1071,4425,1072,4457])) | (df['newband'].isna())) &
            (df['sm_flag'] == sm_flag) &
            (df['upgrade_flag'] == upgrade_flag) &
            (df['state']==53)
        )
        FPL_count = df.loc[FPL_mask, 'cnt_loans'].sum()
        abs_FPL.append(FPL_count)
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
        disb_pct.append((disbursed / FPL_count) * 100 if FPL_count else 0)
    return abs_FPL, disb_pct

with st.expander("📊 View All SM/FM × Upgrade/Non-Upgrade Disbursals", expanded=False):
    col1, col2 = st.columns(2)
    # SM + Upgrade
    with col1:
        may_sm_upg, may_pct_sm_upg = get_disbursal_curves(lm_days,df_may, last_month, "SM", "Upgrade") #DateField
        june_sm_upg, june_pct_sm_upg = get_disbursal_curves(om_days,df_june, ongoing_month, "SM", "Upgrade") #DateField
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=lm_days, y=may_sm_upg, name=f"FPL {lm_name}", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig1.add_trace(go.Scatter(x=om_days, y=june_sm_upg, name=f"{om_name} FPL", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig1.add_trace(go.Scatter(x=lm_days, y=may_pct_sm_upg, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig1.add_trace(go.Scatter(x=om_days, y=june_pct_sm_upg, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig1.update_layout(title="SM – Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig1.update_yaxes(title_text="FPL Count", secondary_y=False)
        fig1.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig1, use_container_width=True)
    # SM + Non-Upgrade
    with col2:
        may_sm_non, may_pct_sm_non = get_disbursal_curves(lm_days,df_may, last_month, "SM", "Non-Upgrade") #DateField
        june_sm_non, june_pct_sm_non = get_disbursal_curves(om_days,df_june, ongoing_month, "SM", "Non-Upgrade") #DateField
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=lm_days, y=may_sm_non, name=f"{lm_name} FPL", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=om_days, y=june_sm_non, name=f"{om_name} FPL", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=lm_days, y=may_pct_sm_non, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig2.add_trace(go.Scatter(x=om_days, y=june_pct_sm_non, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig2.update_layout(title="SM – Non-Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig2.update_yaxes(title_text="FPL Count", secondary_y=False)
        fig2.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig2, use_container_width=True)
    col3, col4 = st.columns(2)
    # FM + Upgrade
    with col3:
        may_fm_upg, may_pct_fm_upg = get_disbursal_curves(lm_days,df_may, last_month, "FM", "Upgrade") #DateField
        june_fm_upg, june_pct_fm_upg = get_disbursal_curves(om_days,df_june, ongoing_month, "FM", "Upgrade") #DateField
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=lm_days, y=may_fm_upg, name=f"{lm_name} FPL", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig3.add_trace(go.Scatter(x=om_days, y=june_fm_upg, name=f"{om_name} FPL", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig3.add_trace(go.Scatter(x=lm_days, y=may_pct_fm_upg, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig3.add_trace(go.Scatter(x=om_days, y=june_pct_fm_upg, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig3.update_layout(title="FM – Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig3.update_yaxes(title_text="FPL Count", secondary_y=False)
        fig3.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig3, use_container_width=True)
    # FM + Non-Upgrade
    with col4:
        may_fm_non, may_pct_fm_non = get_disbursal_curves(lm_days,df_may,last_month, "FM", "Non-Upgrade") #DateField
        june_fm_non, june_pct_fm_non = get_disbursal_curves(om_days,df_june, ongoing_month, "FM", "Non-Upgrade") #DateField
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Scatter(x=lm_days, y=may_fm_non, name=f"{lm_name} FPL", mode='lines', fill='tozeroy', line=dict(color='red', dash='dot')), secondary_y=False)
        fig4.add_trace(go.Scatter(x=om_days, y=june_fm_non, name=f"{om_name} FPL", mode='lines', fill='tozeroy', line=dict(color='lightblue')), secondary_y=False)
        fig4.add_trace(go.Scatter(x=lm_days, y=may_pct_fm_non, name=f"{lm_name} Disb %", mode='lines+markers', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig4.add_trace(go.Scatter(x=om_days, y=june_pct_fm_non, name=f"{om_name} Disb %", mode='lines+markers', line=dict(color='green')), secondary_y=True)
        fig4.update_layout(title="FM – Non-Upgrade Disbursal %", height=400, plot_bgcolor='white')
        fig4.update_yaxes(title_text="FPL Count", secondary_y=False)
        fig4.update_yaxes(title_text="Disb %", secondary_y=True, range=[0, 110])
        st.plotly_chart(fig4, use_container_width=True)


