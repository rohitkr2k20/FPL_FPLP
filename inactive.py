import streamlit as st
import boto3
import time
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import calendar

st.set_page_config(page_title="FPL/FPLP Inactive", layout="wide")
st.header("📊 Inactive Base Analysis")

# Get today's date
today = datetime.today()
ongoing_month = today.strftime("%Y-%m")
# Last month = first day of this month - 1 day → gives last month's date
first_day_this_month = today.replace(day=1)
last_month_date = first_day_this_month - timedelta(days=1)
last_month = last_month_date.strftime("%Y-%m")

################################################################################################################
################################## ----- Setting up the Database Connection ----- ##############################
################################################################################################################

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
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
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
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
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
    

###########################################################################################################################
############################ ----- Query to get the last 3 month's data ( Inc current month ) ----- #######################
##########################################################################################################################
# Today's date
today = datetime.today()

# First day of the current month
first_day_current = today.replace(day=1)
first_day_current_str = first_day_current.strftime("%Y-%m-%d")

# First day of previous month
# Step 1: Subtract 1 day from the first of this month → gives last day of previous month
last_day_prev_month = first_day_current - timedelta(days=1)

# Step 2: Replace day=1 to get first day of that previous month
first_day_prev = last_day_prev_month.replace(day=1)
first_day_prev_str = first_day_prev.strftime("%Y-%m-%d")

# Print both
print("First day of current month:", first_day_current_str)
print("First day of previous month:", first_day_prev_str)


query_fpl_may = f"""
with combine_user_trace_mask as
(
    SELECT *
    FROM
    (
        (
        SELECT
            userId,
            bandId,
            DATE_FORMAT((from_unixtime(creationTime / 1000) + interval '5' hour + interval '30' minute), '%Y-%m-%d')  as createdatee,
            subBand,
            id
            FROM "yp_archived"."yp_user_trace"
            where
            DATE_FORMAT((from_unixtime(creationTime / 1000) + interval '5' hour + interval '30' minute), '%Y-%m') < '2024-07'
           
        )
   
        UNION ALL
   
      SELECT
            userId,
            bandId,
            DATE_FORMAT(((creationTime) + interval '5' hour + interval '30' minute), '%Y-%m-%d')  as createdatee,
            subBand,
            id
            FROM yp_iceberg.yp_user_trace_mask
            where
            DATE_FORMAT(((creationTime ) + interval '5' hour + interval '30' minute), '%Y-%m') >= '2024-07'
           
    )
),


-- select min(DATE_FORMAT(((creationTime) + interval '5' hour + interval '30' minute), '%Y-%m-%d')) from
-- yp_iceberg.yp_user_trace_mask



c1 as (SELECT Cx_Type,
     CASE
     WHEN BAND IN ('diamond') then 'SA'
     WHEN BAND IN ('FBL','FBLSILVER','FBLSILVER-R','FBLGOLD','FBLGOLD-R','FBLPLATINUM','FBLPLATINUM-R') then 'FBL'
     WHEN BAND IN ('FBL2','FBL4','FBL6','FBL8','FBL10') then 'FBLP'
     WHEN Cx_Type IN ('New') AND
     BAND IN ('FPL1', 'FPL2','FPL3','FPL4','FPL5','FPL6','FPL7','FPL8','FPL9','FPL10',
                    'MLASILVER','MLASILVER-R','MLAGOLD','MLAGOLD-R','MLAGOLD_INACTIVE','MLAPLATINUM','MLAPLATINUM-R','MLAPLATINUM_INACTIVE') then 'FPL-N'
     WHEN Cx_Type IN ('Active_Loan','Inactive_Loan') AND
     BAND IN ('FPL1', 'FPL2','FPL3','FPL4','FPL5','FPL6','FPL7','FPL8','FPL9','FPL10',
                    'MLASILVER','MLASILVER-R','MLAGOLD','MLAGOLD-R','MLAGOLD_INACTIVE','MLAPLATINUM','MLAPLATINUM-R','MLAPLATINUM_INACTIVE') then 'FPL-R'
     WHEN Cx_Type IN ('New') AND BAND IN ('MLA2','MLA4','MLA6','MLA8','MLA10') then 'FPLP-N'
     WHEN Cx_Type IN ('Active_Loan','Inactive_Loan') AND BAND IN ('MLA2','MLA4','MLA6','MLA8','MLA10') then 'FPLP-R'
     WHEN BAND IN ('MLAX') AND SUB_BAND NOT IN ('good_mlax10','default_mlax10','bad_mlax10','ntc_mlax10','default1_mlax10','default2_mlax10','good_mlax11','good_mlax12','bad_mlax12') then 'SE'
     WHEN BAND IN ('MLAX') AND SUB_BAND IN ('good_mlax10','default_mlax10','bad_mlax10','ntc_mlax10','default1_mlax10','default2_mlax10','good_mlax11','good_mlax12','bad_mlax12') then 'PBL'
     WHEN BAND IN ('SME') then 'PBL'
     ELSE 'Others' END AS Product,
    y.uid,
    BAND,
    createdatee,
    loan_id2,
    closedon_time,
    DATE_FORMAT(TRY_CAST(Latest_SA_Closed_Date AS TIMESTAMP), '%Y-%m') AS Closed_month

FROM
(SELECT b.uid,b.state,b.substate,a.Firstloan_productName,
d.Latesst_SA_loan_taken_Date, d.Latest_SA_Closed_Date, d.Latest_SA_Closed_Month, d.Latest_SA_Loan_State, e.createdatee,loan_id2,closedon_time,
e.bandId, (SELECT val from yp_iceberg.yp_key where id = e.bandId) as BAND,
e.subBand, (SELECT val from yp_iceberg.yp_key where id = e.subBand) as SUB_BAND,

IF(a.Firstloan_productName IS NULL,'New', IF(d.Latest_SA_Closed_Date < '{first_day_prev_str}','Inactive_Loan','Active_Loan')) as Cx_Type
FROM (
SELECT uid, rank() over(partition by uid order by id desc) rank1,state,substate
from yp_iceberg.yp_user_state
where DATE_FORMAT(( (changedOn  ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') < '{first_day_prev_str}') b
LEFT JOIN (SELECT s.userid as userid,s.productname as Firstloan_productName
          FROM (SELECT l.userid,l.productname,Dense_Rank() over(partition by l.userid order by l.disbursedon ASC) as DenseRank
          FROM yp_iceberg.yp_loan l WHERE l.state IN(47,71) AND
          DATE_FORMAT(( (l.disbursedon ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') < '{first_day_prev_str}') s
          WHERE s.DenseRank=1) a ON b.uid=a.userid
LEFT JOIN (SELECT s.userid as userid,s.productname as Latest_SA_productName, loan_id2,
           DATE_FORMAT(( (s.disbursedon ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') as Latesst_SA_loan_taken_Date,
           DATE_FORMAT(( (s.closedOn ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') as Latest_SA_Closed_Date,
                      DATE_FORMAT(( (s.closedOn ) + interval '5' hour + interval '30' minute),'%Y-%m-%d %H:%i:%s') as closedon_time,
           DATE_FORMAT(( (s.closedOn ) + interval '5' hour + interval '30' minute),'%Y-%m') as Latest_SA_Closed_Month,
           (SELECT val FROM yp_iceberg.yp_key WHERE id=s.state) as Latest_SA_Loan_State
           FROM
           (SELECT l.userid,l.productname,l.disbursedon,l.state,l.closedOn,   l.id AS loan_id2,
           Dense_Rank() over(partition by l.userid order by l.disbursedon DESC) as DenseRank
           FROM yp_iceberg.yp_loan l WHERE l.state IN(47,71) AND
           DATE_FORMAT(( (l.disbursedon ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') < '{first_day_prev_str}') s
           WHERE s.DenseRank=1) d ON b.uid=d.userid
LEFT JOIN (SELECT r.userId,r.bandId,r.subBand,r.createdatee FROM (SELECT l.userId,l.bandId, l.createdatee, l.subBand,rank() OVER(PARTITION BY l.userId ORDER BY l.id DESC) as Rank1
           FROM combine_user_trace_mask l
           WHERE l.bandId IS NOT NULL AND l.subBand IS NOT NULL AND
l.createdatee < '{first_day_prev_str}'

           ) r
           WHERE Rank1=1) e ON b.uid=e.userId
WHERE rank1=1 and state=53) y
),

-- select * from c1
-- where
-- Product = 'Others'
-- and 
-- uid in (165367781
-- )
c2 as (
select Cx_Type, Product, band, DATE_FORMAT(TRY_CAST(createdatee AS TIMESTAMP), '%Y-%m') AS created_month ,
 CASE
        WHEN Closed_month = '{first_day_prev_str[:7]}' THEN 'M'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 6) THEN 'M-1'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 5) THEN 'M-2'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 4) THEN 'M-3'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 3) THEN 'M-4'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 2) THEN 'M-5'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 1) THEN 'M-6'
        ELSE '>M-6'
    END AS Closed_month_bucket,
Closed_month,
loan_id2,
closedon_time,

uid from c1
-- where Product = 'Others'
),

c3 AS (
  SELECT  
   c2.*,    
    xx.userid,
    xx.id AS loan_idd,
    xx.principalDue AS pricipal_disb,
    pmuu.product_name AS product_type_disb,
    CAST(DATE_FORMAT(xx.disbursedon, '%Y-%m-%d') AS DATE) AS disb_date,
    DATE_FORMAT(xx.disbursedon, '%Y-%m') AS disbb_month,
    xx.productname AS disb_productname,
    RANK() OVER (PARTITION BY c2.uid, c2.loan_id2 ORDER BY xx.id ASC) AS rnk,
      RANK() OVER (PARTITION BY xx.userid,  DATE_FORMAT(( (xx.disbursedon ) + INTERVAL '5' HOUR + INTERVAL '30' MINUTE), '%Y-%m') ORDER BY xx.id ASC) AS loan_rank
  FROM c2
  LEFT JOIN (
    yp_iceberg.yp_loan xx
    LEFT JOIN kreditbee_bi_dw_iceberg.yp_product_name_mapping pmuu
      ON xx.disbursedon BETWEEN pmuu.disb_start_date
      AND COALESCE(pmuu.disb_end_date, CURRENT_TIMESTAMP)
      AND (
        (pmuu.subbandid IS NULL AND xx.productname = pmuu.tech_product_name)
        OR
        (pmuu.subbandid = xx.subbandid AND xx.productname = pmuu.tech_product_name)
      )
  )
    ON xx.userid = c2.uid
    AND xx.state IN (47, 71)
    AND DATE_FORMAT((xx.disbursedon ) + INTERVAL '5' HOUR + INTERVAL '30' MINUTE, '%Y-%m-%d %H:%i:%s') > c2.closedon_time
    AND xx.id > c2.loan_id2
)

select
 Cx_Type, Product, band, Closed_month_bucket, disbb_month, product_type_disb, disb_productname,
 Closed_month,
--  loan_rank,
--  rnk,
 disb_date,
 count(loan_idd) as loan_cx,
 count(distinct(uid)) as uidd
from c3
where rnk = 1
group by 1,2,3,4,5,6,7,8,9 ; 

"""
df_inactive_may=load_data(query_fpl_may)


query_fpl_june = f"""
with combine_user_trace_mask as
(

    SELECT *
    FROM
    (
        (
        SELECT
            userId,
            bandId,
            DATE_FORMAT((from_unixtime(creationTime / 1000) + interval '5' hour + interval '30' minute), '%Y-%m-%d')  as createdatee,
            subBand,
            id
            FROM "yp_archived"."yp_user_trace"
            where
            DATE_FORMAT((from_unixtime(creationTime / 1000) + interval '5' hour + interval '30' minute), '%Y-%m') < '2024-07'
           
        )
   
        UNION ALL
   
      SELECT
            userId,
            bandId,
            DATE_FORMAT(((creationTime) + interval '5' hour + interval '30' minute), '%Y-%m-%d')  as createdatee,
            subBand,
            id
            FROM yp_iceberg.yp_user_trace_mask
            where
            DATE_FORMAT(((creationTime ) + interval '5' hour + interval '30' minute), '%Y-%m') >= '2024-07'
           
    )
),


-- select min(DATE_FORMAT(((creationTime) + interval '5' hour + interval '30' minute), '%Y-%m-%d')) from
-- yp_iceberg.yp_user_trace_mask



c1 as (SELECT Cx_Type,
     CASE
     WHEN BAND IN ('diamond') then 'SA'
     WHEN BAND IN ('FBL','FBLSILVER','FBLSILVER-R','FBLGOLD','FBLGOLD-R','FBLPLATINUM','FBLPLATINUM-R') then 'FBL'
     WHEN BAND IN ('FBL2','FBL4','FBL6','FBL8','FBL10') then 'FBLP'
     WHEN Cx_Type IN ('New') AND
     BAND IN ('FPL1', 'FPL2','FPL3','FPL4','FPL5','FPL6','FPL7','FPL8','FPL9','FPL10',
                    'MLASILVER','MLASILVER-R','MLAGOLD','MLAGOLD-R','MLAGOLD_INACTIVE','MLAPLATINUM','MLAPLATINUM-R','MLAPLATINUM_INACTIVE') then 'FPL-N'
     WHEN Cx_Type IN ('Active_Loan','Inactive_Loan') AND
     BAND IN ('FPL1', 'FPL2','FPL3','FPL4','FPL5','FPL6','FPL7','FPL8','FPL9','FPL10',
                    'MLASILVER','MLASILVER-R','MLAGOLD','MLAGOLD-R','MLAGOLD_INACTIVE','MLAPLATINUM','MLAPLATINUM-R','MLAPLATINUM_INACTIVE') then 'FPL-R'
     WHEN Cx_Type IN ('New') AND BAND IN ('MLA2','MLA4','MLA6','MLA8','MLA10') then 'FPLP-N'
     WHEN Cx_Type IN ('Active_Loan','Inactive_Loan') AND BAND IN ('MLA2','MLA4','MLA6','MLA8','MLA10') then 'FPLP-R'
     WHEN BAND IN ('MLAX') AND SUB_BAND NOT IN ('good_mlax10','default_mlax10','bad_mlax10','ntc_mlax10','default1_mlax10','default2_mlax10','good_mlax11','good_mlax12','bad_mlax12') then 'SE'
     WHEN BAND IN ('MLAX') AND SUB_BAND IN ('good_mlax10','default_mlax10','bad_mlax10','ntc_mlax10','default1_mlax10','default2_mlax10','good_mlax11','good_mlax12','bad_mlax12') then 'PBL'
     WHEN BAND IN ('SME') then 'PBL'
     ELSE 'Others' END AS Product,
    y.uid,
    BAND,
    createdatee,
    loan_id2,
    closedon_time,
    DATE_FORMAT(TRY_CAST(Latest_SA_Closed_Date AS TIMESTAMP), '%Y-%m') AS Closed_month

FROM
(SELECT b.uid,b.state,b.substate,a.Firstloan_productName,
d.Latesst_SA_loan_taken_Date, d.Latest_SA_Closed_Date, d.Latest_SA_Closed_Month, d.Latest_SA_Loan_State, e.createdatee,loan_id2,closedon_time,
e.bandId, (SELECT val from yp_iceberg.yp_key where id = e.bandId) as BAND,
e.subBand, (SELECT val from yp_iceberg.yp_key where id = e.subBand) as SUB_BAND,

IF(a.Firstloan_productName IS NULL,'New', IF(d.Latest_SA_Closed_Date < '{first_day_current_str}','Inactive_Loan','Active_Loan')) as Cx_Type
FROM (
SELECT uid, rank() over(partition by uid order by id desc) rank1,state,substate
from yp_iceberg.yp_user_state
where DATE_FORMAT(( (changedOn  ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') < '{first_day_current_str}') b
LEFT JOIN (SELECT s.userid as userid,s.productname as Firstloan_productName
          FROM (SELECT l.userid,l.productname,Dense_Rank() over(partition by l.userid order by l.disbursedon ASC) as DenseRank
          FROM yp_iceberg.yp_loan l WHERE l.state IN(47,71) AND
          DATE_FORMAT(( (l.disbursedon ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') < '{first_day_current_str}') s
          WHERE s.DenseRank=1) a ON b.uid=a.userid
LEFT JOIN (SELECT s.userid as userid,s.productname as Latest_SA_productName, loan_id2,
           DATE_FORMAT(( (s.disbursedon ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') as Latesst_SA_loan_taken_Date,
           DATE_FORMAT(( (s.closedOn ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') as Latest_SA_Closed_Date,
                      DATE_FORMAT(( (s.closedOn ) + interval '5' hour + interval '30' minute),'%Y-%m-%d %H:%i:%s') as closedon_time,
           DATE_FORMAT(( (s.closedOn ) + interval '5' hour + interval '30' minute),'%Y-%m') as Latest_SA_Closed_Month,
           (SELECT val FROM yp_iceberg.yp_key WHERE id=s.state) as Latest_SA_Loan_State
           FROM
           (SELECT l.userid,l.productname,l.disbursedon,l.state,l.closedOn,   l.id AS loan_id2,
           Dense_Rank() over(partition by l.userid order by l.disbursedon DESC) as DenseRank
           FROM yp_iceberg.yp_loan l WHERE l.state IN(47,71) AND
           DATE_FORMAT(( (l.disbursedon ) + interval '5' hour + interval '30' minute),'%Y-%m-%d') < '{first_day_current_str}') s
           WHERE s.DenseRank=1) d ON b.uid=d.userid
LEFT JOIN (SELECT r.userId,r.bandId,r.subBand,r.createdatee FROM (SELECT l.userId,l.bandId, l.createdatee, l.subBand,rank() OVER(PARTITION BY l.userId ORDER BY l.id DESC) as Rank1
           FROM combine_user_trace_mask l
           WHERE l.bandId IS NOT NULL AND l.subBand IS NOT NULL AND
l.createdatee < '{first_day_current_str}'

           ) r
           WHERE Rank1=1) e ON b.uid=e.userId
WHERE rank1=1 and state=53) y
),

-- select * from c1
-- where
-- Product = 'Others'
-- and
-- uid in (165367781
-- )
c2 as (
select Cx_Type, Product, band, DATE_FORMAT(TRY_CAST(createdatee AS TIMESTAMP), '%Y-%m') AS created_month ,
 CASE
        WHEN Closed_month = '{first_day_current_str[:7]}' THEN 'M'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 7) THEN 'M-1'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 6) THEN 'M-2'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 5) THEN 'M-3'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 4) THEN 'M-4'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 3) THEN 'M-5'
        WHEN (YEAR(CAST(Closed_month || '-01' AS DATE)) * 12 + MONTH(CAST(Closed_month || '-01' AS DATE))) = (2025 * 12 + 2) THEN 'M-6'
        ELSE '>M-6'
    END AS Closed_month_bucket,
Closed_month,
loan_id2,
closedon_time,

uid from c1
-- where Product = 'Others'
),

c3 AS (
  SELECT  
   c2.*,    
    xx.userid,
    xx.id AS loan_idd,
    xx.principalDue AS pricipal_disb,
    pmuu.product_name AS product_type_disb,
    CAST(DATE_FORMAT(xx.disbursedon, '%Y-%m-%d') AS DATE) AS disb_date,
    DATE_FORMAT(xx.disbursedon, '%Y-%m') AS disbb_month,
    xx.productname AS disb_productname,
    RANK() OVER (PARTITION BY c2.uid, c2.loan_id2 ORDER BY xx.id ASC) AS rnk,
      RANK() OVER (PARTITION BY xx.userid,  DATE_FORMAT(( (xx.disbursedon ) + INTERVAL '5' HOUR + INTERVAL '30' MINUTE), '%Y-%m') ORDER BY xx.id ASC) AS loan_rank
  FROM c2
  LEFT JOIN (
    yp_iceberg.yp_loan xx
    LEFT JOIN kreditbee_bi_dw_iceberg.yp_product_name_mapping pmuu
      ON xx.disbursedon BETWEEN pmuu.disb_start_date
      AND COALESCE(pmuu.disb_end_date, CURRENT_TIMESTAMP)
      AND (
        (pmuu.subbandid IS NULL AND xx.productname = pmuu.tech_product_name)
        OR
        (pmuu.subbandid = xx.subbandid AND xx.productname = pmuu.tech_product_name)
      )
  )
    ON xx.userid = c2.uid
    AND xx.state IN (47, 71)
    AND DATE_FORMAT((xx.disbursedon ) + INTERVAL '5' HOUR + INTERVAL '30' MINUTE, '%Y-%m-%d %H:%i:%s') > c2.closedon_time
    AND xx.id > c2.loan_id2
)

select
 Cx_Type, Product, band, Closed_month_bucket, disbb_month, product_type_disb, disb_productname,
 Closed_month,
--  loan_rank,
--  rnk,
 disb_date,
 count(loan_idd) as loan_cx,
 count(distinct(uid)) as uidd
from c3
where rnk = 1
group by 1,2,3,4,5,6,7,8,9 ; 

"""
df_inactive_june=load_data(query_fpl_june)

# ##########################################################################################################################
# #################################################### INACTIVE BASE #######################################################
# ##########################################################################################################################

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Copy input DataFrames
df_may = df_inactive_may.copy()
df_june = df_inactive_june.copy()

# ---- FILTER cx_type ----
df_filtered_may = df_may[df_may['Cx_Type'] == 'Inactive_Loan'].copy()
df_filtered_june = df_june[df_june['Cx_Type'] == 'Inactive_Loan'].copy()

# ---- PRODUCT DROPDOWN ----
selected_product = st.selectbox("Select Product", ['FPL-R', 'FPLP-R'])
disb_prod = ""

if(selected_product=="FPL-R"):
    disb_prod="FPL"
elif(selected_product=="FPLP-R"):
    disb_prod="FPLP"

df_filtered_may = df_filtered_may[df_filtered_may['Product'] == selected_product].copy()
df_filtered_june = df_filtered_june[df_filtered_june['Product'] == selected_product].copy()

# ---- Closed Buckets ----
bucket_order = ['M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', '>M-6']

# Step 1: Inactive users per bucket (based on May only for base)
inactive_df_may = df_filtered_may.groupby('Closed_month_bucket')['uidd'].sum().reindex(bucket_order)
print(f"{last_month} inactive count",inactive_df_may)
inactive_df_june = df_filtered_june.groupby('Closed_month_bucket')['uidd'].sum().reindex(bucket_order)
print(f"{ongoing_month} inactive count",inactive_df_june)

# Step 2: Disbursed counts
disbursed_df_may = df_filtered_may[(df_filtered_may['disbb_month']==f"{first_day_prev_str[:7]}") & (df_filtered_may['disb_date'].notna()) & (df_filtered_may['product_type_disb'].isin([disb_prod]))]
disbursed_counts_may = disbursed_df_may.groupby('Closed_month_bucket')['uidd'].sum().reindex(bucket_order)  #############
print(f"{last_month} disbursals : ",disbursed_counts_may)

disbursed_df_june = df_filtered_june[(df_filtered_june['disbb_month']==f"{first_day_current_str[:7]}") & (df_filtered_june['disb_date'].notna()) & (df_filtered_june['product_type_disb'].isin([disb_prod]))]
disbursed_counts_june = disbursed_df_june.groupby('Closed_month_bucket')['uidd'].sum().reindex(bucket_order)    #################
print(f"{ongoing_month} disbursals",disbursed_counts_june)
# Step 3: Disbursal percentages
disbursal_pct_may = (disbursed_counts_may / inactive_df_may) * 100
disbursal_pct_june = (disbursed_counts_june / inactive_df_june) * 100
print(disbursal_pct_may)
print(disbursal_pct_june)


# ---- Final Plot ----
fig = go.Figure()

# Bar: Inactive Users (Base)
fig.add_trace(go.Bar(
    x=bucket_order,
    y=inactive_df_june,
    name=f'Inactive UIDs (Base) {ongoing_month}',
    marker_color='lightgray',
    opacity=0.6
))

fig.add_trace(go.Bar(
    x=bucket_order,
    y=inactive_df_may,
    name=f'Inactive UIDs (Base) {last_month}',
    marker_color='darkgray',
    opacity=0.6
))


# Line: Disbursal % May
fig.add_trace(go.Scatter(
    x=bucket_order,
    y=disbursal_pct_may,
    name=f'{last_month} - Disbursal % ',
    yaxis='y2',
    mode='lines+markers+text',
    text=[f"{v:.1f}%" if v > 0 else "" for v in disbursal_pct_may],
    textposition="top right",
    textfont=dict(color="orange", size=12, family="Arial"),
    line=dict(color='orange', dash='dash')
))

# Line: Disbursal % June
fig.add_trace(go.Scatter(
    x=bucket_order,
    y=disbursal_pct_june,
    name=f'{ongoing_month} - Disbursal %',
    yaxis='y2',
    mode='lines+markers+text',
    text=[f"{v:.1f}%" if v > 0 else "" for v in disbursal_pct_june],
    textposition="bottom left",
    textfont=dict(color="green", size=12, family="Arial"),
    line=dict(color='green', width=2)
))

# ---- Layout ----
fig.update_layout(
    title=f"{selected_product} – Inactive Base vs Disbursed vs % Disbursed ({last_month} & {ongoing_month})",
    barmode='group',
    xaxis_title="Inactivity Bucket",
    yaxis=dict(title=f"Inactive Base - {selected_product}", range=[0, 800000]),
    yaxis2=dict(title="Disbursal %", overlaying='y', side='right', range=[0, 60], tickformat=".1f"),
    plot_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    height=550
)

# ---- Show in Streamlit ----
st.plotly_chart(fig, use_container_width=True)

########################################################################################################################################
################################################################## PART 2 ##############################################################
########################################################################################################################################
# -------------------- DAILY DISBURSAL RATIO PLOT --------------------
import pandas as pd
import plotly.graph_objects as go

# Step 1: Filter data
df_actual_may = df_inactive_may[
    (df_inactive_may['Cx_Type'] == "Inactive_Loan") &
    (df_inactive_may['Product'] == selected_product)
].copy()

df_actual_june = df_inactive_june[
    (df_inactive_june['Cx_Type'] == "Inactive_Loan") &
    (df_inactive_june['Product'] == selected_product)
].copy()

df_disb_may = df_actual_may[df_actual_may['disb_date'].notna()].copy()
df_disb_june = df_actual_june[df_actual_june['disb_date'].notna()].copy()

df_disb_may['disb_date'] = pd.to_datetime(df_disb_may['disb_date'])
df_disb_june['disb_date'] = pd.to_datetime(df_disb_june['disb_date'])

# Step 2: Function to compute daily & cumulative disbursal %
def compute_daily_ratio(df_disb, df_base, target_month):
    days = pd.date_range(f"{target_month}-01", f"{target_month}-{'30' if target_month.endswith('06') else '31'}")
    ratios = []
    cumulative = 0
    total_base = df_base['uidd'].sum()

    for day in days:
        daily_count = df_disb[df_disb['disb_date'] == day]['uidd'].sum()
        ratio = (daily_count / total_base) * 100 if total_base else 0
        cumulative += ratio
        ratios.append((day.day, ratio, cumulative))

    return pd.DataFrame(ratios, columns=['day', 'daily_ratio', 'cumulative_ratio'])

# Step 3: Compute for May and June
may_ratio_df = compute_daily_ratio(df_disb_may, df_actual_may, f"{first_day_prev_str[:7]}")       #####################
june_ratio_df = compute_daily_ratio(df_disb_june, df_actual_june, f"{first_day_current_str[:7]}")        #######################

# Step 4: Plotting
fig_disb_ratio = go.Figure()
# ---- MAY ----
# Area - Cumulative (Y-axis 1)
fig_disb_ratio.add_trace(go.Scatter(
    x=may_ratio_df['day'],
    y=may_ratio_df['cumulative_ratio'],
    name=f'{last_month} – Cumulative',
    mode='lines',
    line=dict(color="#FFE0B2"),
    fill='tozeroy',
    opacity=0.3,
    yaxis='y1',
    showlegend=True
))

# Line - Daily (Y-axis 2)
fig_disb_ratio.add_trace(go.Scatter(
    x=may_ratio_df['day'],
    y=may_ratio_df['daily_ratio'],
    name=f'{last_month} – Daily %',
    mode='lines+markers+text',
    line=dict(color='orange', dash='dash'),
    yaxis='y2',
    text=[f"{v:.1f}%" if d in [1, 5, 10, 15, 20, 25, 30] else "" for d, v in zip(may_ratio_df['day'], may_ratio_df['daily_ratio'])],
    textposition="bottom left",
    textfont=dict(color="#d17008", size=12, family="Arial")

))

# ---- JUNE ----
# Area - Cumulative (Y-axis 1)
fig_disb_ratio.add_trace(go.Scatter(
    x=june_ratio_df['day'],
    y=june_ratio_df['cumulative_ratio'],
    name=f'{ongoing_month} – Cumulative',
    mode='lines',
    line=dict(color="#C8E6C9"),
    fill='tozeroy',
    opacity=0.3,
    yaxis='y1',
    showlegend=True
))

# Line - Daily (Y-axis 2)
fig_disb_ratio.add_trace(go.Scatter(
    x=june_ratio_df['day'],
    y=june_ratio_df['daily_ratio'],
    name=f'{ongoing_month} – Daily %',
    mode='lines+markers+text',
    line=dict(color='green'),
    yaxis='y2',
    text=[f"{v:.1f}%" if d in [1, 5, 10, 15, 20, 25, 30] else "" for d, v in zip(june_ratio_df['day'], june_ratio_df['daily_ratio'])],
    textposition="top right",
    textfont=dict(color="green", size=12, family="Arial")
    
))

# ---- Layout with Dual Y-Axes ----
fig_disb_ratio.update_layout(
    title=f"📈 {ongoing_month} vs {last_month}  :: Daily & Cumulative Disbursal % | {selected_product}",
    xaxis=dict(
        title="Day of Month",
        dtick=1
    ),
    yaxis=dict(
        title="Cumulative %",
        range=[0, 15],  # <-- Update this as needed
        showgrid=True,
        zeroline=True
    ),
    yaxis2=dict(
        title="Daily %",
        overlaying='y',
        side='right',
        range=[0, 1],  # <-- Update this as needed
        showgrid=False
    ),
    plot_bgcolor='white',
    height=500,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Show in Streamlit
st.plotly_chart(fig_disb_ratio, use_container_width=True)


##############################################################################################################################################
########################################################### GET DAILY DISBURSALS ( M /M-1 ) ##################################################
##############################################################################################################################################
# --- Define All Buckets to Plot ---
buckets_to_plot = ['M-1', 'M-2', 'M-3', 'M-4', 'M-5', 'M-6', '>M-6']

# --- Plot Buckets Manually (2 per row) ---
for i in range(0, len(buckets_to_plot), 2):
    col1, col2 = st.columns([5, 5])

    for idx, col in zip([i, i+1], [col1, col2]):
        if idx >= len(buckets_to_plot):
            break

        bucket = buckets_to_plot[idx]

        # ---- Get Data for May and June for this bucket ----
        def get_curve(df, bucket):
            df_bucket = df[df['Closed_month_bucket'] == bucket].copy()
            total_base = df_bucket['uidd'].sum()
            df_disb = df_bucket[df_bucket['disb_date'].notna()]
            df_disb['disb_date'] = pd.to_datetime(df_disb['disb_date'])
            daily = df_disb.groupby(df_disb['disb_date'].dt.day)['uidd'].sum().reindex(range(1, 31), fill_value=0)
            daily_ratio = (daily / total_base * 100).round(2)
            cumulative_ratio = daily_ratio.cumsum().round(2)
            return daily_ratio, cumulative_ratio

        may_daily, may_cumu = get_curve(df_filtered_may, bucket)
        june_daily, june_cumu = get_curve(df_filtered_june, bucket)

        # ---- Plotly Chart ----
        fig = go.Figure()

        # May – Daily %
        fig.add_trace(go.Scatter(
            x=may_daily.index,
            y=may_daily.values,
            name=f"{last_month} – Daily %",
            mode="lines+markers",
            text=[f"{v:.1f}%" if v > 0 else "" for v in may_daily.values],
            textposition="top right",
            textfont=dict(color="black", size=12, family="Arial"),
            line=dict(color="orange", dash="dot", width=2),
            marker=dict(symbol="circle", size=6),
            yaxis="y2"
        ))

        # May – Cumulative %
        fig.add_trace(go.Scatter(
            x=may_cumu.index,
            y=may_cumu.values,
            name=f"{last_month} – Cumulative",
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(255,165,0,0.2)"
        ))

        # June – Daily %
        fig.add_trace(go.Scatter(
            x=june_daily.index,
            y=june_daily.values,
            name=f"{ongoing_month} – Daily %",
            mode="lines+markers",
            text=[f"{v:.1f}%" if v > 0 else "" for v in june_daily.values],
            textposition="top right",
            textfont=dict(color="black", size=12, family="Arial"),
            line=dict(color="green", width=2),
            marker=dict(symbol="square", size=6),
            yaxis="y2"
        ))

        # June – Cumulative %
        fig.add_trace(go.Scatter(
            x=june_cumu.index,
            y=june_cumu.values,
            name=f"{ongoing_month} – Cumulative",
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(0,128,0,0.2)"
        ))

        # Layout
        daily_layout=[7,5,4,3,2,2,1]
        cum_layout = [70,60,40,30,30,30,10]
        fig.update_layout(
            title=f"📊 {bucket} – Daily & Cumulative Disbursal % ({last_month} vs {ongoing_month}) | {selected_product}",
            xaxis=dict(title="Day of Month", dtick=1),
            yaxis=dict(title="Cumulative %", range=[0, cum_layout[i]], showgrid=True, zeroline=True),
            yaxis2=dict(title="Daily %", overlaying="y", side="right", range=[0, daily_layout[i]], showgrid=False),
            plot_bgcolor="white",
            height=450,
            width=900,
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center')
        )

        # Display in Streamlit
        with col:
            st.plotly_chart(fig, use_container_width=True)
