import pandas as pd
import pyarrow
import s3fs
import numpy as np

import sagemaker
import boto3
from sagemaker.session import Session
from sagemaker import get_execution_role

# Initialize the SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()  # This is the IAM role that SageMaker would assume

# Specify your bucket and prefix on S3 where the data and model artifacts would be stored
s3_path = 's3://<nombre-Bucket>/datasets2/<nombre-fichero>.parqiet' #o .csv
df=pd.read_parquet(s3_path,engine='pyarrow') #quizá no sea necesario especificar el motor pyarrow

# o si es .csv
df=pd.read_csv(s3_path,engine='pyarrow') #quizá no sea necesario especificar el motor pyarrow
