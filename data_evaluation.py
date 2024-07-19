# Where I evaluate the dataset by itself
# without the model.

# 8/9/23


# imports dump
import pandas as pd
from pandas_profiling import ProfileReport

# Read the data
data = pd.read_csv("cancer.csv")

# Create the ProfileReport
profile = ProfileReport(data, title="Lung Cancer")
profile.to_file("cancer_report.html")