import pandas as pd
from datetime import datetime

filename = 'data.xlsx'
sheet_name = 'Dados anemo'

dataset = pd.read_excel(
  filename,
  sheet_name=sheet_name,
  index_col=0,
)

dataset.columns = ['day', 'month', 'year', 'hour', 'v_anemo', 'dir', 'temp', 'moisture', 'pressure']
dataset.to_csv('dataset.csv')