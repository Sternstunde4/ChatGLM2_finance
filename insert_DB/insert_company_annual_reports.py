from sqlalchemy import create_engine, MetaData, Table, String, Column, Integer, text, Float
import pandas as pd
from transfer_to_excel import pad_stock_codes


# DATABASE_URL = "postgresql://dbusername:dbpassword@ip:port/db"
DATABASE_URL = "postgresql://postgres:123456@localhost:5432/postgres"
FILE_PATH = "../data/company_annual_reports.xlsx"

df = pd.read_excel(FILE_PATH)
df["stock_code"] = pad_stock_codes(df["stock_code"])
engine = create_engine(DATABASE_URL)
df.to_sql('company_annual_reports', engine, if_exists='replace', index=False)
engine.dispose()