import pandas as pd
import yfinance as yf
from datetime import datetime

# Lấy ngày hiện tại
end_date = datetime.today().strftime('%Y-%m-%d')

excel_file = "D:\\Quang Bao(LQB)\\NCKH\\Stock_Symbols\\100_Stock_Symbols_Still_Active_1900.xlsx"
df = pd.read_excel(excel_file)

column_data = df.iloc[:, 0]  # Lấy cột đầu tiên của DataFrame

# Chuyển dữ liệu trong cột thành một mảng NumPy
symbols_array = column_data.values

#print(symbols_array)
for symbol in symbols_array:
    stock_data = yf.download(symbol, start="1900-01-01", end=end_date)
    excel_file = excel_file = f"D:\\Quang Bao(LQB)\\NCKH\DATA\\{symbol}_stock_data.xlsx"
    stock_data.to_excel(excel_file)
    