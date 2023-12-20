import yfinance as yf
import pandas as pd
from datetime import datetime

# Lấy ngày hiện tại
end_date = datetime.today().strftime('%Y-%m-%d')

# Tên mã chứng khoán bạn quan tâm
stock_symbol = "AAPL"

# Lấy dữ liệu chứng khoán từ Yahoo Finance
#stock_data = yf.download(stock_symbol, start="1981-01-01", end="1981-02-01")
stock_data = yf.download(stock_symbol, start="1981-01-01", end=end_date)
#stock_data = yf.download(stock_symbol, start="2014-01-01", end=end_date)

# Hiển thị và lưu dữ liệu vào file Excel
excel_file = f"D:\\Quang Bao(LQB)\\NCKH\DATA\\{stock_symbol}_stock_data.xlsx"
stock_data.to_excel(excel_file)

# Hiển thị thông báo khi hoàn tất
print(f"Dữ liệu chứng khoán đã được lưu vào file Excel: {excel_file}")
