import pandas as pd
import yfinance as yf


from datetime import datetime

# Lấy ngày hiện tại
end_date = datetime.today().strftime('%Y-%m-%d')

# Đường dẫn đến tệp Excel
excel_file_path = 'D:\\Quang Bao(LQB)\\NCKH\\Stock_Symbols\\Stock_Symbol.xlsx'

# Tên cột bạn muốn đọc
column_name = 'Ticker'

# Sử dụng pandas để đọc dữ liệu từ cột
df = pd.read_excel(excel_file_path)

# Lấy dữ liệu từ cột cụ thể
column_data = df[column_name]

# In ra dữ liệu
#print(column_data[:100])

my_list = []

for i in column_data[:]:
    print(i)
    stock_data = yf.download(i, start="1900-01-01", end=end_date)
    if not stock_data.empty:
        my_list.append(i)
        if len(my_list) == 100:
            break

print(my_list)

import pandas as pd
# Tạo DataFrame từ mảng
df = pd.DataFrame(my_list, columns=['Column_Name'])

# Lưu vào tệp Excel
df.to_excel('D:\\Quang Bao(LQB)\\NCKH\\Stock_Symbols\\100_Stock_Symbols_Still_Active_1900.xlsx', index=False, header=False)
"""stock_symbol = "AAPL"

# Lấy thông tin về cổ phiếu
stock_info = yf.Ticker(stock_symbol).get_info()

# In ra năm phát hành cổ phiếu
ipo_year = stock_info.get("ipoYear")
print(f"{stock_symbol} phát hành cổ phiếu vào năm {ipo_year}")"""