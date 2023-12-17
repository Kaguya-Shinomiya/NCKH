import pandas as pd #đọc dữ liệu
import numpy as np #xử lý dữ liệu
import matplotlib.pyplot as plt #vẽ biểu đồ
from sklearn.preprocessing import MinMaxScaler #chuẩn hóa dữ liệu
from keras.callbacks import ModelCheckpoint #lưu lại huấn luyện tốt nhất

from tensorflow.keras.models import load_model 

#các lớp để xây dựng mô hình
from keras.models import Sequential #đầu vào
from keras.layers import LSTM, GRU #học phụ thuộc
from keras.layers import Dropout #tránh học tủ
from keras.layers import Dense #đầu ra
from keras.optimizers import Adam

#kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score #đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error #đo sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error #đo % sai số tuyệt đối trung bình

import yfinance as yf
from datetime import datetime

# Lấy ngày hiện tại
end_date = datetime.today().strftime('%Y-%m-%d')

# Tên mã chứng khoán bạn quan tâm
stock_symbol = "AAPL"

# Lấy dữ liệu chứng khoán từ Yahoo Finance
#stock_data = yf.download(stock_symbol, start="2010-01-01", end=end_date)
"""stock_data = yf.download(stock_symbol, start="2014-01-01", end=end_date)

# Hiển thị và lưu dữ liệu vào file Excel
excel_file = f"{stock_symbol}_stock_data.xlsx"
stock_data.to_excel(excel_file)
"""
# Đọc dữ liệu từ file excel
excel_file_path = f"D:\\Quang Bao(LQB)\\NCKH\\DATA\\AAPL_stock_data.xlsx"
df = pd.read_excel(excel_file_path)

# Xóa hai dòng "KL" và "Thay đổi %" từ DataFrame df
df = df.drop(columns=["Volume"])

# Hiển thị lại DataFrame sau khi xóa
print(df)

df.shape

df.head()

#xác định kiểu dữ liệu
df.info()

#mô tả bộ dữ liệu
df.describe()

from matplotlib.dates import YearLocator, DateFormatter, MonthLocator # Thêm MonthLocator vào để sửa lỗi

# Chuyển đổi cột "Ngày" sang dạng datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sắp xếp lại dữ liệu theo thứ tự thời gian
df = df.sort_values(by='Date')

# Chuyển đổi định dạng các cột giá thành số thực
df['Close'] = df['Close'].apply(str).str.replace('.', '').astype(float)
df['Open'] = df['Open'].apply(str).str.replace('.', '').astype(float)
df['High'] = df['High'].apply(str).str.replace('.', '').astype(float)
df['Low'] = df['Low'].apply(str).str.replace('.', '').astype(float)

# Lấy thông tin năm từ cột "Ngày"
df['Year'] = df['Date'].dt.year

# Tạo đồ thị giá đóng cửa qua các năm
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label='Adj Close', color='red')
plt.xlabel('Year')
plt.ylabel('Adj Close')
plt.title('Biểu đồ giá đóng cửa của '+ stock_symbol + ' qua các năm')
plt.legend(loc='best')

# Định dạng đồ thị hiển thị các ngày tháng theo năm-tháng
years = YearLocator()
yearsFmt = DateFormatter('%Y')
months = MonthLocator()  # Thêm dòng này để khai báo MonthLocator
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(yearsFmt)
plt.gca().xaxis.set_minor_locator(months)

plt.tight_layout()
plt.show()

df1 = pd.DataFrame(df,columns=['Date','Adj Close'])
df1.index = df1.Date
df1.drop('Date',axis=1,inplace=True)
df1

#chia tập dữ liệu
data = df1.values
train_data = data[:10000]
test_data = data[10000:]

#chuẩn hóa dữ liệu
sc = MinMaxScaler(feature_range=(0,1))
sc_train = sc.fit_transform(data)

#tạo vòng lặp các giá trị
x_train,y_train=[],[]
for i in range(100,len(train_data)):
  x_train.append(sc_train[i-100:i,0]) #lấy 50 giá đóng cửa liên tục
  y_train.append(sc_train[i,0]) #lấy ra giá đóng cửa ngày hôm sau
  
x_train

y_train

#xếp dữ liệu thành 1 mảng 2 chiều
x_train = np.array(x_train)
y_train = np.array(y_train)

#xếp lại dữ liệu thành mảng 1 chiều
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
y_train = np.reshape(y_train,(y_train.shape[0],1))

#xây dựng mô hình
model = Sequential() #tạo lớp mạng cho dữ liệu đầu vào
#2 lớp LSTM
model.add(LSTM(units=256,input_shape=(x_train.shape[1],1),return_sequences=True))
#model.add(LSTM(units=64))
model.add(GRU(units=128, return_sequences=True))
model.add(Dropout(0.5)) #loại bỏ 1 số đơn vị tránh học tủ (overfitting)
model.add(GRU(units=64))  # Có thể thêm thêm lớp GRU hoặc LSTM tùy ý
model.add(Dropout(0.5))
model.add(Dense(1)) #output đầu ra 1 chiều
#đo sai số tuyệt đối trung bình có sử dụng trình tối ưu hóa adam
model.compile(loss='mean_absolute_error',optimizer='adam')

import os.path

file_path = "D:\BaiLam\save_model.hdf5"  # Thay đổi đường dẫn và tên tập tin tùy theo tên thực tế của bạn

if not os.path.exists(file_path):
  #huấn luyện mô hình
  save_model = "save_model.hdf5"
  best_model = ModelCheckpoint(save_model,monitor='loss',verbose=2,save_best_only=True,mode='auto')
  model.fit(x_train,y_train,epochs=150,batch_size=50,verbose=2,callbacks=[best_model])

#dữ liệu train
y_train = sc.inverse_transform(y_train) #giá thực
final_model = load_model("save_model.hdf5")
y_train_predict = final_model.predict(x_train) #dự đoán giá đóng cửa trên tập đã train
y_train_predict = sc.inverse_transform(y_train_predict) #giá dự đoán

#xử lý dữ liệu test
test = df1[len(train_data)-100:].values
test = test.reshape(-1,1)
sc_test = sc.transform(test)

x_test = []
for i in range(100,test.shape[0]):
  x_test.append(sc_test[i-100:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#dữ liệu test
y_test = data[10000:] #giá thực
y_test_predict = final_model.predict(x_test)
y_test_predict = sc.inverse_transform(y_test_predict) #giá dự đoán

#lập biểu đồ so sánh
train_data1 = df1[100:10000]
test_data1 = df1[10000:]

plt.figure(figsize=(24,8))
plt.plot(df1,label='Giá thực tế',color='red') #đường giá thực
train_data1['Dự đoán'] = y_train_predict #thêm dữ liệu
plt.plot(train_data1['Dự đoán'],label='Giá dự đoán train',color='black') #đường giá dự báo train
test_data1['Dự đoán'] = y_test_predict #thêm dữ liệu
plt.plot(test_data1['Dự đoán'],label='Giá dự đoán test',color='blue') #đường giá dự báo test
plt.title('So sánh giá dự báo và giá thực tế') #đặt tên biểu đồ
plt.xlabel('Thời gian') #đặt tên hàm x
plt.ylabel('Giá đóng cửa ($)') #đặt tên hàm y
plt.legend() #chú thích
plt.show()

#r2
print('Độ phù hợp tập train:',r2_score(y_train,y_train_predict))
#mae
print('Sai số tuyệt đối trung bình trên tập train ($):',mean_absolute_error(y_train,y_train_predict))
#mae
print('Phần trăm sai số tuyệt đối trung bình tập train:',mean_absolute_percentage_error(y_train,y_train_predict))

train_data1

#r2
print('Độ phù hợp tập test:',r2_score(y_test,y_test_predict))
#mae
print('Sai số tuyệt đối trung bình trên tập test ($):',mean_absolute_error(y_test,y_test_predict))
#mae
print('Phần trăm sai số tuyệt đối trung bình tập test:',mean_absolute_percentage_error(y_test,y_test_predict))

test_data1

# Lấy ngày kế tiếp sau ngày cuối cùng trong tập dữ liệu để dự đoán
next_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)

# Chuyển đổi ngày kế tiếp sang dạng datetime
next_date = pd.to_datetime(next_date)

# Lấy giá trị của ngày cuối cùng trong tập dữ liệu
next_closing_price = np.array([df['Adj Close'].iloc[-1]])  # Lấy giá trị đóng cửa của ngày cuối cùng

# Chuẩn hóa giá trị của ngày cuối cùng
next_closing_price_normalized = sc.transform(next_closing_price.reshape(-1, 1))  # Chuyển thành mảng 2D

# Tạo dự đoán cho ngày kế tiếp bằng mô hình đã huấn luyện
x_next = np.array([sc_train[-50:, 0]])  # Lấy 50 giá đóng cửa gần nhất
x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1], 1))
y_next_predict = final_model.predict(x_next)
y_next_predict = sc.inverse_transform(y_next_predict)

# Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
df_next = pd.DataFrame({'Date': [next_date], 'Adj Close': [y_next_predict[0][0]]})
df1 = pd.concat([df1, df_next])

# Vẽ biểu đồ mới với dự đoán cho ngày kế tiếp
plt.figure(figsize=(15, 5))
plt.plot(df1['Date'], df1['Adj Close'], label='Giá thực tế', color='red')
plt.plot(train_data1.index, train_data1['Dự đoán'], label='Giá dự đoán train', color='black')
plt.plot(test_data1.index, test_data1['Dự đoán'], label='Giá dự đoán test', color='blue')
plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
plt.xlabel('Thời gian')
plt.ylabel('Giá đóng cửa ($)')
plt.title('So sánh giá dự báo và giá thực tế')
plt.legend()
plt.show()


# Lấy giá trị của ngày cuối cùng trong tập dữ liệu
actual_closing_price = df['Adj Close'].iloc[-1]

# Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
comparison_df = pd.DataFrame({'Ngày': [next_date], 'Giá dự đoán': [y_next_predict[0][0]], 'Giá ngày trước': [actual_closing_price]})

# In ra bảng so sánh
print(comparison_df)