import redis
import csv

# 连接 Redis 数据库
r = redis.Redis(host='organwalk.ink', port=36379, db=1, password='c209c209')

# 获取有序集合中的所有元素
data = r.zrange('1_data_2023-06-27', 0, -1)

# 创建 CSV 文件并写入数据
with open('data/1_data_2023-06-27.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # 写入 CSV 文件的列标题
    writer.writerow(['Time', 'Temperature', 'Humidity', 'Speed', 'Direction', 'Rain', 'Sunlight', 'PM2.5', 'PM10'])

    # 将每行数据写入 CSV 文件
    for item in data:
        values = eval(item)
        writer.writerow(values)
