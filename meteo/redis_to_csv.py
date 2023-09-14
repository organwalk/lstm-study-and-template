import redis
import csv
from datetime import datetime, timedelta

# 连接 Redis 数据库
r = redis.Redis(host='organwalk.ink', port=36379, db=1, password='c209c209')

# 定义起始日期和结束日期
start_date = datetime(2023, 7, 1)
end_date = datetime(2023, 7, 28)

# 遍历日期范围
current_date = start_date
while current_date <= end_date:
    # 根据日期生成文件名和 Redis 键名
    date_str = current_date.strftime("%Y-%m-%d")
    file_name = f"data/1_data_{date_str}.csv"
    redis_key = f"1_data_{date_str}"

    # 从 Redis 中获取数据
    data = r.zrange(redis_key, 0, -1)

    # 创建 CSV 文件并写入数据
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入 CSV 文件的列标题
        writer.writerow(['Time', 'Temperature', 'Humidity', 'Speed', 'Direction', 'Rain', 'Sunlight', 'PM2.5', 'PM10'])

        # 将每行数据写入 CSV 文件
        for item in data:
            values = eval(item)
            writer.writerow(values)

    # 增加一天
    current_date += timedelta(days=1)
