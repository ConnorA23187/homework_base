import io
import pandas as pd
from matplotlib import pyplot as plt

def analyze_salary(csv_file_name, csv_file):
    data = pd.read_csv(csv_file)

    # 分组并计算
    result = data.groupby("district")["salary"].agg(["mean", "min", "max"])

    buffer = io.BytesIO()

    # 将统计结果写入 Excel 文件并转换为字节流
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="salary_analyze", index=True)

    buffer.seek(0)

    return buffer

def analyze_company_count(csv_file):
    data = pd.read_csv(csv_file)

    # 分组计数
    result = data.groupby("district")["positionId"].count()

    # 绘图
    plt.bar(result.index, result.values)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("company_count")
    plt.xlabel("company")
    plt.ylabel("count")

    # 图表写入字节流
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return buffer
