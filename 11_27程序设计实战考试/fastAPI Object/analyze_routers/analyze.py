from fastapi import APIRouter, File, UploadFile
from analyze_tools.analyze_tool import analyze_salary, analyze_company_count
import io

from fastapi.responses import FileResponse, StreamingResponse

analyze_router = APIRouter()


@analyze_router.post("/salary_by_district")
def salary_by_district(csv_file: UploadFile = File()):
    """
        编写一个 FastAPI 接口 /analyze/salary_by_district

        要求：
        1. 前端上传 CSV（字段至少包含 district, salary）
        2. 后端使用 Pandas 计算：
           - 各地区 salary 的平均值
           - 各地区 salary 的最小值
           - 各地区 salary 的最大值
        3. 将统计结果写入 Excel 文件
        4. 将 Excel 文件返回给前端下载
    """
    # 以字节流传回文件
    csv_data = io.BytesIO(csv_file.file.read())

    return StreamingResponse(
        analyze_salary(csv_file.filename, csv_data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


@analyze_router.post("/company_count_chart")
def company_count_chart(csv_file: UploadFile = File()):
    """
        FastAPI 接口 /analyze/company_count_chart

        要求：
        1. 前端上传 CSV（字段至少包含 district, company）
        2. 后端使用 Pandas 对各地区 company 数量进行统计
        3. 使用 matplotlib 绘制柱状图
        4. 将图保存为 .png 文件
        5. 返回 .png 文件给前端下载
    """
    # 以字节流传回文件
    csv_data = io.BytesIO(csv_file.file.read())

    return StreamingResponse(
        analyze_company_count(csv_data),
        media_type="image/png",
    )
