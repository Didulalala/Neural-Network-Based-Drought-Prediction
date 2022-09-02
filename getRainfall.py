import netCDF4 as nc
import xlwings as xw
import numpy as np
import os
from os import path

xl_filepath = r"c:\Users\D.Haoliang\Desktop\Rainfall.xlsx"
Rainfall = []

# 长江中下游平原北纬25-35度，东经108-120度
url = r"C:\Users\D.Haoliang\Downloads\dataset-satellite-precipitation-fc0ecf0e-2a7c-4c24-b607-3cef9543ac3d"
#遍历当前路径下所有文件
file  = os.listdir(url)
for f in file:
    #字符串拼接
    print(f)
    nc_filepath = path.join (url , f)
    nc_file = nc.Dataset(nc_filepath)
    precip_values = np.squeeze(nc_file.variables["precip"][:])
    print(precip_values.shape)
    Rainfall.append(sum(map(sum, precip_values))*30 / 20)

xl_file = xw.Book(xl_filepath)
sht = xl_file.sheets('sheet1')
sht.range('A1').options(transpose=True).value=Rainfall
xl_file.save()