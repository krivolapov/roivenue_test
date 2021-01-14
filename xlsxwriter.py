# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:00:29 2021

@author: Max
"""

import xlsxwriter


# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('new.xlsx')
worksheet = workbook.add_worksheet()

# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 30)
worksheet.set_column('B:B', 30)

# Insert an image.
worksheet.write('A2', 'Insert an image in a cell:')
worksheet.insert_image('B2', 'fig1.png')

# Insert an image offset in the cell.
worksheet.write('A12', 'Insert an image with an offset:')
worksheet.insert_image('B12', 'python.jpg', {'x_offset': 15, 'y_offset': 10})

# Insert an image with scaling.
worksheet.write('A23', 'Insert a scaled image:')
worksheet.insert_image('B23', 'python.jpg', {'x_scale': 0.5, 'y_scale': 0.5})

workbook.close()