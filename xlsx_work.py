# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:10:18 2021

@author: Max
"""

import xlsxwriter



# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('output\scatter_plot.xlsx')
worksheet = workbook.add_worksheet('First')

bold = workbook.add_format({'bold': True})

# Widen the first column to make the text clearer.
worksheet.set_column('A:A', 30)
worksheet.set_column('B:B', 20)
worksheet.set_column('C:C', 40)

worksheet.write('A1', 'Item', bold)
worksheet.write('B1', 'Item', bold)
worksheet.write('C1', 'Chart', bold)

# Insert an image.
worksheet.write('A'+str(2), 'Insert an image in a cell:')
worksheet.insert_image('C2', 'Image with annotations.jpg', {'x_scale': 0.5, 'y_scale': 0.5})

# Insert an image offset in the cell.
worksheet.write('A12', 'Insert an image with an offset:')
worksheet.insert_image('C12', 'Image with annotations.jpg', {'x_scale': 0.5, 'y_scale': 0.5})

# Insert an image with scaling.
worksheet.write('A22', 'Insert a scaled image:')
worksheet.insert_image('C22', 'Image with annotations.jpg', {'x_scale': 0.5, 'y_scale': 0.5})

workbook.close()