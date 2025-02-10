import numpy as np
import masks
import decode_functions as df

mat=[]
with open('input.txt','r') as file:
    for line in file:
        row = list(map(int, line.strip().split()))
        mat.append(row)
mat=np.array(mat)

n=len(mat)
version = ((n - 21) // 4) + 1

data_mat=masks.get_data_matrix(mat)

format_info = df.extract_format_information(mat)
mask_pattern = df.get_mask_pattern(format_info)
error_correction_level = df.get_error_correction_level(format_info)
error_correction_bits = df.get_error_correction_bits(format_info)

mat = masks.mask(masks.correct_mask(mask_pattern),mat)

data_bits = df.extract_data_bits(mat)

mode = df.get_encoding_mode(data_bits)
count = df.get_character_count(data_bits,mode)
data = df.extract_data(data_bits[4 + count[0]:], mode, count[1])

print(data)
