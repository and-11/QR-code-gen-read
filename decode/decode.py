import numpy as np
import masks
import decode_functions as df
from PIL import Image

def png_to_binary_matrix(file_path, target_size=29):
    img = Image.open(file_path).convert("L")
    width, height = img.size
    if width != height:
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img = img.crop((left, top, right, bottom))

    img_resized = img.resize((target_size, target_size), Image.NEAREST)

    binary_img = img_resized.point(lambda x: 1 if x < 128 else 0, mode='1')

    binary_matrix = []
    for y in range(target_size):
        row = []
        for x in range(target_size):
            pixel = binary_img.getpixel((x, y))
            row.append(pixel)
        binary_matrix.append(row)
    return binary_matrix

def display(mat):
    scale = 20
    scaled_matrix = np.kron(mat, np.ones((scale, scale)))

    inverted_matrix=1-scaled_matrix

    image = Image.fromarray(inverted_matrix * 255)
    image.show()

def debordare(mat):
    def debord1(amat):
        amat.pop(0)
        amat.pop(-1)
        bmat=[]
        for line in amat:
            line.pop(0)
            line.pop(-1)
            bmat.append(line)
        return bmat
    while mat[0][0]==0:
        mat=debord1(mat)
    return mat

imgpath=input('File name: ')
mat=png_to_binary_matrix(imgpath,target_size=31)
display(mat)
mat=debordare(mat)

# mat=[]
# with open('input.txt','r') as file:
#     for line in file:
#         row = list(map(int, line.strip().split()))
#         mat.append(row)
# mat=np.array(mat)

# n=len(mat)
# version = ((n - 21) // 4) + 1

mat=np.array(mat)

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
