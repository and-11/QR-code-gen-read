
import re
import math
import numpy as np
from PIL import Image


#part2

# print(get_encoding_mode('https://www.qrcode.com/'))

LENGTH_BITS = [
    [10, 12, 14],
    [9, 11, 13],
    [8, 16, 16]
]

def get_length_bits(mode, version):
    mode_index = 31 - (mode.bit_length() - 1)
    bits_index = 2 if version > 26 else 1 if version > 9 else 0
    return LENGTH_BITS[mode_index][bits_index]


def get_byte_data(content, length_bits, data_codewords):
    data = np.zeros(data_codewords, dtype=np.uint8)
    right_shift = (4 + length_bits) & 7
    left_shift = 8 - right_shift
    and_mask = (1 << right_shift) - 1
    data_index_start = 2 if length_bits > 12 else 1

    data[0] = 64 + (len(content) >> (length_bits - 4))
    if length_bits > 12:
        data[1] = (len(content) >> right_shift) & 255
    data[data_index_start] = (len(content) & and_mask) << left_shift

    for index in range(len(content)):
        byte = ord(content[index])
        data[index + data_index_start] |= byte >> right_shift
        data[index + data_index_start + 1] = (byte & and_mask) << left_shift

    remaining = data_codewords - len(content) - data_index_start - 1
    for index in range(remaining):
        byte = 17 if index & 1 else 236
        data[index + len(content) + 2] = byte

    return data


#                                                       unul e dif
# print(get_byte_data('https://www.qrcode.com/', 8, 28))


#part3          -_--------------------------------------------------------------------_>>>
LOG = np.zeros(256, dtype=np.uint8)
EXP = np.zeros(256, dtype=np.uint8)

value = 1
for exponent in range(1, 256):
    value = ((value << 1) ^ 285) if value > 127 else (value << 1)
    LOG[value] = exponent % 255
    EXP[exponent % 255] = value


# print(LOG)
# print(EXP)

def mul(a, b):
    return EXP[(int(LOG[a]) + int(LOG[b])) % 255] if a and b else 0

def div(a, b):
    return EXP[(int(LOG[a]) + int(LOG[b]) * 254) % 255]




def poly_mul(poly1, poly2):
    # Allocate result polynomial array with length (len(poly1) + len(poly2) - 1)
    coeffs = np.zeros(len(poly1) + len(poly2) - 1, dtype=np.uint8)

    # Compute coefficients of the result polynomial
    for index in range(len(coeffs)):
        coeff = 0
        for p1_index in range(index + 1):
            p2_index = index - p1_index
            # Ensure indices are within bounds before multiplication
            if p1_index < len(poly1) and p2_index < len(poly2):
                coeff ^= mul(poly1[p1_index], poly2[p2_index])
        coeffs[index] = coeff

    return coeffs


def poly_rest(dividend, divisor):
    quotient_length = len(dividend) - len(divisor) + 1
    # Initialize the remainder as a copy of the dividend
    rest = np.array(dividend, dtype=np.uint8)

    for _ in range(quotient_length):
        # If the first term is 0, skip this iteration
        if rest[0]:
            factor = div(rest[0], divisor[0])
            subtr = np.zeros(len(rest), dtype=np.uint8)
            subtr[:len(divisor)] = poly_mul(divisor, [factor])
            rest = np.bitwise_xor(rest, subtr)[1:]  # XOR and remove first element
        else:
            rest = rest[1:]  # Remove first element

    return rest


def get_generator_poly(degree):
    last_poly = np.array([1], dtype=np.uint8)
    for index in range(degree):
        last_poly = poly_mul(last_poly, np.array([1, EXP[index]], dtype=np.uint8))
    return last_poly

# print(get_generator_poly(16))

def get_edc(data, codewords):
    degree = codewords - len(data)
    message_poly = np.zeros(codewords, dtype=np.uint8)
    message_poly[:len(data)] = data  # Equivalent to `messagePoly.set(data, 0)`
    return poly_rest(message_poly, get_generator_poly(degree))
mes='https://cs.unibuc.ro/~crusu/asc/lectures.html'                                                           # sadkasdasd

# dat = get_byte_data('https://www.qrcode.com/', 8, 28);
# print( get_edc(dat, 44) )

#part 4
def get_size(version):
    return version * 4 + 17

def get_new_matrix(version):
    length = get_size(version)  
    return [[0] * length for _ in range(length)]
#afisare
mat=get_new_matrix(2)


def fill_area(matrix, row, column, width, height, fill=1):
    fill_row = [fill] * width
    for index in range(row, row + height):
        matrix[index][column:column + width] = fill_row

def get_module_sequence(version):
    matrix = get_new_matrix(version)
    size = get_size(version)

    # Finder patterns + divisors
    fill_area(matrix, 0, 0, 9, 9)
    fill_area(matrix, 0, size - 8, 8, 9)
    fill_area(matrix, size - 8, 0, 9, 8)
    # Alignment pattern - yes, we just place one. For the general
    # implementation, wait for the next parts in the series!
    fill_area(matrix, size - 9, size - 9, 5, 5)
    # Timing patterns
    fill_area(matrix, 6, 9, version * 4, 1)
    fill_area(matrix, 9, 6, 1, version * 4)
    # Dark module
    matrix[size - 8][8] = 1

    row_step = -1
    row = size - 1
    column = size - 1
    sequence = []
    index = 0
    while column >= 0:
        if matrix[row][column] == 0:
            sequence.append((row, column))
        # Checking the parity of the index of the current module
        if index & 1:
            row += row_step
            if row == -1 or row == size:
                row_step = -row_step
                row += row_step
                column -= 2 if column == 7 else 1
            else:
                column += 1
        else:
            column -= 1
        index += 1
    return sequence

print(len(get_module_sequence(3)))
# x= get_module_sequence(2)
# print( type(x[1][1]) )


# print( get_module_sequence(2) )

def get_raw_qr_code(message):
    # One day, we'll compute these values. But not today!
    VERSION = 3
    TOTAL_CODEWORDS =70
    LENGTH_BITS = 8
    DATA_CODEWORDS = 55

    codewords = [0]*TOTAL_CODEWORDS                                         # dif dar pare ok
    byte_data = get_byte_data(message, LENGTH_BITS, DATA_CODEWORDS)
    codewords[:len(byte_data)] = byte_data
    codewords[DATA_CODEWORDS:] = get_edc(byte_data, TOTAL_CODEWORDS)

    size = get_size(VERSION)
    qr_code = get_new_matrix(VERSION)
    module_sequence = get_module_sequence(VERSION)

    # Placing the fixed patterns
    # Finder patterns
    for row, col in [[0, 0], [size - 7, 0], [0, size - 7]]:
        fill_area(qr_code, row, col, 7, 7)
        fill_area(qr_code, row + 1, col + 1, 5, 5, 0)
        fill_area(qr_code, row + 2, col + 2, 3, 3)

    # Separators
    fill_area(qr_code, 7, 0, 8, 1, 0)
    fill_area(qr_code, 0, 7, 1, 7, 0)
    fill_area(qr_code, size - 8, 0, 8, 1, 0)
    fill_area(qr_code, 0, size - 8, 1, 7, 0)
    fill_area(qr_code, 7, size - 8, 8, 1, 0)
    fill_area(qr_code, size - 7, 7, 1, 7, 0)

    # Alignment pattern
    fill_area(qr_code, size - 9, size - 9, 5, 5)
    fill_area(qr_code, size - 8, size - 8, 3, 3, 0)
    qr_code[size - 7][size - 7] = 1

    # Timing patterns
    for pos in range(8, VERSION * 4 + 8+1, 2):
        qr_code[6][pos] = 1
        qr_code[6][pos + 1] = 0
        qr_code[pos][6] = 1
        qr_code[pos + 1][6] = 0
    qr_code[6][size - 7] = 1
    qr_code[size - 7][6] = 1

    # Dark module
    qr_code[size - 8][8] = 1

    # Placing message and error data
    index = 0
    for codeword in codewords:
        # Counting down from the leftmost bit
        for shift in range(7, -1, -1):
            bit = (codeword >> shift) & 1
            row, column = module_sequence[index]
            index += 1
            qr_code[row][column] = bit

    return qr_code


mat=get_raw_qr_code(mes)
                                #                           da skip cum trebuie la formaturi

#part 5

# MASK_FNS = [
#     lambda row, column: ((row + column) & 1) == 0,
#     lambda row, column: (row & 1) == 0,
#     lambda row, column: column % 3 == 0,
#     lambda row, column: (row + column) % 3 == 0,
#     lambda row, column: (((row >> 1) + (column // 3)) & 1) == 0,
#     lambda row, column: ((row * column) & 1) + ((row * column) % 3) == 0,
#     lambda row, column: ((((row * column) & 1) + ((row * column) % 3)) & 1) == 0,
#     lambda row, column: ((((row + column) & 1) + ((row * column) % 3)) & 1) == 0,
# ]


def inside( x,seq):        
    for y in seq:
        if y == x :
            return 0
    return 1

                        #a asjknda sjkdasj dnaskjdn askjdn asjkndask jdjkas dnasj dkasjkd askjd as
                #a asjknda sjkdasj dnaskjdn askjdn asjkndask jdjkas dnasj dkasjkd askjd as
                            #a asjknda sjkdasj dnaskjdn askjdn asjkndask jdjkas dnasj dkasjkd askjd as


def first_penalty (matrix): 
    sum = 0
    for i in range (len(matrix)): 
        ct =1
        for j in range (1, len(matrix)):
            if matrix[i][j] == matrix[i][j - 1]:
                ct+=1
            elif ct >= 5:
                sum+=ct
                ct= 1
    for i in range (len(matrix)): 
        ct =1
        for j in range (1, len(matrix)):
            if matrix[i][j] == matrix[i - 1][j]:
                ct+=1
            elif ct >= 5:
                sum+=ct
                ct= 1
    return sum

def second_penalty (matrix):
    sum= 0
    for i in range (len(matrix) - 1):
        for j in range (len(matrix) - 1):
            if matrix[i][j] == matrix[i][j + 1] and matrix[i + 1][j] == matrix[i][j] and matrix[i + 1][j + 1] == matrix[i][j]:
                sum += 2
    return sum

def third_penalty (matrix):
    nrpat=0
    pat=[1,0,1,1,1,0,1,0,0,0,0]
    patrev=[0,0,0,0,1,0,1,1,1,0,1]
    for i in range (len(matrix) - 1):
        for j in range (len(matrix) - 12):
            if matrix[i][j:j+10]==pat or matrix[i][j:j+10]==patrev:
                nrpat=nrpat+1
    for j in range (len(matrix) - 1):
        for i in range (len(matrix) - 12):
                    col_seg = [matrix[x][j] for x in range(i, i+11)]
                    if col_seg==pat or col_seg==patrev:
                        nrpat=nrpat+1

    
    return nrpat*40


def fourth_penalty (matrix):
    white=0
    black=0
    for i in range (len(matrix) - 1):
        for j in range (len(matrix) - 1):
            if matrix[i][j]==1:
                black=black+1
            else:
                white=white+1
                
                
    total=white+black
    darkpercent=(total//black)*100
    nextfive=(darkpercent-darkpercent%5+5)/5
    lastfive=(darkpercent-darkpercent%5)/5
    return min(lastfive,nextfive)*10


def calculate_penalty(matrix):
    return first_penalty(matrix) + second_penalty(matrix) + third_penalty(matrix) + fourth_penalty(matrix)



                            #a asjknda sjkdasj dnaskjdn askjdn asjkndask jdjkas dnasj dkasjkd askjd as
                #a asjknda sjkdasj dnaskjdn askjdn asjkndask jdjkas dnasj dkasjkd askjd as
                            #a asjknda sjkdasj dnaskjdn askjdn asjkndask jdjkas dnasj dkasjkd askjd as


def get_masked_matrix(version, codewords, error_level):

#   1
    matrix = get_masked_matrix_1(version, codewords)
    place_format_modules(matrix, error_level, 0)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    final_matrix = matrix
    final_penalty = penalty


   

#   2
    matrix = get_masked_matrix_2(version, codewords)
    place_format_modules(matrix, error_level,1)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix
#   3
    matrix = get_masked_matrix_3(version, codewords)
    place_format_modules(matrix, error_level, 2)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix

#   4
    matrix = get_masked_matrix_4(version, codewords)
    place_format_modules(matrix, error_level,3)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix

#   5
    matrix = get_masked_matrix_5(version, codewords)
    place_format_modules(matrix, error_level, 4)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix

#   6
    matrix = get_masked_matrix_6(version, codewords)
    place_format_modules(matrix, error_level, 5)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix

#   7
    matrix = get_masked_matrix_7(version, codewords)
    place_format_modules(matrix, error_level, 6)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix

#   8
    matrix = get_masked_matrix_8(version, codewords)
    place_format_modules(matrix, error_level, 7)
    place_fixed_patterns(matrix)
    penalty = calculate_penalty(matrix)

    if penalty < final_penalty:
        final_penalty = penalty
        final_matrix = matrix

    return final_matrix
            # ajsdbj asdjhas hdasb djhasbd hjasbd hjasbd hjasbdhj abjhdbas hjbda sbasjh bhjasdb hjsab asd hjasbdhjas dasjh



def get_masked_matrix_1(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if (i + j) % 2 == 0 :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_2(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if i % 2 == 0   :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_3(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if j % 3 == 0 :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_4(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if (i + j) % 3 == 0  :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_5(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if (i // 2 + j // 3) % 2 == 0  :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_6(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if ((i * j) %2 == 0 + (i * j) % 3 == 0) == 0  :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_7(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if ((i * j) % 2 == 0 + (i * j) % 3 == 0) % 2 == 0   :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix
def get_masked_matrix_8(version, codewords):
    sequence = get_module_sequence(version)
    matrix = get_new_matrix(version)
    for i in range (len (matrix)):
        for j in range (len (matrix)):
            matrix[i][j] = mat[i][j]
            if inside( (i,j), sequence) == 0:
                if ((i + j) % 2 + (i * j) % 3 == 0) % 2 == 0  :
                    matrix[i][j] = mat[i][j]^1
                    # print()
    return matrix



#asjkdasjd asdhjsa dhjasd hjabsd asjhdashj bdashjbd hjasbd jasbd hjsabdjhbas jhdbashjbd hjasbd ashj


EDC_ORDER = 'MLHQ'
FORMAT_DIVISOR = bytearray([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
FORMAT_MASK = bytearray([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])


def get_format_modules(error_level, mask_index):
    format_poly = [0]*15
    error_level_index = EDC_ORDER.index(error_level)
    format_poly[0] = error_level_index >> 1
    format_poly[1] = error_level_index & 1
    format_poly[2] = mask_index >> 2
    format_poly[3] = (mask_index >> 1) & 1
    format_poly[4] = mask_index & 1
    rest = poly_rest(format_poly, FORMAT_DIVISOR)
    format_poly[5:5 + len(rest)] = rest
    masked_format_poly = bytearray(bit ^ FORMAT_MASK[index] for index, bit in enumerate(format_poly))
    return masked_format_poly
def place_fixed_patterns(matrix):
    size = len(matrix)
    
    # Finder patterns
    for row, col in [[0, 0], [size - 7, 0], [0, size - 7]]:
        fill_area(matrix, row, col, 7, 7)
        fill_area(matrix, row + 1, col + 1, 5, 5, 0)
        fill_area(matrix, row + 2, col + 2, 3, 3)
    
    # Separators
    fill_area(matrix, 7, 0, 8, 1, 0)
    fill_area(matrix, 0, 7, 1, 7, 0)
    fill_area(matrix, size - 8, 0, 8, 1, 0)
    fill_area(matrix, 0, size - 8, 1, 7, 0)
    fill_area(matrix, 7, size - 8, 8, 1, 0)
    fill_area(matrix, size - 7, 7, 1, 7, 0)
    
    # Alignment pattern
    fill_area(matrix, size - 9, size - 9, 5, 5)
    fill_area(matrix, size - 8, size - 8, 3, 3, 0)
    matrix[size - 7][size - 7] = 1
    
    # Timing patterns
    for pos in range(8, size - 9+1, 2):
        matrix[6][pos] = 1
        matrix[6][pos + 1] = 0
        matrix[pos][6] = 1
        matrix[pos + 1][6] = 0
    
    matrix[6][size - 7] = 1
    matrix[size - 7][6] = 1
    
    # Dark module
    matrix[size - 8][8] = 1
def place_format_modules(matrix, error_level, mask_index):
    format_modules = get_format_modules(error_level, mask_index)
    matrix[8][:6] = format_modules[:6]
    matrix[8][7:8] = format_modules[6:8]
    matrix[8][len(matrix) - 8:] = format_modules[7:]
    matrix[7][8] = format_modules[8]
    for index, cell in enumerate(format_modules[:7]):
        matrix[len(matrix) - index - 1][8] = cell
    for index, cell in enumerate(format_modules[9:]):
        matrix[5 - index][8] = cell
        
def get_masked_qr_code(version, codewords, error_level):
    matrix = get_masked_matrix(version, codewords, error_level)
    # place_format_modules(matrix, error_level, mask_index)
    # place_fixed_patterns(matrix)
    return matrix

def get_raw_qr_code1(message):
    # One day, we'll compute these values. But not today!
    VERSION = 3
    TOTAL_CODEWORDS = 70
    LENGTH_BITS = 8
    DATA_CODEWORDS = 55

    codewords = [0]*TOTAL_CODEWORDS
    byte_data = get_byte_data(message, LENGTH_BITS, DATA_CODEWORDS)
    codewords[:len(byte_data)] = byte_data
    codewords[DATA_CODEWORDS:] = get_edc(byte_data, TOTAL_CODEWORDS)

    size = get_size(VERSION)
    qr_code = get_new_matrix(VERSION)
    module_sequence = get_module_sequence(VERSION)
    
    qr_code=get_masked_qr_code(VERSION,codewords,'L')
    return qr_code
    
mat=get_raw_qr_code1(mes)


def matrix_to_qrcode(matrix, scale=10, output_file='qrcode.png'):
    """
    Converts a 0/1 matrix to a QR code-like image.

    :param matrix: 2D list containing 0s and 1s
    :param scale: Size of each pixel block
    :param output_file: Name of the output image file
    """

    matrix.insert(0, [0 for _ in range(len(matrix[0]) + 2)])
    matrix.append ([0 for _ in range (len(matrix[0]))])
    for i in range (1, len(matrix) - 1):
         matrix[i].insert(0, 0)
         matrix[i].append (0)

    if not matrix:
        raise ValueError("The matrix is empty.")

    height = len(matrix)
    width = len(matrix[0])

    # Create a new image with white background
    img = Image.new('RGB', (width * scale, height * scale), 'white')
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            color = (0, 0, 0) if matrix[y][x] == 1 else (255, 255, 255)
            for i in range(scale):
                for j in range(scale):
                    pixels[x * scale + i, y * scale + j] = color

    img.save(output_file)
    print(f"QR code image saved as {output_file}")
    
    #           #    #           #    #           #    #           #    #           #    #           #    #           #    #           #

matrix_to_qrcode(mat) 