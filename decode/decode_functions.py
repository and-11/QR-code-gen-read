import masks

def extract_format_information(matrix):

    n = len(matrix)
    format_info = []

    for x in range(8):
        if x != 6:
            format_info.append(matrix[8][x])

    for y in range(7, -1, -1):
        if y != 6:
            format_info.append(matrix[y][8])

    return format_info

def get_mask_pattern(format_info):

    mask_pattern_bits = format_info[2:5]
    mask_pattern = 0
    for bit in mask_pattern_bits:
        mask_pattern = (mask_pattern << 1) | bit
    return mask_pattern

def get_error_correction_level(format_info):

    error_correction_level_bits = format_info[:2]
    error_correction_level = 0
    for bit in error_correction_level_bits:
        error_correction_level = (error_correction_level << 1) | bit
    return error_correction_level

def get_error_correction_bits(format_info):

    error_correction_bits_bits = format_info[5:]
    error_correction_bits = 0
    for bit in error_correction_bits_bits:
        error_correction_bits = (error_correction_bits << 1) | bit
    return error_correction_bits

def get_encoding_mode(data_bits):

    mode_indicator = data_bits[:4]
    if mode_indicator == [0, 0, 0, 1]:
        return "Numeric"
    elif mode_indicator == [0, 0, 1, 0]:
        return "Alphanumeric"
    elif mode_indicator == [0, 1, 0, 0]:
        return "Byte"
    elif mode_indicator == [1, 0, 0, 0]:
        return "Kanji"
    else:
        return "Unknown"

def extract_data_bits(matrix):

    data_mat=masks.get_data_matrix(matrix)
    n = len(matrix)

    row_step = -1
    row = n - 1
    column = n - 1
    sequence = []
    index = 0
    
    while column >= 0:
        if data_mat[row][column] == 0:
            sequence.append(matrix[row][column])

        if index & 1:
            row += row_step
            if row == -1 or row == n:
                row_step = -row_step
                row += row_step
                column -= 2 if column == 7 else 1
            else:
                column += 1
        else:
            column -= 1
        index += 1

    return sequence

def get_character_count(data_bits, mode):
    if mode == "Numeric":
        count_bits = 10
    elif mode == "Alphanumeric":
        count_bits = 9
    elif mode == "Byte":
        count_bits = 8
    elif mode == "Kanji":
        count_bits = 8
    else:
        count_bits = 8

    count = 0
    for bit in data_bits[4:4 + count_bits]:
        count = (count << 1) | bit
    return [count_bits,count]

def extract_data(data_bits, mode, count):

    if mode == "Numeric":
        data = ""
        for i in range(0, count * 10, 10):
            chunk = data_bits[i:i + 10]
            value = 0
            for bit in chunk:
                value = (value << 1) | bit
            data += f"{value:03}"
        return data
    elif mode == "Byte":
        data = ""
        for i in range(0, count * 8, 8):
            chunk = data_bits[i:i + 8]
            value = 0
            for bit in chunk:
                value = (value << 1) | bit
            data += chr(value)
        return data
    elif mode == "Alphanumeric":
        data = ""
        alphanumeric_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
        i = 0
        while i < count * 11:
            chunk = data_bits[i:i + 11]
            value = 0
            for bit in chunk:
                value = (value << 1) | bit
            i += 11

            char1 = value // 45
            char2 = value % 45
            data += alphanumeric_table[char1]
            if i < count * 11:
                data += alphanumeric_table[char2]
        return data
