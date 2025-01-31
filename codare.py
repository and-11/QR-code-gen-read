from PIL import Image

text_codat = []
matrice = []
N = 31      # marime 29 +2 bordare





def citire_binar():
    global text_codat
    text = input()
    for ch in text:
        cod_ascii = ord(ch) 
        aux = ""
        for i in range(0,8):            #vrem 8 biti
            if cod_ascii%2 == 0:
                aux = "0"+aux
            else :
                aux = "1"+aux
            cod_ascii = cod_ascii // 2
        text_codat.append(aux)

def initializare():
    global N
    # N = 3
    matrice = [ [ 0 for i in range(0,N) ] for j in range(0,N) ]
    print( matrice )

def Generare_matrice():
    
    return

def Afisare():

    for i in range(0,19):
        matrice[3][i]=1


def png_to_binary_matrix(file_path, target_size=29):
    """
    Converts a QR code PNG image into a binary matrix of 1s (black) and 0s (white),
    resizing it to target_size x target_size before binarization.

    Args:
        file_path (str): Path to the QR code PNG file.
        target_size (int): Desired size of the binary matrix (e.g., 29 for Version 3 QR codes).

    Returns:
        list of lists: 2D binary matrix representing the QR code.
    """
    # Step 1: Open the image and convert to grayscale
    img = Image.open(file_path).convert("L")  # 'L' mode is for grayscale

    # Optional: Ensure the image is square by cropping the center square
    width, height = img.size
    if width != height:
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img = img.crop((left, top, right, bottom))
        print(f"Image cropped to square: {img.size}")

    # Step 2: Resize the image to target_size x target_size using nearest neighbor

    img_resized = img.resize((target_size, target_size), Image.NEAREST)
    print(f"Image resized to: {img_resized.size}")

    # Step 3: Convert to binary (1 = black, 0 = white) using threshold
    # Here, we use a lambda function to apply the threshold
    binary_img = img_resized.point(lambda x: 1 if x < 128 else 0, mode='1')

    # Step 4: Convert the binary image to a binary matrix (list of lists)
    binary_matrix = []
    for y in range(target_size):
        row = []
        for x in range(target_size):
            pixel = binary_img.getpixel((x, y))
            row.append(pixel)
        binary_matrix.append(row)

    return binary_matrix


# Path to your QR code PNG file
file_path = "qr40.png"

# Convert the QR code image to a 29x29 binary matrix
binary_matrix = png_to_binary_matrix(file_path, target_size=177)



def matrix_to_qrcode(matrix, scale=10, output_file='qrcode.png'):
    """
    Converts a 0/1 matrix to a QR code-like image.

    :param matrix: 2D list containing 0s and 1s
    :param scale: Size of each pixel block
    :param output_file: Name of the output image file
    """

    # matrix.insert(0, [0 for _ in range(len(matrix[0]) + 2)])
    # matrix.append ([0 for _ in range (len(matrix[0]))])
    # for i in range (1, len(matrix) - 1):
    #     matrix[i].insert(0, 0)
    #     matrix[i].append (0)

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

# Example us

# matrix_to_qrcode(binary_matrix, scale=20, output_file='qrcode.png')




citire_binar()
initializare()
Afisare()