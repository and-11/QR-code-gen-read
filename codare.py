from PIL import Image
 
text_codat = []
matrice = []
N = 29
 
 
 
 
 
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
    global N,matrice
    # N = 3
    matrice = [ [ 0 for i in range(0,N) ] for j in range(0,N) ]
    #print( matrice )
 
def Generare_matrice():
 
    return
 
def Afisare():
 
    for i in range(0,19):
        matrice[3][i]=1
        matrice[5][i]=1
        matrice[25][i]=1
        matrice[i][6]=1
        matrice[0][i]=1
    matrice[20][20]=1
 
 
 
 
 
 
 
 
 
 
 
 
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
 
# Example us
 
# matrix_to_qrcode(binary_matrix, scale=20, output_file='qrcode.png')
 
 
 
#mainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
#citire_binar()
initializare()
Afisare()
matrix_to_qrcode(matrice)

