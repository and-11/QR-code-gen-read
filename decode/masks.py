mask_condition = [
    lambda row, column: ((row + column) & 1) == 0,
    lambda row, column: (row & 1) == 0,
    lambda row, column: column % 3 == 0,
    lambda row, column: (row + column) % 3 == 0,
    lambda row, column: (((row >> 1) + (column // 3)) & 1) == 0,
    lambda row, column: ((row * column) & 1) + ((row * column) % 3) == 0,
    lambda row, column: ((((row * column) & 1) + ((row * column) % 3)) & 1) == 0,
    lambda row, column: ((((row + column) & 1) + ((row * column) % 3)) & 1) == 0,
]
    

def get_alignment_positions(version):
    n=21+4*(version-1)

    if version == 1:
        return []

    divs = 2 + version // 7
    total_dist = n - 7 - 6
    divisor = 2 * (divs - 1)
    step = (total_dist + divisor // 2 + 1) // divisor * 2
    coords = [6]
    for i in range(divs - 2, -1, -1):
        coords.append(n - 7 - i * step)
    
    alignment_positions = []
    for row in coords:
        for col in coords:
            if not (row < 9 and col < 9):
                if not (row < 9 and col > n - 9):
                    if not (row > n - 9 and col < 9):
                        alignment_positions.append((row, col))

    return alignment_positions

def get_data_matrix(mat):

    n=len(mat)
    data_mat=[[0 for i in range(n)] for j in range(n)]
    version = ((len(mat) - 21) // 4) + 1

    finder_patterns=[[0,0],[0,-7],[-7,0]]

    for afixe in finder_patterns:
        for i in range(afixe[0],afixe[0]+7):
            for j in range(afixe[1],afixe[1]+7):
                data_mat[i][j]=1
    
    for i in range(9):
        data_mat[i][7]=data_mat[i][8]=data_mat[7][i]=data_mat[8][i]=1
    for i in range(8):
        data_mat[i][-8]=data_mat[7][n-i-1]=data_mat[8][n-i-1]=1
    for i in range(8):
        data_mat[-8][i]=data_mat[n-i-1][7]=data_mat[n-i-1][8]=1
    data_mat[-8][8]=1

    for i in range(n):
        data_mat[6][i]=data_mat[i][6]=1

    if version>=7:
        for i in range(-11,-8):
            for j in range(6):
                data_mat[i][j]=1
        for i in range(6):
            for j in range(-11,-8):
                data_mat[i][j]=1

    allignment_positions=get_alignment_positions(version)

    for pair in allignment_positions:
        x=pair[0]-2
        y=pair[1]-2
        for i in range(x,x+5):
            for j in range(y,y+5):
                data_mat[i][j]=1

    return data_mat

def mask(index,mat):
    n=len(mat)
    data_mat=get_data_matrix(mat)
    aux_mat=[[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if mask_condition[index](i,j) and data_mat[i][j]==0:
                aux_mat[i][j]=1
    mat=mat^aux_mat
    return mat

def correct_mask(k):
    d={
        0:5,1:4,2:7,3:6,4:1,5:0,6:3,7:2
    }
    return d[k]