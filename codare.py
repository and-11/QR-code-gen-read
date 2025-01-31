





text = input()
text_codat = []

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

print(text_codat)