Codare:

    fisierul cere un text ce urmeaza a fi codat intr-un cod qr si il salveaza in fisierul qrcode.png

    pasi:

    citim un text de la tastaura

    Luam textul si il convertim intr-un binar avand grija sa respectam convetiile unui cod qr

    Folosim Reed Solomon pentru a genera si partea de error corection

    Punem mesajul nostru intr-o matrice echivalenta cu viitorul nostru cod qr si ii adaugam elementele standar ale unui cod qr

    Aplicam cele 8 masti posibile pe matricea noastra si o alegem pe cea cu scor de penalizare cat mai mic



Decodare:

    Decodeaza coduri qr cu versiunea intre 1 si 3, date ca o matrice de 1 si 0 in input.txt


