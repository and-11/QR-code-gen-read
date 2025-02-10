Echipa: Moldova_talpa_Power

    Airinei Gabriel-Vlad 152
    Buzdugan Ioan-Michael 152
    Turcu Andrei-Cristian 152
    



Codare:

    fisierul cere un text ce urmeaza a fi codat intr-un cod qr si il salveaza in fisierul qrcode.png

    pasi:

    citim un text de la tastaura

    Luam textul si il convertim intr-un binar avand grija sa respectam convetiile unui cod qr

    Folosim Reed Solomon pentru a genera si partea de error corection

    Punem mesajul nostru intr-o matrice echivalenta cu viitorul nostru cod qr si ii adaugam elementele standard ale unui cod qr

    Aplicam cele 8 masti posibile pe matricea noastra si o alegem pe cea cu scor de penalizare cat mai mic

    La final din matricea generata binar o trnsformam intr un png



Decodare:

     Merge pentru coduri qr cu versiunea 1, 2 sau 3 . 
     Comanda folosire: python3 decode.py .
     Se da ca input numele fisierului cu imaginea (ex: codeqr.png) .
     Imaginea trebuie sa fie in fisierul cu scriptul .


Referinte:

    https://www.thonky.com/qr-code-tutorial/introduction
    https://www.nayuki.io/page/creating-a-qr-code-step-by-step
    https://dev.to/maxart2501/let-s-develop-a-qr-code-generator-part-i-basic-concepts-510a
    https://www.youtube.com/watch?v=w5ebcowAJD8
