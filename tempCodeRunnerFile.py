import cv2
import numpy as np
import imutils
import pytesseract
import re
import os

path_tesseract = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(path_tesseract):
    path_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = path_tesseract

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_hdr_improved(roi):
    h, w = roi.shape[:2]
    
    # 1. RECORTE (El que funcionó para quitar Querétaro)
    roi = roi[int(h*0.25):int(h*0.70), 0:w]
    
    # 2. GAMMA (Oscurecer para resaltar relieve)
    roi = adjust_gamma(roi, gamma=2.0)
    
    # 3. HDR / DETAIL ENHANCE (Lo que rescató el 866)
    # sigma_r alto ayuda a definir bordes en texturas metálicas
    hdr = cv2.detailEnhance(roi, sigma_s=10, sigma_r=0.15)
    
    # 4. ESCALA DE GRISES
    gray = cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)
    
    # 5. BLACKHAT (Extractor de sombras)
    # Usamos kernel ancho para capturar la estructura horizontal de las letras
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    
    # 6. UMBRAL (Binarización)
    _, thresh = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
    
    # --- LA MEJORA: COSTURA VERTICAL ---
    # Creamos un kernel que es Alto (4px) y Flaco (2px).
    # Esto le dice al código: "Si ves pixeles rotos verticalmente (como la base de la U), únelos".
    # Pero no une horizontalmente (no pega el 8 con el 6).
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_vertical, iterations=1)
    
    # 7. INVERSIÓN
    thresh = cv2.bitwise_not(thresh)
    
    # 8. Limpieza suave
    thresh = cv2.medianBlur(thresh, 3)
    
    # Borde
    thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    return thresh

def correccion_logica(texto):
    """Lógica para reparar errores comunes U->I, G->0, S->5"""
    chars = list(texto)
    # Reemplazos seguros basados en posición
    for i in range(len(chars)):
        c = chars[i]
        
        # Correcciones numéricas (centro de la placa)
        if 2 < i < len(chars)-1: 
            if c == 'B': chars[i] = '8'
            if c == 'D': chars[i] = '0'
            if c == 'S': chars[i] = '5'
            if c == 'G': chars[i] = '6'
            
        # Correcciones de letras (inicio de la placa)
        if i < 3:
            if c == '1' or c == 'I': chars[i] = 'U' # Si empieza con I, es U
            if c == '5': chars[i] = 'S'
            if c == '0': chars[i] = '0' # O la letra D
            
    return "".join(chars)

def detect_and_read_plate(img_path):
    if not os.path.exists(img_path):
        print("No imagen.")
        return

    img = cv2.imread(img_path)
    orig = img.copy()
    
    # Detección
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray_blur, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            if 1.5 < ar < 5 and w > 50:
                screenCnt = approx
                break
    if screenCnt is None:
        print("No placa.")
        return

    x, y, w, h = cv2.boundingRect(screenCnt)
    roi = orig[y:y+h, x:x+w]
    
    # Procesamiento MEJORADO
    final_roi = preprocess_hdr_improved(roi)
    
    # OCR
    configs = [
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    ]
    
    print("\n--- INTENTO MEJORADO ---")
    lecturas = []
    for conf in configs:
        text = pytesseract.image_to_string(final_roi, config=conf)
        clean = re.sub(r'[^A-Z0-9-]', '', text)
        if len(clean) >= 3:
            print(f"Raw: {clean}")
            corregido = correccion_logica(clean)
            lecturas.append(corregido)
            
    best = max(lecturas, key=len) if lecturas else "FALLO"
    print(f">>> GANADOR: {best}")

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(img, best, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("HDR + COSTURA VERTICAL", final_roi)
    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_and_read_plate("image.png")