import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import logging

# --- 1. CONFIGURACIÓN INICIAL ---

# Silenciar logs innecesarios de Paddle para limpiar la consola
logging.getLogger("ppocr").setLevel(logging.WARNING)

# CORRECCIÓN: Inicializamos SIN 'show_log'
ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')

def generar_opciones_ocr(roi):
    """
    Genera varias versiones de la imagen de la placa para probar
    diferentes filtros y asegurar la lectura de letras difíciles.
    """
    opciones = []
    
    # 1. Escala de grises simple + Zoom
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    zoom_gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    opciones.append(cv2.cvtColor(zoom_gray, cv2.COLOR_GRAY2BGR))

    # 2. Binarización Adaptativa (Para letras cromadas/difíciles)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 19, 9)
    
    # Borde negro para seguridad
    padded = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    zoom_bin = cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    opciones.append(cv2.cvtColor(zoom_bin, cv2.COLOR_GRAY2BGR))
    
    return opciones

def detectar_placa_final(img_path):
    if not os.path.exists(img_path):
        print("Error: No se encuentra la imagen.")
        return

    # Cargar imagen
    img = cv2.imread(img_path)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- FASE 1: DETECCIÓN (UBICAR LA PLACA) ---
    print(">>> Buscando placa...")
    
    # Operaciones morfológicas
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    roi = None
    location = None

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        
        # Filtros de tamaño
        if w > 50 and h > 15 and 1.5 < ar < 5.0:
            location = (x, y, w, h)
            roi = orig[y:y+h, x:x+w]
            break 

    if roi is None:
        print(">>> FALLO: No se detectó ninguna forma de placa.")
        return

    # --- FASE 2: LECTURA OCR INTELIGENTE ---
    print(">>> Placa detectada. Iniciando lectura IA...")
    
    imagenes_a_probar = generar_opciones_ocr(roi)
    text_final = ""
    max_conf = 0
    
    for i, ai_input in enumerate(imagenes_a_probar):
        # cv2.imshow(f"Filtro {i+1}", ai_input) # Descomentar para debug
        
        # Corrección: Llamada limpia sin argumentos extra
        result = ocr_engine.ocr(ai_input)
        
        if result is None or len(result) == 0 or result[0] is None:
            continue

        for line in result[0]:
            try:
                txt = line[1][0]
                conf = line[1][1]
                
                clean = "".join([c for c in txt if c.isalnum()]).upper()
                
                print(f"   Filtro {i+1} detectó: '{clean}' (Conf: {conf:.2f})")

                palabras_prohibidas = ["QUERETARO", "MEXICO", "TRANSPORTE", "AUTO", "FRONT", "TRASERA"]
                es_basura = any(palabra in clean for palabra in palabras_prohibidas)

                if 5 <= len(clean) <= 9 and not es_basura:
                    if conf > max_conf:
                        text_final = clean
                        max_conf = conf
            except Exception as e:
                continue

    # --- RESULTADO FINAL ---
    if text_final:
        print(f"\n=====================================")
        print(f"   PLACA LEÍDA:  {text_final}")
        print(f"=====================================\n")

        (x, y, w, h) = location
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, text_final, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        print("\n>>> ADVERTENCIA: Se detectó el recuadro, pero no se pudo leer el texto con claridad.")

    cv2.imshow("Recorte", roi)
    cv2.imshow("Resultado Final", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_placa_final("image1.png")