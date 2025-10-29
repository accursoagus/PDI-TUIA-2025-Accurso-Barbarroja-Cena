import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv

'''
PROBLEMA 1 - Ecualización local de histograma

Desarrolle una función para implementar la ecualización local del histograma, que reciba como parámetros de entrada la imagen a procesar,  
y el tamaño de la ventana de procesamiento (MxN). Utilice dicha función para analizar la imagen que se muestra en la Figura 1 
(archivo Imagen_con_objetos_ocultos.tiff) e informe cuáles son los detalles escondidos en las diferentes zonas de la misma. 
Luego, desarrolle un análisis sobre la influencia del tamaño de la ventana en los resultados obtenidos.
'''

def lhe(img, H: int, W: int):

# PADDING:
    # Nos aseguramos que H y W sean impares para tener un centro único
    if H % 2 == 0: H += 1
    if W % 2 == 0: W += 1

    # Calculamos el padding necesario (radio del kernel) para 'agrandar' la imagen
    P_H = (H - 1) // 2    #top
    P_W = (W - 1) // 2    #left

    # BORDER_REPLICATE para replicar los valores del borde
    img_padded = cv2.copyMakeBorder(
    src=img, 
    top=P_H, 
    bottom=P_H, 
    left=P_W, 
    right=P_W, 
    borderType=cv2.BORDER_REPLICATE
    )

    # Dimensiones de la imagen original
    R, C = img.shape

    # Inicializamos la matriz de salida
    # Debe ser de punto flotante para cálculos intermedios, luego se convierte a uint8 al retornar
    img_out = np.zeros_like(img, dtype=np.float32)

# BUCLE Y ECUALIZACIÓN DEL VECINDARIO:
    for r in range(R):
        for c in range(C):

            vecindario = img_padded[r:r+H, c:c+W]
            
            # Se obtiene el valor del píxel original que estamos procesando
            pixel_original_intensity = img[r, c]
            
            # Calcular el Histograma Local
            hist_local = cv2.calcHist([vecindario], [0], None, [256], [0, 256]).flatten()
            
            # Normalizamos el histograma y calculamos la CDF local (dividimos cada frecuencia por el total de pixeles de la ventana)
            hist_normalizado = hist_local / (H * W) 
            
            # Calculamos la suma acumulativa de probabilidades
            cdf_local = hist_normalizado.cumsum() 
            
            # Mapear la intensidad del píxel central:
            # La fórmula discreta de ecualización es: s_k = round( (L-1) * CDF(r_k) ), donde L-1 = 255 para uint8 (256 niveles de gris)
            
            # Entonces, obtenemos el valor de la CDF correspondiente a la intensidad del píxel, 
            # y convertimos a entero (0-255) para obtener el nuevo nivel de gris
            nuevo_valor_mapeado = int(255 * cdf_local[pixel_original_intensity])

            # Asignación
            img_out[r, c] = nuevo_valor_mapeado
            
    #return img_out
    return np.clip(img_out, 0, 255).astype(np.uint8)

path_img_detalles = r"G:\AGUS 2\PDI\TP1\Imagen_con_detalles_escondidos.tif"
img_detalles = cv2.imread(path_img_detalles, cv2.IMREAD_GRAYSCALE)

for i in range(5,26,4):
    img = lhe(img_detalles, i, i)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagen ecualizada con una ventana de {i}x{i}')
    plt.show()

# Luego de visualizar las distintas opciones, vemos que en tamaños de ventana más pequeños se visualiza mejor
img_out = lhe(img_detalles, 7, 7)
# img_out = cv2.medianBlur(img_out, 3)
plt.imshow(img_out, cmap='gray')
plt.show()

# Guardamos la imagen
path_guardar = r"G:\AGUS 2\PDI\TP1\img_ecualizada.png"
cv2.imwrite(path_guardar, img_out)



'''
PROBLEMA 2 - Validación de formulario

Se tiene una serie de formularios con información, en formato de imagen, y se pretende validarlos de forma automática
por medio de un script en Python. Para ello, debe considerar distintas restricciones en cada campo.
'''

# ============================================================
# CARGA Y BINARIZACIÓN
# ============================================================
def cargar_y_binarizar(path, th=236):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_th = img < th
    return img, img_th


# ============================================================
# DETECCIÓN DE LÍNEAS
# ============================================================
def detectar_lineas(img_th, umbral_relativo=0.7):
    img_th_ones = img_th.astype(np.uint8)
    img_rows = np.sum(img_th_ones, axis=1)
    img_cols = np.sum(img_th_ones, axis=0)

    th_row = umbral_relativo * np.max(img_rows)
    th_col = umbral_relativo * np.max(img_cols)

    img_rows_th = img_rows > th_row
    img_cols_th = img_cols > th_col

    return img_rows_th, img_cols_th


# ============================================================
# OBTENCIÓN DE ÍNDICES DE LÍNEAS
# ============================================================
def get_line_indices(thresholded_vector, min_dist=10):
    diff = np.diff(thresholded_vector.astype(int), prepend=0)
    line_start_indices = np.where(diff == 1)[0]
    final_indices = []

    if len(line_start_indices) > 0:
        final_indices.append(line_start_indices[0])
        for i in range(1, len(line_start_indices)):
            if line_start_indices[i] - final_indices[-1] > min_dist:
                final_indices.append(line_start_indices[i])

    return final_indices


# ============================================================
# RECORTE DE SECCIONES
# ============================================================
def recortar_secciones(img, row_indices, col_indices, offset=2):
    x_inicio = col_indices[1] + offset
    x_fin = col_indices[-1] - offset

    secciones = []
    for i in range(len(row_indices) - 1):
        y1 = row_indices[i] + offset
        y2 = row_indices[i + 1] - offset
        seccion = img[y1:y2, x_inicio:x_fin]
        secciones.append(seccion)
    return secciones


# ============================================================
# DICCIONARIO DE CELDAS
# ============================================================
def crear_diccionario_celdas(secciones):
    encabezados = [
        "Tipo Formulario", "Nombre y apellido", "Edad", "Mail", "Legajo",
        "Sí/No", "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"
    ]

    if len(secciones) < 10:
        raise ValueError(f"Solo hay {len(secciones)} secciones, se necesitan 10.")

    celdas_dict = {encabezados[i]: secciones[i] for i in range(len(encabezados))}
    celdas_dict.pop('Sí/No')
    return celdas_dict


# ============================================================
# DIVISIÓN DE PREGUNTAS (Sí / No)
# ============================================================
def dividir_preguntas(celdas_dict, offset=4, th_linea=230):

    preguntas = ["Pregunta 1", "Pregunta 2", "Pregunta 3"]

    for preg in preguntas:
        celda = celdas_dict[preg]
        
        # 1. Umbralización simple para detectar tinta (línea central)
        img_th = celda < th_linea
        
        # 2. Suma vertical de píxeles (por columna)
        col_sums = np.sum(img_th, axis=0)
        
        # 3. Localizar la columna de máxima densidad (la línea central)
        x_split = np.argmax(col_sums)
        
        # 4. Dividir la celda en dos mitades (Sí / No)
        celda_si = celda[:, :x_split - offset]
        celda_no = celda[:, x_split + offset:]
        
        # 5. Guardar en el diccionario
        celdas_dict[f"{preg}_SI"] = celda_si
        celdas_dict[f"{preg}_NO"] = celda_no

    return celdas_dict


# ============================================================
# DETECCIÓN DE COMPONENTES
# ============================================================
def extract_and_clean_components(celda_img, th_area_min=5):

    celda_th = cv2.adaptiveThreshold(
        celda_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)
    stats_without_bg = stats[1:]
    ix_area = stats_without_bg[:, cv2.CC_STAT_AREA] > th_area_min
    stats_clean = stats_without_bg[ix_area, :]
    return len(stats_clean), stats_clean


# ============================================================
# CONTEO DE COMPONENTES POR CAMPO
# ============================================================
def contar_componentes_todos(celdas_dict, th_area_min=5):
    conteos = {}
    for nombre, celda in celdas_dict.items():
        num, stats = extract_and_clean_components(celda, th_area_min)
        conteos[nombre] = num
    return conteos


# ============================================================
# VALIDACIÓN BÁSICA POR CANTIDAD DE COMPONENTES
# ============================================================
def validar_por_cantidad(conteos):
    resultados = {}
    for campo, n in conteos.items():
        if "Tipo Formulario" in campo:
            resultados[campo] = "OK"
        elif campo == "Nombre y apellido":
            resultados[campo] = "OK" if 2 <= n <= 25 else "MAL"
        elif campo == "Edad":
            resultados[campo] = "OK" if 1 < n <= 3 else "MAL"
        elif campo == "Mail":
            resultados[campo] = "OK" if n <= 25 and n > 0 else "MAL"
        elif campo == "Legajo":
            resultados[campo] = "OK" if n == 8 else "MAL"
        elif "Pregunta" in campo:
            resultados[campo] = "OK"  # Se validan aparte
        elif campo == "Comentarios":
            resultados[campo] = "OK" if 1 <= n <= 25 else "MAL"
        else:
            resultados[campo] = "OK"
    return resultados


# ============================================================
# VALIDACIÓN DE PREGUNTAS SÍ / NO
# ============================================================
def validar_preguntas(conteos, resultados):
    for i in [1, 2, 3]:
        n_si = conteos.get(f"Pregunta {i}_SI", 0)
        n_no = conteos.get(f"Pregunta {i}_NO", 0)

        if n_si > 1 or n_no > 1:
            resultados[f"Pregunta {i}"] = "MAL"
        elif (n_si == 0 and n_no == 0) or (n_si == 1 and n_no == 1):
            resultados[f"Pregunta {i}"] = "MAL"
        else:
            resultados[f"Pregunta {i}"] = "OK"

        for opcion in ["SI", "NO"]:
            resultados.pop(f"Pregunta {i}_{opcion}", None)

    return resultados


# ============================================================
# FUNCIÓN CONTAR PALABRAS
# ============================================================
def contar_palabras(celda, th_area_min=5, min_space=5, devolver_recortes=False):
    num_comp, stats = extract_and_clean_components(celda, th_area_min)
    if num_comp == 0:
        return (0, []) if devolver_recortes else 0

    stats_sorted = stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]
    palabras = 1
    grupos = [[stats_sorted[0]]]

    for i in range(1, len(stats_sorted)):
        prev_x = stats_sorted[i - 1, cv2.CC_STAT_LEFT]
        prev_w = stats_sorted[i - 1, cv2.CC_STAT_WIDTH]
        curr_x = stats_sorted[i, cv2.CC_STAT_LEFT]
        gap = curr_x - (prev_x + prev_w)
        if gap > min_space:
            palabras += 1
            grupos.append([])
        grupos[-1].append(stats_sorted[i])

    if devolver_recortes:
        recortes = []
        for g in grupos:
            g = np.array(g)
            x_min = np.min(g[:, cv2.CC_STAT_LEFT])
            x_max = np.max(g[:, cv2.CC_STAT_LEFT] + g[:, cv2.CC_STAT_WIDTH])
            y_min = np.min(g[:, cv2.CC_STAT_TOP])
            y_max = np.max(g[:, cv2.CC_STAT_TOP] + g[:, cv2.CC_STAT_HEIGHT])
            recorte = celda[y_min:y_max, x_min:x_max]
            recortes.append(recorte)
        return palabras, recortes
    return palabras


# ============================================================
# IDENTIFICAR TIPO DE FORMULARIO
# ============================================================
def identificar_tipo_formulario(celda, th_area_min=5, min_space=7):
    """
    Identifica el tipo de formulario ('A', 'B', 'C') 
    midiendo la cantidad de tinta (pixeles blancos) en la segunda palabra.
    """
    n_pal, recortes = contar_palabras(celda, th_area_min, min_space, devolver_recortes=True)

    if len(recortes) < 2:
        return "Desconocido"

    segunda_palabra = recortes[1]
    
    # Binarizamos
    _, binaria = cv2.threshold(segunda_palabra, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Contamos píxeles blancos (tinta)
    area_tinta = np.sum(binaria > 0)
    altura, ancho = binaria.shape
    area_total = altura * ancho
    propor_tinta = area_tinta / area_total

    # Heurísticas
    if propor_tinta > 0.60:
        tipo = "B"
    elif propor_tinta > 0.40:
        tipo = "A"
    else:
        tipo = "C"

    return f"Formulario {tipo}"


# ============================================================
# VALIDACIONES POR CANTIDAD DE PALABRAS
# ============================================================
def validar_palabras(celdas_dict, resultados, th_area_min=5, min_space=5):

    # Nombre y Apellido
    campo = "Nombre y apellido"
    if resultados.get(campo) == "OK" and contar_palabras(celdas_dict[campo], th_area_min, min_space) < 2:
        resultados[campo] = "MAL"

    # Edad
    campo = "Edad"
    if resultados.get(campo) == "OK" and contar_palabras(celdas_dict[campo], th_area_min, min_space) > 1:
        resultados[campo] = "MAL"

    # Mail
    campo = "Mail"
    if resultados.get(campo) == "OK" and contar_palabras(celdas_dict[campo], th_area_min, min_space) > 1:
        resultados[campo] = "MAL"

    # Legajo
    campo = "Legajo"
    if resultados.get(campo) == "OK" and contar_palabras(celdas_dict[campo], th_area_min, min_space) > 1:
        resultados[campo] = "MAL"

    return resultados


# ============================================================
# VALIDAR SI EL FORMULARIO ESTÁ OK
# ============================================================
def obtener_estado_global(resultados_dict):
    """Determina si el formulario es globalmente OK (todos los campos son OK)."""
    
    for campo, resultado in resultados_dict.items():
        if campo == "Tipo Formulario":
            continue # Ignoramos el campo de identificación del formulario
        if resultado == "MAL":
            return "MAL"
    return "OK"


# ============================================================
# INDICADOR PARA LA IMAGEN DE SALIDA
# ============================================================
def crear_indicador(crop, estado_global, width=30):

    # Agrega una banda lateral con texto indicativo según el estado del formulario.
    
    # Verificar si el recorte está casi en blanco (sin nombre)
    if np.all(crop > 245):
        alto, ancho = crop.shape
        crop = np.ones((alto, ancho), dtype=np.uint8) * 255  # fondo blanco
        cv2.putText(crop, "NOMBRE DESCONOCIDO", (5, alto // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,), 1, cv2.LINE_AA)

    # Crear la banda de indicador
    alto = crop.shape[0]
    indicador = np.ones((alto, width), dtype=np.uint8) * (200 if estado_global == "OK" else 80)

    # Texto del indicador
    texto = "OK" if estado_global == "OK" else "X"
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.5
    grosor = 1
    (tw, th), _ = cv2.getTextSize(texto, font, escala, grosor)
    x = (width - tw) // 2
    y = (alto + th) // 2
    cv2.putText(indicador, texto, (x, y), font, escala, (0,), grosor, cv2.LINE_AA)

    # Unir recorte + indicador
    crop_con_indicador = np.hstack((crop, indicador))
    return crop_con_indicador


# ============================================================
# FUNCIÓN PARA CREAR CSV
# ============================================================
def guardar_resultados_csv(id_formulario, resultados, nombre_archivo="resultados.csv"):

    encabezado = ["ID", "Nombre y apellido", "Edad", "Mail", "Legajo",
                  "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"]

    # Verificar si el archivo ya existe intentando abrirlo
    try:
        with open(nombre_archivo, "r", encoding="utf-8") as f:
            primera_fila = f.readline().strip().split(",")
            encabezado_presente = primera_fila == encabezado
    except FileNotFoundError:
        encabezado_presente = False
   
    # Abrir en modo append
    with open(nombre_archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Escribir encabezado solo si no está
        if not encabezado_presente:
            writer.writerow(encabezado)
        # Agregar fila de resultados
        fila = [id_formulario] + [resultados.get(campo, "MAL") for campo in encabezado[1:]]
        writer.writerow(fila)


# ============================================================
# PIPELINE PRINCIPAL (Corregido)
# ============================================================
def procesar_formulario(path):
    img, img_th = cargar_y_binarizar(path)
    img_rows_th, img_cols_th = detectar_lineas(img_th)
    row_indices = get_line_indices(img_rows_th)
    col_indices = get_line_indices(img_cols_th)
    secciones = recortar_secciones(img, row_indices, col_indices)
    celdas_dict = crear_diccionario_celdas(secciones)
    celdas_dict = dividir_preguntas(celdas_dict=celdas_dict)
    conteos = contar_componentes_todos(celdas_dict)
    resultados = validar_por_cantidad(conteos)
    resultados = validar_preguntas(conteos, resultados)
    resultados = validar_palabras(celdas_dict, resultados)

    resultados["Tipo Formulario"] = identificar_tipo_formulario(
         celdas_dict["Tipo Formulario"]
    )
   
    estado_global = obtener_estado_global(resultados)

    return resultados, estado_global, celdas_dict


path_img1 = r"G:\AGUS 2\PDI\TP1\formulario_01.png"
path_img2 = r"G:\AGUS 2\PDI\TP1\formulario_02.png"
path_img3 = r"G:\AGUS 2\PDI\TP1\formulario_03.png"
path_img4 = r"G:\AGUS 2\PDI\TP1\formulario_04.png"
path_img5 = r"G:\AGUS 2\PDI\TP1\formulario_05.png"

images = [path_img1, path_img2, path_img3, path_img4, path_img5]

lista_crops_con_indicador = []

for i, img in enumerate(images, start=1):

    resultados_finales, estado_global, celdas_dict = procesar_formulario(img)
    
    print('\n', 'REVISIÓN DE LA IMAGEN', i)
    print(f"ESTADO GLOBAL: {estado_global}")

    for campo, resultado in resultados_finales.items():
        print(f"{campo}: {resultado}")

    # Guardar los resultados en un csv
    guardar_resultados_csv(i, resultados_finales, nombre_archivo="resultados_validacion.csv")

    # Crear el crop con el indicador
    crop_nombre_apellido = celdas_dict["Nombre y apellido"]
    crop_con_indicador = crear_indicador(crop_nombre_apellido, estado_global, width=30)
    lista_crops_con_indicador.append(crop_con_indicador)


path_guardar = r"G:\AGUS 2\PDI\TP1\resumen_validaciones_simple.png"
reporte = np.vstack(lista_crops_con_indicador)
cv2.imwrite(path_guardar, reporte)
print("\n Imagen de resumen generada como: resumen_validaciones_simple.png")
print("Archivo CSV guardado como: resultados_validacion.csv")

