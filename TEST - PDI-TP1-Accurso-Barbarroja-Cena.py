import cv2
import matplotlib.pyplot as plt
import numpy as np

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

img_detalles = cv2.imread(r"G:\AGUS 2\PDI\TP1\Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

img_out = lhe(img_detalles, 7, 7)
# img_out = cv2.medianBlur(img_out, 3)
plt.imshow(img_out, cmap='gray')
plt.show()

for i in range(5,26,4):
    img = lhe(img_detalles, i, i)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagen ecualizada con una ventana de {i}x{i}')
    plt.show()


'''
PROBLEMA 2 - Validación de formulario

Se tiene una serie de formularios con información, en formato de imagen, y se pretende validarlos de forma automática
por medio de un script en Python. Para ello, debe considerar distintas restricciones en cada campo.
'''

# Cargar imagen y binarizar:
img = cv2.imread(r"G:\AGUS 2\PDI\TP1\formulario_05.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show(block=False)

th = 236     # Por ahora parece un número ideal para que no se unan dos letras
img_th = img < th   # True donde hay tinta, False donde es fondo
plt.imshow(img_th, cmap='gray')
plt.show(block=False)

# Detectar líneas para luego segmentar:
# Sumar pixeles en cada fila y columna (primero paso de booleano a entero)
img_th_ones = img_th.astype(np.uint8)
img_rows = np.sum(img_th_ones, axis=1)
img_cols = np.sum(img_th_ones, axis=0)

# Detectar posiciones de líneas horizontales y verticales (umbral 0.7)
th_row = 0.7 * np.max(img_rows)
th_col = 0.7 * np.max(img_cols)

img_rows_th = img_rows > th_row
img_cols_th = img_cols > th_col

# Visualización del progreso
# Suma de píxeles por fila y columna:
plt.figure(figsize=(7,4))
plt.plot(img_rows, label='Suma por fila')
plt.axhline(th_row, color='r', linestyle='--', label='Umbral filas')
plt.title("Perfil de suma de píxeles por fila (líneas horizontales)")
plt.legend()

plt.figure(figsize=(7,4))
plt.plot(img_cols, label='Suma por columna')
plt.axhline(th_col, color='r', linestyle='--', label='Umbral columnas')
plt.title("Perfil de suma de píxeles por columna (líneas verticales)")
plt.legend()
plt.show(block=False)

# Imagen original con las líneas detectadas:
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
for y in np.where(img_rows_th)[0]:
    plt.axhline(y, color='r', linewidth=0.5)
for x in np.where(img_cols_th)[0]:
    plt.axvline(x, color='b', linewidth=0.5)
plt.title("Líneas detectadas (rojas = horizontales, azules = verticales)")
plt.show(block=False)

# Vemos que todavía no logramos dividir la parte de si/no de las preguntas

# Localizamos los índices de las líneas detectadas:
def get_line_indices(thresholded_vector):
    '''
    Detecta los índices (coordenadas) de las líneas a partir del vector umbralizado.
    Busca transiciones 0->1 (inicio de línea) para marcar la posición.
    Filtra por distancia para evitar múltiples detecciones por líneas gruesas.
    '''
    # Diferencia entre píxeles adyacentes: 1 marca inicio (0->1), -1 marca fin (1->0)
    # Se agrega un 0 al inicio para detectar la primera línea si está en el borde 
    diff = np.diff(thresholded_vector.astype(int), prepend=0)
    
    # Obtenemos los índices donde el valor cambia a 1 (inicio de una línea)
    line_start_indices = np.where(diff == 1)[0]
    
    # Es necesario un filtrado adicional: si una línea es gruesa (ej. 3 píxeles), 
    # queremos solo un índice, en este caso elegimos el borde superior
    
    final_indices = []
    if len(line_start_indices) > 0:
        # La primera línea siempre se agrega
        final_indices.append(line_start_indices[0])
        
        # Filtrar las siguientes líneas por una distancia mínima (anchura de la línea)
        for i in range(1, len(line_start_indices)):
            # Distancia mínima, ejemplo heurístico: 10 píxeles
            if line_start_indices[i] - final_indices[-1] > 10: 
                final_indices.append(line_start_indices[i])
                
    return final_indices

# Obtenemos las coordenadas de las líneas horizontales (filas) y verticales (columnas)
row_indices = get_line_indices(img_rows_th)
col_indices = get_line_indices(img_cols_th)
print(f"Líneas Horizontales (filas): {row_indices}")
print(f"Líneas Verticales (columnas): {col_indices}")


# Recorte de las secciones de interés del formulario:

offset = 2 # Para evitar incluir líneas negras
# Queremos tomar solo la parte de datos (a la derecha de las 'etiquetas' o nombres de cada campo)
# La columna de datos empieza en col_indices[1], ya que habíamos segmentado en 2 columnas
x_inicio = col_indices[1] + offset
x_fin = col_indices[-1] - offset 

secciones = []
for i in range(len(row_indices) - 1):
    y1 = row_indices[i] + offset
    y2 = row_indices[i + 1] - offset

    # recortamos el renglón para quedarnos con la parte completada, sin la etiqueta del campo
    seccion = img[y1:y2, x_inicio:x_fin]
    secciones.append(seccion)

# Ahora asociaremos en un diccionario las secciones con el 'encabezado' al que pertenecen
encabezados = [
    "Tipo Formulario", "Nombre y apellido", "Edad", "Mail", "Legajo", 
    'Sí/No', "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"
]
# Verificamos si hay suficientes secciones para asociar con las claves (en este caso 10 elementos)
if len(secciones) < 10:
    raise ValueError(f"Error: La lista 'secciones' solo tiene {len(secciones)} elementos, pero se necesitan 10 claves.")

# Inicializamos el diccionario
celdas_dict = {}

for i, key in enumerate(encabezados):
    # Esto asigna la 'imagen' de cada seccion (secciones[i]) a su clave correspondiente
    celdas_dict[key] = secciones[i]

# ANTES DE SEGUIR CORRIENDO EL CÓDIGO, SE DEBE HACER ENTER EN LA TERMINAL (sino, por algún motivo, el último bucle queda abierto y falla)

# Eliminamos del diccionario el renglón de Sí/No que encabeza a la parte de las preguntas, ya que no sirve para nada:
celdas_dict.pop('Sí/No')

# División de las preguntas en dos columnas (sí y no):
preguntas = ["Pregunta 1", "Pregunta 2", "Pregunta 3"]
TH_LINEA = 230  # umbral para detectar la línea dentro de la celda
OFFSET = 4      # para evitar incluir la línea negra en las mitades

for preg in preguntas:
    celda = celdas_dict[preg]
    
    # Umbralización para detectar tinta
    img_th = celda < TH_LINEA
    
    # Suma  de píxeles por columna
    col_sums = np.sum(img_th, axis=0)
    
    # Localizar la columna de máxima densidad para encontrar la línea central
    x_split = np.argmax(col_sums)
    
    # Dividir la celda en dos mitades (Sí / No)
    celda_si = celda[:, :x_split - OFFSET]
    celda_no = celda[:, x_split + OFFSET:]
    
    # Guardar en el diccionario
    celdas_dict[f"{preg}_SI"] = celda_si
    celdas_dict[f"{preg}_NO"] = celda_no

celdas_dict.pop("Pregunta 1")
celdas_dict.pop("Pregunta 2")
celdas_dict.pop("Pregunta 3")

# Visualizamos los recortes hechos hasta ahora:
num_celdas = len(celdas_dict)
n_rows = 6
n_cols = 2 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 10))
axes = axes.flatten()
i = 0

for titulo, recorte in celdas_dict.items():

    axes[i].imshow(recorte, cmap='gray')    # Mostrar el recorte en el subplot actual
    
    axes[i].set_title(titulo, fontsize=10)  # Asignar el título (usamosla clave del diccionario)
    
    axes[i].axis('off')         # Eliminamos los ejes para una visualización más limpia 
    
    i += 1

plt.tight_layout()
axes[i].imshow(recorte, cmap='gray', vmin=0, vmax=255)
plt.suptitle("Revisión de la Segmentación del Formulario", fontsize=14, y=0.98)
plt.show()

# Por qué se ven celdas de preguntas negras en el gráfico?

celda_vacia = celdas_dict["Pregunta 1_NO"]
print("Rango de intensidades:", celda_vacia.min(), celda_vacia.max())
# Si bien se veían negras las celdas como "Pregunta 1_NO", sus píxeles son de valor 255 (blancos), solo era un error de visualización

# EXTRAER Y CONTAR CARACTERES:

def extract_and_clean_components(celda_img, th_area_min):

    # Aísla las componentes conectadas (caracteres) en una celda y elimina ruido (fragmentos de líneas divisorias pequeñas)
    
    # 1. Binarización adaptativa, robusta para separar caracteres
    celda_th = cv2.adaptiveThreshold(
        src=celda_img, 
        maxValue=255, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,           # Invierte: tinta es 255
        blockSize=11,                                 # Vecindad de píxeles
        C=2                                           # Constante de ajuste
    )

    # Encontrar componentes conectadas y sus estadísticas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)
    
    # Filtrar ruido 
    stats_without_bg = stats[1:]
    ix_area = stats_without_bg[:, cv2.CC_STAT_AREA] > th_area_min 
    stats_clean = stats_without_bg[ix_area, :]
    num_clean_components = len(stats_clean)
    
    return num_clean_components, stats_clean

conteos_componentes = {}
for nombre, celda in celdas_dict.items():
    num, stats = extract_and_clean_components(celda, th_area_min=5)
    conteos_componentes[nombre] = num
    print(nombre, "→", num, "componentes")

# VISUALIZACIÓN (para entender mejor, y ajustar parámetros de extract_and_clean_components):
campo = "Mail"
celda = celdas_dict[campo]

# Parámetros de prueba  
TH_BIN = 200
TH_AREA = 5

# Paso 1: Umbralización inversa
_, celda_th = cv2.threshold(celda, TH_BIN, 255, cv2.THRESH_BINARY_INV)

# Paso 2: Componentes conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)

# Paso 3: Filtrado por área
stats_no_bg = stats[1:]  # Sacamos el fondo
ix_area = stats_no_bg[:, cv2.CC_STAT_AREA] > TH_AREA
stats_filtradas = stats_no_bg[ix_area, :]

plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.imshow(celda, cmap='gray')
plt.title(f"{campo} (original)")

plt.subplot(1,2,2)
plt.imshow(celda_th, cmap='gray')
for x, y, w, h, a in stats_filtradas:
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=1)
    plt.gca().add_patch(rect)
plt.title(f"Componentes detectadas ({len(stats_filtradas)})")
plt.show()

# VALIDACIONES DE CANTIDAD DE CARACTERES
resultados = {}
for campo, n in conteos_componentes.items():
    
    # --- Campos con reglas específicas ---
    if campo == "Nombre y apellido":
        resultados[campo] = "OK" if 2 <= n <= 25 else "MAL"
        
    elif campo == "Edad":
        resultados[campo] = "OK" if 1 <= n <= 3 else "MAL"
        
    elif campo == "Mail":
        resultados[campo] = "OK" if n <= 25 and n > 0 else "MAL"
        
    elif campo == "Legajo":
        resultados[campo] = "OK" if n == 8 else "MAL"
        
    elif campo == "Comentarios":
        resultados[campo] = "OK" if 1 <= n <= 25 else "MAL"
        
    # --- Preguntas de Sí/No ---
    elif "Pregunta" in campo:
        resultados[campo] = "OK"  # las vamos a verificar después (una sola marca)
        
    # --- Tipo de formulario ---
    elif "Tipo Formulario" in campo:
        resultados[campo] = "OK"  # no se valida por cantidad
        
    # --- Por defecto ---
    else:
        resultados[campo] = "OK"

# Mostrar resultados provisorios
for k, v in resultados.items():
    print(f"{k}: {v}")

# Validación de las preguntas Sí / No
for i in [1, 2, 3]:
    n_si = conteos_componentes.get(f"Pregunta {i}_SI", 0)
    n_no = conteos_componentes.get(f"Pregunta {i}_NO", 0)

    # Caso inválido: más de 1 marca en alguna celda
    if n_si > 1 or n_no > 1:
        resultados[f"Pregunta {i}"] = "MAL"
    
    # Caso inválido: ambas vacías o ambas marcadas
    elif (n_si == 0 and n_no == 0) or (n_si == 1 and n_no == 1):
        resultados[f"Pregunta {i}"] = "MAL"
    
    # Caso válido: solo una marcada
    else:
        resultados[f"Pregunta {i}"] = "OK"

# Eliminamos campos de respuestas individuales (Sí / No)
for i in [1, 2, 3]:
    for opcion in ["SI", "NO"]:
        resultados.pop(f"Pregunta {i}_{opcion}", None)

for k, v in resultados.items():
    print(f"{k}: {v}")

# VALIDAR POR CANTIDAD DE PALABRAS:

def contar_palabras(celda, th_area_min, min_space, devolver_recortes=False):
    """
    Cuenta cuántas 'palabras' hay en una celda basándose en la separación horizontal.
    Si devolver_recortes=True, devuelve también una lista con las subimágenes de cada palabra
    """
    num_comp, stats = extract_and_clean_components(celda, th_area_min)
    if num_comp == 0:
        return (0, []) if devolver_recortes else 0

    # Ordenar componentes de izquierda a derecha
    stats_sorted = stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]

    palabras = 1
    grupos = [[stats_sorted[0]]]  # lista de listas: cada grupo = una palabra

    # Agrupar componentes por palabra
    for i in range(1, len(stats_sorted)):
        prev_x = stats_sorted[i - 1, cv2.CC_STAT_LEFT]
        prev_w = stats_sorted[i - 1, cv2.CC_STAT_WIDTH]
        curr_x = stats_sorted[i, cv2.CC_STAT_LEFT]
        gap = curr_x - (prev_x + prev_w)

        if gap > min_space:
            palabras += 1
            grupos.append([])
        grupos[-1].append(stats_sorted[i])

    # Si se piden recortes, generarlos
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

TH_AREA_MIN = 5
MIN_SPACE = 5  # Distancia mínima para considerar dos 'palabras'

# Guardamos en el dicc el tipo de formulario:
campo = "Tipo Formulario"
n_pal, recortes = contar_palabras(celdas_dict[campo], TH_AREA_MIN, MIN_SPACE, devolver_recortes=True)
resultados[campo] = recortes[1]

# Nombre y Apellido: mínimo 2 palabras
campo = "Nombre y apellido"
if resultados.get(campo) == "OK":
    n_pal = contar_palabras(celdas_dict[campo], TH_AREA_MIN, MIN_SPACE)
    print('Nro de palabras detectadas:', n_pal)
    if n_pal < 2:
        resultados[campo] = "MAL"

# Edad: solo una 'palabra'
campo = "Edad"
if resultados.get(campo) == "OK":
    n_pal = contar_palabras(celdas_dict[campo], TH_AREA_MIN, MIN_SPACE)
    print('Nro de palabras detectadas:', n_pal)
    if n_pal > 1:
        resultados[campo] = "MAL"

# Mail: solo una palabra
campo = "Mail"
if resultados.get(campo) == "OK":
    n_pal = contar_palabras(celdas_dict[campo], TH_AREA_MIN, MIN_SPACE)
    print('Nro de palabras detectadas:', n_pal)
    if n_pal > 1:
        resultados[campo] = "MAL"

# Legajo: solo una palabra
campo = "Legajo"
if resultados.get(campo) == "OK":
    n_pal = contar_palabras(celdas_dict[campo], TH_AREA_MIN, MIN_SPACE)
    print('Nro de palabras detectadas:', n_pal)
    if n_pal > 1:
        resultados[campo] = "MAL"


# Mostrar resultados finales
print("Resultados finales:")
for k, v in resultados.items():
    print(f"{k}: {v}")
