# PROJECT_CONTEXT.md

## 1. Estructura de Archivos

```
proyecto_tp1/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── PROJECT_CONTEXT.md
├── gui/
│   ├── __init__.py
│   ├── app.py
│   ├── utils.py
│   └── frames/
│       ├── __init__.py
│       ├── home_frame.py
│       ├── gamma_frame.py
│       ├── negative_frame.py
│       ├── histogram_frame.py
│       ├── threshold_frame.py
│       ├── equalize_frame.py
│       ├── noise_generator_frame.py
│       ├── contamination_frame.py
│       ├── spatial_filters_frame.py
│       ├── experimentation_frame.py
│       ├── edge_detection_frame.py
│       └── edge_noise_frame.py
├── processing/
│   ├── __init__.py
│   ├── point_operators.py
│   ├── noise_generators.py
│   ├── noise_contamination.py
│   ├── spatial_filters.py
│   └── edge_detection.py
└── assets/
    └── exports/
```

---

## 2. Funciones en `processing/`

### point_operators.py

| Función | Parámetros | Retorno | Descripción | Implementación |
|----------|------------|---------|-------------|---------------|
| `gamma_correction(img, gamma)` | img: np.ndarray, gamma: float | np.ndarray (uint8) | Corrección gamma. `s = r^γ * 255^(1-γ)`. Ref: pág. 8 | Manual (NumPy) |
| `negative(img)` | img: np.ndarray | np.ndarray (uint8) | Negativo: `s = 255 - r`. Ref: pág. 13 | Manual (NumPy) |
| `calc_histogram(img)` | img: np.ndarray | np.ndarray (float64, 256) | Histograma normalizado. `np.bincount(flat)/N`. Ref: pág. 18 | Manual (NumPy) |
| `threshold(img, u)` | img: np.ndarray, u: int | np.ndarray (uint8) | Umbralización: `0 si < u, 255 si ≥ u`. Ref: pág. 11 | Manual (NumPy) |
| `histogram_equalization(img)` | img: np.ndarray | np.ndarray (uint8) | Ecualización: CDF manual, mapeo a [0,255]. Ref: págs. 22-26 | Manual (NumPy) |

### noise_generators.py

| Función | Parámetros | Retorno | Descripción | Implementación |
|----------|------------|---------|-------------|---------------|
| `generate_gaussian(n, mean, sigma)` | n: int, mean: float, sigma: float | np.ndarray | Gaussiana. `np.random.normal(mean, sigma, n)`. Ref: pág. 16 | Librería (np.random) |
| `generate_exponential(n, lam)` | n: int, lam: float | np.ndarray | Exponencial. Transformada inversa: `-ln(1-u)/λ`. Ref: págs. 11-12 | Manual (NumPy) |
| `generate_rayleigh(n, xi)` | n: int, xi: float | np.ndarray | Rayleigh. Transformada inversa: `√(-2ξ²*ln(1-u))` | Manual (NumPy) |
| `gaussian_pdf(x, mean, sigma)` | x: np.ndarray, mean: float, sigma: float | np.ndarray | PDF Gaussiana para graficar | Manual (NumPy) |
| `exponential_pdf(x, lam)` | x: np.ndarray, lam: float | np.ndarray | PDF Exponencial para graficar | Manual (NumPy) |
| `rayleigh_pdf(x, xi)` | x: np.ndarray, xi: float | np.ndarray | PDF Rayleigh para graficar | Manual (NumPy) |

### noise_contamination.py

| Función | Parámetros | Retorno | Descripción | Implementación |
|----------|------------|---------|-------------|---------------|
| `add_noise(img, func, pct, params, mode)` | img: np.ndarray, func, pct: float, params: dict, mode: str | np.ndarray (uint8) | Aplica ruido a % aleatorio de píxeles | Manual (NumPy) |
| `add_gaussian_noise(img, pct, mean, sigma)` | img: np.ndarray, pct: float, mean: float, sigma: float | np.ndarray (uint8) | Ruido Gaussiano aditivo. `I' = I + N(μ,σ)`. Ref: pág. 17 | Manual (NumPy) |
| `add_exponential_noise(img, pct, lam)` | img: np.ndarray, pct: float, lam: float | np.ndarray (uint8) | Ruido Exponencial multiplicativo. `I' = I * Exp(λ)`. Ref: pág. 14 | Manual (NumPy) |
| `add_rayleigh_noise(img, pct, xi)` | img: np.ndarray, pct: float, xi: float | np.ndarray (uint8) | Ruido Rayleigh multiplicativo. `I' = I * Rayleigh(ξ)` | Manual (NumPy) |
| `add_salt_pepper_noise(img, p)` | img: np.ndarray, p: float | np.ndarray (uint8) | Sal y Pimienta. `x≤p→0, x≥1-p→255`. Ref: pág. 19 | Manual (NumPy) |

### spatial_filters.py

| Función | Parámetros | Retorno | Descripción | Implementación |
|----------|------------|---------|-------------|---------------|
| `manual_convolution(img, kernel)` | img: np.ndarray, kernel: np.ndarray | np.ndarray (uint8) | Convolución 2D manual, padding 'edge' | Manual (NumPy, bucles) |
| `manual_median_filter(img, k)` | img: np.ndarray, k: int | np.ndarray (uint8) | Mediana manual, ventana kxk | Manual (NumPy, bucles) |
| `mean_filter(img, k)` | img: np.ndarray, k: int | np.ndarray (uint8) | Filtro media. Kernel uniforme. Ref: pág. 22 | Manual (NumPy) |
| `median_filter(img, k)` | img: np.ndarray, k: int | np.ndarray (uint8) | Filtro mediana. Ref: pág. 30 | Manual (NumPy) |
| `weighted_median_filter(img, weights)` | img: np.ndarray, weights: np.ndarray | np.ndarray (uint8) | Mediana ponderada. Ref: pág. 34 | Manual (NumPy, bucles) |
| `gaussian_kernel(sigma)` | sigma: float | np.ndarray | Kernel Gaussiano 2D. Ref: págs. 25-28 | Manual (NumPy, bucles) |
| `gaussian_filter(img, sigma)` | img: np.ndarray, sigma: float | np.ndarray (uint8) | Filtro Gaussiano | Manual (NumPy) |
| `edge_enhancement_filter(img)` | img: np.ndarray | np.ndarray (uint8) | Realce bordes. Kernel `[-1,-1,-1;-1,9,-1;-1,-1,-1]`. Ref: págs. 36-37 | Manual (NumPy) |

### edge_detection.py

| Función | Parámetros | Retorno | Descripción | Implementación |
|----------|------------|---------|-------------|---------------|
| `convolve2d(img, kernel)` | img: np.ndarray, kernel: np.ndarray | np.ndarray (float64) | Convolución 2D manual, padding `mode="edge"` | Manual (NumPy, bucles) |
| `gradient_magnitude(gx, gy)` | gx: np.ndarray, gy: np.ndarray | np.ndarray (uint8) | Magnitud gradiente = sqrt(gx²+gy²), clip [0,255] | Manual (NumPy) |
| `prewitt_operator(img)` | img: np.ndarray | np.ndarray (uint8) | Detector Prewitt: convolve Gx/Gy, magnitud | Manual (NumPy, bucles) |
| `sobel_operator(img)` | img: np.ndarray | np.ndarray (uint8) | Detector Sobel: convolve Gx/Gy, magnitud | Manual (NumPy, bucles) |
| `prewitt_operator_channel(channel)` | channel: np.ndarray | np.ndarray (uint8) | Prewitt sobre un canal individual | Manual (NumPy, bucles) |
| `sobel_operator_channel(channel)` | channel: np.ndarray | np.ndarray (uint8) | Sobel sobre un canal individual | Manual (NumPy, bucles) |
| `prewitt_color(img, method)` | img: np.ndarray, method: str | np.ndarray (uint8) | Prewitt sobre RGB: combina canales por 'euclidean' o 'max' | Manual (NumPy, bucles) |
| `sobel_color(img, method)` | img: np.ndarray, method: str | np.ndarray (uint8) | Sobel sobre RGB: combina canales por 'euclidean' o 'max' | Manual (NumPy, bucles) |

---

## 3. Frames en `gui/frames/`

## 3. Frames en `gui/frames/`

### home_frame.py
- **Punto TP**: 1 (Carga de imagen)
- **Llama**: `cv2.imread`, `cv2.cvtColor`, `convert_cv_to_ctk`, `plot_histogram`
- **Carga**: Guarda `app.current_image` (grises) y `app.current_image_color` (RGB). .raw solo grises.
- **Controles**:
  - `CTkButton`: "Cargar Imagen"
  - `CTkButton`: "Verificar Carga"
  - `CTkLabel`: Visualización imagen
  - `FigureCanvasTkAgg`: Histograma
- **Librerías**: `customtkinter`, `tkinter.filedialog`, `cv2`, `numpy`, `os`, `gui.utils`

### gamma_frame.py
- **Punto TP**: 1 (Corrección Gamma)
- **Llama**: `gamma_correction`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkSlider`: Gamma (0.1-2.0)
  - `CTkButton`: "Aplicar"
  - 2x `CTkLabel`: Imagen original y procesada
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `processing.point_operators`, `gui.utils`

### negative_frame.py
- **Punto TP**: 2 (Negativo)
- **Llama**: `negative`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkButton`: "Obtener Negativo"
  - 2x `CTkLabel`: Imagen original y procesada
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `processing.point_operators`, `gui.utils`

### histogram_frame.py
- **Punto TP**: 3 (Visualización histograma)
- **Llama**: `calc_histogram` (vía `plot_histogram`)
- **Controles**:
  - `CTkButton`: "Mostrar Histograma"
  - `FigureCanvasTkAgg`: Histograma
- **Librerías**: `customtkinter`, `gui.utils`

### threshold_frame.py
- **Punto TP**: 4 (Umbralización)
- **Llama**: `threshold`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkSlider`: Umbral (0-255)
  - `CTkEntry`: Umbral numérico
  - `CTkButton`: "Umbralizar"
  - 2x `CTkLabel`: Imagen original y procesada
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `processing.point_operators`, `gui.utils`

### equalize_frame.py
- **Punto TP**: 5-6 (Ecualización y Doble Ecualización)
- **Llama**: `histogram_equalization`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkButton`: "Ecualizar"
  - `CTkButton`: "Doble Ecualización"
  - `CTkLabel`: Info idempotencia
  - 2x `CTkLabel`: Imagen original y procesada
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `processing.point_operators`, `gui.utils`

### noise_generator_frame.py
- **Punto TP**: 7 (Generadores de ruido)
- **Llama**: `generate_gaussian`, `generate_exponential`, `generate_rayleigh`, PDFs, `FigureCanvasTkAgg`
- **Controles**:
  - `CTkOptionMenu`: Tipo distribución
  - `CTkEntry`: Parámetros (μ, σ, λ, ξ)
  - `CTkEntry`: Muestras
  - `CTkButton`: "Generar y Graficar"
  - `FigureCanvasTkAgg`: Histograma + PDF teórica
- **Librerías**: `customtkinter`, `numpy`, `matplotlib.figure`, `matplotlib.backends.backend_tkagg`, `processing.noise_generators`

### contamination_frame.py
- **Punto TP**: 8-9 (Contaminación con ruido)
- **Llama**: `add_gaussian_noise`, `add_exponential_noise`, `add_rayleigh_noise`, `add_salt_pepper_noise`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkOptionMenu`: Tipo ruido
  - `CTkEntry`: Parámetros según ruido
  - `CTkSlider`: Porcentaje (0-100%)
  - `CTkButton`: "Aplicar Ruido"
  - 2x `CTkLabel`: Imagen original y contaminada
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `numpy`, `processing.noise_contamination`, `gui.utils`

### spatial_filters_frame.py
- **Punto TP**: 10 (Filtros espaciales)
- **Llama**: `mean_filter`, `median_filter`, `weighted_median_filter`, `gaussian_filter`, `edge_enhancement_filter`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkOptionMenu`: Tipo filtro
  - `CTkEntry`: Parámetros según filtro
  - `CTkButton`: "Aplicar Filtro"
  - 2x `CTkLabel`: Imagen original y filtrada
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `numpy`, `processing.spatial_filters`, `gui.utils`

### experimentation_frame.py
- **Punto TP**: 11-12 (Experimentación y análisis)
- **Llama**: Contaminación + filtros (todas las funciones anteriores), `convert_cv_to_ctk`, `plot_histogram`, `FigureCanvasTkAgg`
- **Controles**:
  - `CTkOptionMenu`: Tipo ruido
  - `CTkEntry`: Parámetros ruido
  - `CTkSlider`: Porcentaje ruido
  - `CTkOptionMenu`: Tipo filtro
  - `CTkEntry`: Parámetros filtro
  - `CTkButton`: "Aplicar Secuencia Completa"
  - `CTkButton`: "Exportar Comparación"
  - 3x `CTkLabel`: Original, Ruidosa, Filtrada
  - 3x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `numpy`, `os`, `datetime`, `matplotlib.figure`, `matplotlib.backends.backend_tkagg`, `processing.noise_contamination`, `processing.spatial_filters`, `gui.utils`

### edge_detection_frame.py
- **Punto TP**: TP2-1 (Detectores de bordes: Prewitt, Sobel)
- **Llama**: `prewitt_operator`, `sobel_operator`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkButton`: "Prewitt"
  - `CTkButton`: "Sobel"
  - `CTkButton`: "Reset"
  - 2x `CTkLabel`: Original, Bordes
  - 2x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `processing.edge_detection`, `gui.utils`

### edge_noise_frame.py
- **Punto TP**: TP2-2 (Bordes sobre imágenes contaminadas con ruido)
- **Llama**: `add_gaussian_noise`, `add_exponential_noise`, `add_rayleigh_noise`, `add_salt_pepper_noise`, `prewitt_operator`, `sobel_operator`, `convert_cv_to_ctk`, `plot_histogram`
- **Controles**:
  - `CTkOptionMenu`: Tipo ruido (Gaussiano/Exponencial/Rayleigh/Sal y Pimienta)
  - `CTkEntry`: Parámetros según ruido
  - `CTkSlider`: Porcentaje (0-100%)
  - `CTkButton`: "Aplicar Ruido"
  - `CTkOptionMenu`: Operador (Prewitt/Sobel)
  - `CTkButton`: "Detectar Bordes"
  - 3x `CTkLabel`: Original, Con Ruido, Bordes
  - 3x `FigureCanvasTkAgg`: Histogramas
- **Librerías**: `customtkinter`, `numpy`, `processing.noise_contamination`, `processing.edge_detection`, `gui.utils`

---

## 4. Librerías por Archivo

| Archivo | Librerías |
|---------|-----------|
| main.py | - |
| gui/app.py | `customtkinter`, `tkinter.filedialog`, `gui.frames.*` |
| gui/utils.py | `numpy`, `customtkinter`, `PIL.Image`, `matplotlib.figure`, `matplotlib.backends.backend_tkagg` |
| gui/frames/home_frame.py | `customtkinter`, `tkinter.filedialog`, `cv2`, `numpy`, `os`, `gui.utils` |
| gui/frames/gamma_frame.py | `customtkinter`, `processing.point_operators`, `gui.utils` |
| gui/frames/negative_frame.py | `customtkinter`, `processing.point_operators`, `gui.utils` |
| gui/frames/histogram_frame.py | `customtkinter`, `gui.utils` |
| gui/frames/threshold_frame.py | `customtkinter`, `processing.point_operators`, `gui.utils` |
| gui/frames/equalize_frame.py | `customtkinter`, `processing.point_operators`, `gui.utils` |
| gui/frames/noise_generator_frame.py | `customtkinter`, `numpy`, `matplotlib.figure`, `matplotlib.backends.backend_tkagg`, `processing.noise_generators` |
| gui/frames/contamination_frame.py | `customtkinter`, `numpy`, `processing.noise_contamination`, `gui.utils` |
| gui/frames/spatial_filters_frame.py | `customtkinter`, `numpy`, `processing.spatial_filters`, `gui.utils` |
| gui/frames/experimentation_frame.py | `customtkinter`, `numpy`, `os`, `datetime`, `matplotlib.figure`, `matplotlib.backends.backend_tkagg`, `processing.noise_contamination`, `processing.spatial_filters`, `gui.utils` |
| gui/frames/edge_detection_frame.py | `customtkinter`, `processing.edge_detection`, `gui.utils` |
| gui/frames/edge_noise_frame.py | `customtkinter`, `numpy`, `processing.noise_contamination`, `processing.edge_detection`, `gui.utils` |
| processing/point_operators.py | `numpy` |
| processing/noise_generators.py | `numpy` |
| processing/noise_contamination.py | `numpy`, `processing.noise_generators` |
| processing/spatial_filters.py | `numpy` |
| processing/edge_detection.py | `numpy` |

---

## 5. Limitaciones Conocidas / Pendientes

1. **Filtros espaciales con bucles**: `manual_convolution` y `manual_median_filter` usan bucles Python (lentos para imágenes grandes). Optimizar con `scipy.signal.convolve2d` NO permitido por TP.

2. **Mediana ponderada custom**: Solo acepta matrices cuadradas. No valida que pesos sean números enteros positivos.

3. **Exportación experimentación**: Imagen exportada usa `matplotlib` (no CTkImage). Los labels de imagen no se exportan, solo las imágenes en escala de grises.

4. **Carga .raw**: Solo funciona con nombres predefinidos en `RAW_SIZES`. No detecta dimensiones automáticamente.

5. **Histograma binario**: Para imágenes umbralizadas, el eje y se escala a 1.3x pero no se destacan las barras en 0 y 255 visualmente.

6. **Sin validación avanzada**: Los entries aceptan cualquier texto. Manejo de errores básico con try/except.

7. **No hay undo/redo**: Una vez aplicado un filtro/ruido, no se puede deshacer salvo recargando la imagen.

8. **Memoria**: Procesar imágenes muy grandes (>2048x2048) puede consumir mucha RAM debido a copias en contaminación y filtros.

9. **edge_detection.gradient_magnitude**: Solo hace clip [0,255], NO normaliza por max. Si gx/gy superan 255, la magnitud se trunca. Prewitt y Sobel internamente llaman a `gradient_magnitude` que trunca en vez de normalizar.

10. **TP2 Pendiente**: Faltan puntos 2-4 (detectores sobre ruido, filtro bilateral, difusión anisótropa, umbralización automática).
