# Procesamiento de Imágenes y Visión por Computadora - TP1

## Operadores Puntuales, Histogramas y Filtrado Espacial

**Universidad Nacional de Hurlingham**

**Autores:** Hingrid Queiroz - Juan Esteban Cruz
**Docente:** Dra. Juliana Gambini
**Fecha:** Abril 2026

---

## Descripcion

Este proyecto implementa una aplicacion de escritorio para el procesamiento de imagenes en escala de grises, desarrollada como parte del Trabajo Practico 1 de la materia. La aplicacion permite:

- Aplicar operadores puntuales (correccion gamma, negativo, umbralizacion).
- Calcular y visualizar histogramas de frecuencia relativa.
- Realizar ecualizacion de histograma para mejorar el contraste.
- Generar y aplicar ruido sintetico (Gaussiano, Exponencial, Rayleigh, Sal y Pimienta).
- Aplicar filtros espaciales manuales (Media, Mediana, Mediana Ponderada, Gaussiano, Realce de bordes).
- Experimentar con combinaciones de ruido y filtrado en una interfaz dedicada.

Todas las operaciones de procesamiento estan implementadas **manualmente** utilizando unicamente NumPy, sin recurrir a funciones predefinidas de OpenCV o SciPy, cumpliendo con los requisitos del TP.

---

## Estructura del Proyecto

```
proyecto_tp1/
├── main.py                          # Punto de entrada
├── gui/                             # Interfaz grafica
│   ├── app.py                       # Clase principal App
│   ├── frames/
│   │   ├── home_frame.py            # Carga de imagen
│   │   ├── gamma_frame.py           # Correccion gamma
│   │   ├── negative_frame.py        # Negativo
│   │   ├── histogram_frame.py       # Visualizacion histograma
│   │   ├── threshold_frame.py       # Umbralizacion
│   │   ├── equalize_frame.py        # Ecualizacion
│   │   ├── noise_generator_frame.py # Generadores de ruido
│   │   ├── contamination_frame.py   # Contaminacion
│   │   ├── spatial_filters_frame.py # Filtros espaciales
│   │   └── experimentation_frame.py # Experimentacion
│   └── utils.py                     # Utilidades de visualizacion
├── processing/                      # Logica de procesamiento
│   ├── point_operators.py           # Puntos 1-6
│   ├── noise_generators.py          # Punto 7
│   ├── noise_contamination.py       # Puntos 8-9
│   └── spatial_filters.py           # Punto 10
├── assets/
│   └── exports/                     # Exportaciones generadas
├── requirements.txt                 # Dependencias
└── README.md                       # Documentacion
```

---

## Requisitos e Instalacion

### Requisitos

- Python 3.10+
- Librerias necesarias:

```
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
customtkinter>=5.2.0
Pillow>=10.0.0
```

### Instalacion

1. Clonar o descargar el repositorio.
2. Crear entorno virtual (recomendado):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecucion

```bash
python main.py
```

---

## Guia de Uso

### Inicio
Cargar una imagen en escala de grises (.png, .jpg, .bmp). Visualiza la imagen original y su histograma.

### Gamma
Aplicar correccion gamma con valor entre 0.1 y 2.0:
- gamma < 1: aclara sombras
- gamma > 1: oscurece altas luces
- *Ref: Apunte pag. 8*

### Negativo
Obtener el negativo fotografico: `s = 255 - r`
- *Ref: Apunte pag. 13*

### Histograma
Visualizar histograma de frecuencias relativas de la imagen actual.
- *Ref: Apunte pag. 18*

### Umbral
Convertir imagen a binaria con umbral u:
- pixeles >= u → 255 (blanco)
- pixeles < u → 0 (negro)
- *Ref: Apunte pag. 11*

### Ecualizar
Aplicar ecualizacion de histograma para mejorar contraste.
- *Ref: Apunte pagg. 22-26*

### Doble Ecualizacion
Aplicar ecualizacion dos veces consecutivas. Demuestra idempotencia.
- *Ref: Apunte pag. 24*

### Ruido
Generar muestras aleatorias con distribuciones:
- **Gaussiana**: N(mean, sigma)
- **Exponencial**: Exp(lambda)
- **Rayleigh**: Rayleigh(xi)

Visualiza histograma con curva teorica superpuesta.
- *Ref: Apunte pagg. 11-12, 16*

### Contaminar Imagen
Aplicar ruido a la imagen cargada:

| Tipo | Formula | Parametros |
|------|---------|------------|
| Gaussiano | I_c = I + N(u,sigma) | u, sigma, % |
| Exponencial | I_c = I * Exp(lambda) | lambda, % |
| Rayleigh | I_c = I * Rayleigh(xi) | xi, % |
| Sal y Pimienta | p→0, 1-p→255 | p (0-0.5) |

- *Ref: Apunte pagg. 14, 17, 19*

### Filtros Espaciales
Filtros implementados manualmente:

| Filtro | Descripcion | Parametros |
|--------|-------------|------------|
| Media | Kernel uniforme kxk | k (impar) |
| Mediana | Mediana de vecindad kxk | k (impar) |
| Mediana Ponderada | Pesos custom | matriz nx n |
| Gaussiano | Kernel G(x,y) con sigma | sigma |
| Realce | Kernel [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]] | - |

- *Ref: Apunte pagg. 22-37*

### Experimentacion
Combina ruido y filtrado: Original -> Ruidosa -> Filtrada. Visualiza las 3 imagenes con histogramas. Exporta comparacion como PNG.

---

## Referencias Teoricas

Basado en apuntes de la materia:
- Operadores Puntuales e Histogramas (pagg. 8-30)
- Procesamiento Espacial y Ruido de Imagen (pagg. 5-37)

---

## Ejemplos de Resultados

| Ruido | Filtro | Observacion |
|-------|--------|-------------|
| Sal y Pimienta (p=0.1) | Mediana 3x3 | Elimina casi completamente ruido impulsivo |
| Sal y Pimienta (p=0.1) | Media 3x3 | Reduce ruido pero deja manchas grises |
| Gaussiano (sigma=25) | Gauss (sigma=1.5) | Suaviza efectivamente ruido gaussiano |
| Gaussiano (sigma=25) | Mediana 3x3 | Menos efectivo que filtro Gauss |
| Exponencial (lambda=0.3) | Media 5x5 | Suavizado moderado |

---

## Autor

Hingrid Queiroz - Juan Esteban Cruz
Universidad Nacional de Hurlingham
Procesamiento de Imagenes y Vision por Computadora

---

## Licencia

Desarrollado con fines academicos para el TP1 de la materia. Todos los derechos reservados.