# SniesManager — Pipelines SNIES + Estudio de Mercado

## Descripción

Este repositorio contiene una aplicación de escritorio (Tkinter) y varios procesos ETL para trabajar con información del **SNIES**.

El sistema está organizado en dos flujos principales:

- **Pipeline SNIES (programas nuevos + referentes EAFIT)**: descarga/normaliza `Programas.xlsx`, detecta programas nuevos, clasifica referentes con ML y permite ajustes manuales + reentrenamiento.
- **Pipeline de Estudio de Mercado Colombia (Fases 1–6 + reportes segmentados)**: construye una sábana consolidada, agrega por `CATEGORIA_FINAL`, calcula métricas (crecimiento/participación) y scoring, exporta `Estudio_Mercado_Colombia.xlsx` y genera reportes segmentados (Bogotá/Antioquia/Eje Cafetero/Virtual).

## Características principales

- **GUI única** (`app/main.py`): menú con módulos/páginas por funcionalidad.
- **Pipeline SNIES (programas)**:
  - Descarga automática de `Programas.xlsx` vía Selenium.
  - Normalización, detección de programas nuevos e histórico.
  - Clasificación de referentes EAFIT con ML (y probabilidad).
  - Ajustes manuales, sincronización y reentrenamiento.
  - Imputación opcional de `ÁREA_DE_CONOCIMIENTO`.
- **Estudio de Mercado Colombia**:
  - Fase 1: base maestra (clasificación a categorías).
  - Fase 2: insumos históricos (matriculados/inscritos/primer curso/graduados) + OLE.
  - Fase 3: sábana consolidada (`sabana_consolidada.parquet`).
  - Fase 4: agregado + métricas + scoring.
  - Fase 5: exportación Excel nacional + hoja de cambios entre ejecuciones.
  - Fase 6: hoja `eafit_vs_mercado` (opcional).
  - **Reportes segmentados**: Excels independientes por Bogotá / Antioquia / Eje Cafetero / Virtual.
- **Editor de resultados**: página para abrir/filtrar/editar y guardar el Excel final del estudio de mercado.

## Estructura del proyecto

```
proyectoMejora/
├── app/
│   └── main.py                    # GUI (menú + páginas)
├── etl/
│   ├── descargaSNIES.py           # Descarga de Programas.xlsx (Selenium)
│   ├── procesamientoSNIES.py      # Programas nuevos + histórico
│   ├── clasificacionProgramas.py  # Modelo ML referentes EAFIT
│   ├── imputacionAreas.py         # Imputación ÁREA_DE_CONOCIMIENTO
│   ├── mercado_pipeline.py        # Estudio de mercado (Fases 1–6 + segmentos)
│   ├── merge_incremental.py       # Merge incremental + snapshots del estudio
│   ├── scraper_matriculas.py      # Lectura local históricos SNIES (matriculados/inscritos/primer curso/graduados)
│   ├── scraper_ole.py             # OLE (scraper/backup)
│   ├── scoring.py                 # Scoring ponderado (Fase 4)
│   └── config.py                  # Configuración de rutas (incluye outputs/estudio_de_mercado/)
├── models/                        # Modelos ML
├── outputs/
│   ├── Programas.xlsx
│   ├── historico/
│   │   └── raw/                   # CSVs intermedios (Fase 2 mercado)
│   ├── temp/                      # Parquets/cachés (mercado)
│   └── estudio_de_mercado/        # Excels (nacional + segmentos + base maestra F1)
├── ref/
│   ├── backup/                    # Insumos locales (ver sección)
│   ├── referentesUnificados.csv
│   └── catalogoOfertasEAFIT.csv
├── logs/
└── requirements.txt
```

## Requisitos

- Python 3.8+
- Google Chrome instalado (para la descarga automatizada SNIES)
- Entorno virtual recomendado

## Instalación

```bash
python -m venv env
# Windows:
env\Scripts\activate
# Linux / macOS:
# source env/bin/activate

pip install -r requirements.txt
```

## Configuración

### Archivos de referencia (ML)

En `ref/`:

- `referentesUnificados.csv`: dataset de entrenamiento.
- `catalogoOfertasEAFIT.csv`: catálogo de programas EAFIT.

### Estudio de Mercado — referente de categorías (Fase 1)

- Archivo esperado: `Referente_Categorias` en `ref/` (`.xlsx` o `.csv` según lo que tengas en el proyecto).
- La hoja usada para consolidado se define en código (`1_Consolidado` en `etl/config.py` como `HOJA_REFERENTE_CATEGORIAS`). Si cambia el nombre de la hoja en tu Excel, ajusta esa constante.

### Archivo `config.json` (opcional, raíz del proyecto)

`etl/config.py` puede leer un `config.json` junto a la raíz del proyecto (o la carpeta del `.exe`) para:

- `**base_dir**`: carpeta raíz si quieres apuntar datos y salidas a otra ubicación.
- `**outputs_dir**`, `**ref_dir**`, `**models_dir**`, `**docs_dir**`, `**logs_dir**`: rutas absolutas alternativas a los directorios por defecto.
- `**umbral_referente**`: probabilidad mínima (0–1) para marcar un programa como referente en el clasificador; por defecto `0.70` si no se define.

Si el archivo no existe, se usan las rutas relativas habituales dentro del repositorio.

### Ejecución empaquetada (PyInstaller)

Si distribuyes la app como `.exe`, la lógica de `config.py` trata de usar como raíz la carpeta **padre** de `dist/` cuando el ejecutable vive en `dist/`, para que `outputs/`, `ref/` y `models/` sigan siendo los de la raíz del proyecto y no una copia aislada dentro de `dist/`.

### Insumos locales para Estudio de Mercado (`ref/backup/`)

El pipeline de mercado lee insumos desde `ref/backup/` (y subcarpetas). Si falta un archivo/año, el pipeline registra warning y continúa con ceros/NaN según corresponda.

- **Matrículas (matriculados)**: `ref/backup/matriculas/` (Excels por año; el nombre debe contener el año).
- **Inscritos (SNIES oficial)**: `ref/backup/inscritos/inscritos_YYYY.xlsx`.
- **Primer curso (SNIES oficial)**: `ref/backup/matriculas primer curso/primer_curso_YYYY.xlsx`.
- **Graduados (SNIES oficial)**: `ref/backup/graduados/graduados_YYYY.xlsx`.
- **IES**: `ref/backup/ies/Instituciones.xlsx` (hoja `Instituciones`, columna `ACREDITADA_ALTA_CALIDAD`).
- **OLE** (si aplica): `ref/backup/ole_indicadores.csv` o `ref/backup/ole_indicadores.xlsx`.

### Rutas importantes (config)

Las rutas se centralizan en `etl/config.py`. Algunas constantes útiles:

- `ARCHIVO_PROGRAMAS`: `outputs/Programas.xlsx`
- `RAW_HISTORIC_DIR`: `outputs/historico/raw/`
- `ARCHIVO_ESTUDIO_MERCADO`: `outputs/estudio_de_mercado/Estudio_Mercado_Colombia.xlsx`
- `TEMP_DIR`: `outputs/temp/` (parquets de trabajo, cachés y snapshot de “delta”)
- `ESTUDIO_MERCADO_DIR`: `outputs/estudio_de_mercado/`
- `CHECKPOINT_BASE_MAESTRA`: `outputs/temp/base_maestra.parquet`

## Estudio de mercado — notas operativas

- **Validación al arranque**: el pipeline de mercado ejecuta comprobaciones de archivos y carpetas mínimas antes de la Fase 2. Los **errores** bloquean la ejecución; las **advertencias** se registran y el proceso sigue (p. ej. años de matrícula faltantes OLE opcional).
- **Fase 1 y Excel base programas-categoría**: el checkpoint interno es `outputs/temp/base_maestra.parquet`. Desde la GUI, la exportación a Excel va por defecto a `outputs/estudio_de_mercado/Base_Programas_Categoria_F1_<fecha>.xlsx` (no hace falta elegir carpeta de destino manualmente).
- **Fase 3**: genera `outputs/temp/sabana_consolidada.parquet` (y limpia CSV intermedios en `outputs/historico/raw/` tras incorporarlos).
- **Fase 4–5**: el agregado nacional se suele materializar en `outputs/temp/agregado_categorias.parquet`. El Excel nacional puede incluir la hoja `**cambios_vs_anterior`**, que compara contra el snapshot `outputs/temp/agregado_categorias_anterior.parquet` (en la primera corrida solo se crea el snapshot).
- **Reportes segmentados** (`Estudio_Mercado_Bogota.xlsx`, `Estudio_Mercado_Antioquia.xlsx`, `Estudio_Mercado_Eje_Cafetero.xlsx`, `Estudio_Mercado_Virtual.xlsx`): recalculan métricas solo para ese subconjunto. El agregado por segmento puede **reutilizarse desde caché** (`outputs/temp/agregado_<nombre>.parquet`) si la sábana no cambió; en la pantalla de segmentos existe la opción de **forzar recálculo completo** para ignorar esa caché.
- **Hojas útiles en el Excel nacional**: entre otras, `total`, `programas_detalle`, `resumen_ejecutivo`, `eafit_vs_mercado` (si aplica) y `cambios_vs_anterior`. En segmentos, además `**contexto_nacional`** compara el segmento con el agregado país.

## Uso

### Ejecutar la aplicación (GUI)

```bash
python app/main.py
```

Desde el menú podrás:

- Ejecutar el **Pipeline SNIES** (programas nuevos + referentes).
- Usar utilidades como **merge/consolidación**, **imputación** y **reentrenamiento**.
- Ejecutar el **Estudio de Mercado** (Fase 1 y Fases 2–5).
- Generar **reportes segmentados** (botón dedicado).
- Abrir **Resultados Estudio de Mercado** para ver/editar y guardar el Excel.

### Comandos por terminal (opcional)

Si prefieres ejecutar pasos sueltos sin la GUI:

```bash
# Descargar Programas.xlsx desde el portal SNIES
python etl/descargaSNIES.py

# Marcar programas nuevos / histórico (según implementación actual)
python etl/procesamientoSNIES.py

# Entrenar el clasificador de referentes EAFIT
python etl/clasificacionProgramas.py entrenar

# Clasificar programas nuevos
python etl/clasificacionProgramas.py
```

Los flujos del **estudio de mercado** están pensados principalmente para lanzarse desde la GUI (Fases 1–6 y segmentos).

## Salidas clave

- **SNIES**: `outputs/Programas.xlsx` + `outputs/historico/Programas_*.xlsx`
- **Mercado (nacional)**: `outputs/estudio_de_mercado/Estudio_Mercado_Colombia.xlsx`
- **Mercado (segmentos)**: `outputs/estudio_de_mercado/Estudio_Mercado_<Segmento>.xlsx`
- **Checkpoints/cachés**: `outputs/temp/*.parquet` (`base_maestra.parquet`, `sabana_consolidada.parquet`, `agregado_categorias.parquet`, cachés por segmento, etc.)
- **Histórico del estudio**: `outputs/estudio_de_mercado/historico_estudio_de_mercado/` (copias o respaldos según uses la app).

## Modelo de referentes EAFIT (pipeline SNIES)

El clasificador combina **embeddings de frases** (`sentence-transformers`, modelo multilingüe tipo MiniLM) con un **Random Forest** sobre variables derivadas (similitud con candidatos EAFIT, nivel de formación, etc.). El entrenamiento usa etiquetas en `referentesUnificados.csv`. La decisión final respecto a “es referente” también usa el umbral configurable `umbral_referente` en `config.json` (véase arriba).

## Solución de problemas (rápido)

- **Faltan insumos**: revisa `ref/backup/` y `logs/pipeline.log`.
- **Excel en uso**: cierra el archivo y reintenta desde la GUI.
- **Datos mezclados / columnas `_x/_y` en la sábana**: elimina el parquet de `outputs/temp/` asociado y re-ejecuta la fase correspondiente.

## Dependencias principales

- `pandas`, `numpy`
- `scikit-learn`, `sentence-transformers`, `joblib`
- `selenium`, `webdriver-manager`
- `openpyxl`
- `rapidfuzz`, `unidecode`

Ver `requirements.txt` para el listado completo.

## Documentación adicional

- `**ARCHIVOS_PROYECTO.md`**: inventario de entradas/salidas (SNIES + estudio de mercado).
- `**GUIA_EMPAQUETADO.md**` y `**INSTRUCCIONES_EMPAQUETADO.md**`: PyInstaller y `SniesManager.exe`.
- `**docs/ANALISIS_FLUJO_Y_EXCEPCIONES.md**`: flujos y manejo de errores (incluye mercado).
- `**docs/MEJORAS_SISTEMA.md**`: backlog de mejoras técnicas.
- `**tests/README.md**`: cómo ejecutar pruebas (incluye tests del pipeline de mercado).
- `**DIAGNOSTICO_SISTEMA.md**`: informe puntual de diagnóstico; ver también `README` y `config.py` para el estado actual.

