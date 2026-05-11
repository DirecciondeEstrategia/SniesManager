# Archivos utilizados en el proyecto

Catálogo orientativo de entradas, salidas y artefactos generados. El proyecto tiene **dos flujos**: pipeline **SNIES / referentes EAFIT** y **Estudio de mercado (Fases 1–6)**. Las rutas reales salen de `etl/config.py` y opcionalmente de `config.json` en la raíz.

**Extensiones:** En `ref/`, `referentesUnificados` y `catalogoOfertasEAFIT` suelen estar como `.csv`; el código también puede resolver `.xlsx` según `config.py`.

---

## Archivos de entrada (`ref/`)

### Referentes EAFIT (clasificación SNIES)

| Recurso | Ubicación típica | Uso |
|--------|-------------------|-----|
| **referentesUnificados** | `ref/referentesUnificados.csv` (o `.xlsx`) | Entrenamiento del clasificador de referentes |
| **catalogoOfertasEAFIT** | `ref/catalogoOfertasEAFIT.csv` (o `.xlsx`) | Catálogo EAFIT para comparar con programas SNIES |

**Módulos:** `etl/clasificacionProgramas.py`, `etl/calibracionUmbrales.py` (si aplica).

### Estudio de mercado — Fase 1 (categorías)

| Recurso | Ubicación típica | Uso |
|--------|-------------------|-----|
| **Programas.xlsx** | `outputs/Programas.xlsx` | Universo SNIES filtrado por la hoja `Programas` (debe existir tras el pipeline SNIES o copia válida) |
| **Referente_Categorias** | `ref/Referente_Categorias.xlsx` (o `.csv`) | Matriz de categorías; hoja consolidado según `HOJA_REFERENTE_CATEGORIAS` en `config.py` (p. ej. `1_Consolidado`) |

**Módulos:** `etl/mercado_pipeline.py` (`run_fase1`, `validar_archivos_entrada`).

### Insumos locales SNIES / OLE (`ref/backup/`)

Si falta algún archivo por año, el mercado suele registrar *warning* y continuar con ceros o NaN donde corresponda.

| Tipo | Rutas / patrón |
|------|----------------|
| Matriculados | `ref/backup/matriculas/*.xlsx` (nombre con año) |
| Inscritos | `ref/backup/inscritos/inscritos_YYYY.xlsx` |
| Primer curso | `ref/backup/matriculas primer curso/primer_curso_YYYY.xlsx` |
| Graduados | `ref/backup/graduados/graduados_YYYY.xlsx` |
| IES | `ref/backup/ies/Instituciones.xlsx` (hoja `Instituciones`) |
| OLE (opcional) | `ref/backup/ole_indicadores.csv` o `.xlsx` |

**Módulos:** `etl/scraper_matriculas.py`, `etl/scraper_ole.py`, `etl/mercado_pipeline.py`.

### Referencia adicional (opcional)

- `ref/posParesPositivos.csv`, `ref/preParesPositivos.csv` — material de referencia / análisis.

---

## Archivos de salida principales

### Pipeline SNIES

| Archivo | Ubicación | Notas |
|---------|-----------|--------|
| **Programas.xlsx** | `outputs/Programas.xlsx` | Principal I/O del flujo SNIES |
| **Históricos Programas** | `outputs/historico/Programas_*.xlsx` | Respaldo al renovar descarga |
| **Histórico programas nuevos** | `outputs/HistoricoProgramasNuevos .xlsx` | Nombre con espacio final según `ARCHIVO_HISTORICO` |
| **Calibración** (opcional) | `outputs/calibracion_embeddings.csv`, `outputs/calibracion_resumen.txt` | `etl/calibracionUmbrales.py` |
| **error_screenshot.png** | `outputs/` | Solo si falla la descarga SNIES |

### Estudio de mercado

| Archivo / carpeta | Ubicación | Notas |
|-------------------|-----------|--------|
| Excel nacional | `outputs/estudio_de_mercado/Estudio_Mercado_Colombia.xlsx` | Fase 5; puede incluir hoja `cambios_vs_anterior` |
| Excels segmentados | `outputs/estudio_de_mercado/Estudio_Mercado_Bogota.xlsx`, `..._Antioquia.xlsx`, `..._Eje_Cafetero.xlsx`, `..._Virtual.xlsx` | Reportes regionales / modal |
| Base programas-categoría F1 (export GUI) | `outputs/estudio_de_mercado/Base_Programas_Categoria_F1_*.xlsx` | Desde GUI; datos en parquet de trabajo |
| Histórico estudio | `outputs/estudio_de_mercado/historico_estudio_de_mercado/` | Respaldos según uso |
| CSV intermedios Fase 2 | `outputs/historico/raw/` | `matriculados_*`, `inscritos_*`, `primer_curso_*`, `graduados_*`, OLE; pueden borrarse al cerrar Fase 3 |

### Parquets y cachés (`outputs/temp/`)

| Archivo | Uso |
|---------|-----|
| `base_maestra.parquet` | Checkpoint Fase 1 |
| `sabana_consolidada.parquet` | Sábana Fase 3 |
| `agregado_categorias.parquet` | Agregado nacional Fase 4 |
| `agregado_categorias_anterior.parquet` | Snapshot para hoja `cambios_vs_anterior` |
| `agregado_<segmento>.parquet` | Caché de Fase 4 por segmento (Bogota, etc.) |

Otros parquets pueden existir según merge incremental u opciones de ejecución.

---

## Modelos (`models/`)

| Archivo | Uso |
|---------|-----|
| `clasificador_referentes.pkl` | Random Forest referentes EAFIT |
| `modelo_embeddings.pkl` | SentenceTransformer cacheado |
| `encoder_programas_eafit.pkl` | LabelEncoder EAFIT |
| `clasificador_mercado.pkl` | Clasificador categorías (Fase 1 mercado), `MODELO_CLASIFICADOR_MERCADO` |

---

## Logs

- `logs/pipeline.log` — operaciones generales (`etl/pipeline_logger.py`).

---

## Flujo de archivos (resumen)

### A) Pipeline SNIES (alto nivel)

1. **Descarga** `etl/descargaSNIES.py` → escribe `outputs/Programas.xlsx`; respalda anterior en `outputs/historico/`.
2. **Normalización** `etl/normalizacion.py` → lee/escribe `Programas.xlsx`.
3. **Programas nuevos** `etl/procesamientoSNIES.py` → histórico + `Programas.xlsx`.
4. **Clasificación** `etl/clasificacionProgramas.py` → `ref/` + `models/*.pkl` + `Programas.xlsx`.

### B) Estudio de mercado (`etl/mercado_pipeline.py`)

1. **Fase 1:** `outputs/Programas.xlsx` + `ref/Referente_Categorias*` → `base_maestra.parquet`.
2. **Fase 2:** Excels/CSV en `ref/backup/` + lectura scrapers → CSV en `outputs/historico/raw/`.
3. **Fase 3:** merge a `sabana_consolidada.parquet`; limpieza de CSV raw según implementación.
4. **Fase 4:** agregación + scoring → `agregado_categorias.parquet` (típico).
5. **Fase 5:** Excel nacional + delta vs snapshot anterior.
6. **Fase 6 / segmentos:** Excels adicionales; cachés `agregado_<nombre>.parquet`.

---

## Resumen por tipo

| Tipo | Ejemplos |
|------|-----------|
| CSV | `referentesUnificados.csv`, `catalogoOfertasEAFIT.csv`, `ole_indicadores.csv`, raw Fase 2 |
| XLSX | `Programas.xlsx`, Excels en `ref/backup/`, estudio mercado en `outputs/estudio_de_mercado/` |
| Parquet | `outputs/temp/*.parquet` |
| PKL | `models/*.pkl` |
| TXT / log | `logs/pipeline.log`, calibración |

---

## Notas

1. **Requisitos SNIES:** catálogo EAFIT, referentes para entrenar, y `Programas.xlsx` tras descarga; modelos `.pkl` salvo primer entrenamiento.
2. **Requisitos mercado Fase 1+:** `Programas.xlsx`, `Referente_Categorias`, insumos en `ref/backup/` según métricas deseadas.
3. **Portabilidad:** usar `config.json` (`base_dir`, `outputs_dir`, etc.) en lugar de rutas fijas en el código.
4. **Búsqueda en código:** `grep -r "ARCHIVO_REFERENTE_CATEGORIAS" etl/` o el nombre del archivo en `mercado_pipeline.py`.

---

## Documentación relacionada

- Visión general: `README.md`.
- Empaquetado: `INSTRUCCIONES_EMPAQUETADO.md`, `GUIA_EMPAQUETADO.md`.
