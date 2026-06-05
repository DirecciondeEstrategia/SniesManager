"""
Módulo de configuración centralizado para manejar rutas de archivos.

Este módulo detecta automáticamente si el código se está ejecutando como:
- Script de Python (desarrollo; raíz = carpeta del repo)
- Ejecutable .EXE (PyInstaller; raíz = carpeta del .exe, p. ej. SniesManager/)

Y configura las rutas de manera apropiada para cada caso.

También soporta configurar una carpeta raíz del usuario mediante config.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import datetime

import pandas as pd


def _esta_en_onedrive(ruta: Path) -> bool:
    """
    Detecta si una ruta está dentro de una carpeta sincronizada
    con OneDrive o SharePoint en Windows.
    Funciona comprobando si algún componente de la ruta contiene
    indicadores conocidos de OneDrive en Windows.
    """
    try:
        ruta_str = str(ruta.resolve()).lower()
        indicadores = [
            "onedrive",
            "sharepoint",
            "\\onedrive - ",  # OneDrive personal o empresarial
            "/onedrive - ",
        ]
        return any(ind in ruta_str for ind in indicadores)
    except Exception:
        return False


def _get_temp_dir_local(base_path: Path) -> Path:
    """
    Si la app corre desde OneDrive/SharePoint, devuelve una ruta local
    en %LOCALAPPDATA%\\SniesManager\\temp para evitar conflictos de
    sincronización con los archivos intermedios del pipeline.
    Si no está en OneDrive, usa outputs/temp/ junto al ejecutable.
    """
    if _esta_en_onedrive(base_path):
        # Usar carpeta local del sistema que OneDrive nunca sincroniza
        local_app_data = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser()
        temp_local = local_app_data / "SniesManager" / "temp"
        temp_local.mkdir(parents=True, exist_ok=True)
        return temp_local
    else:
        # Ruta estándar junto al ejecutable
        return base_path / "outputs" / "temp"


def _get_default_base_path() -> Path:
    """
    Obtiene la ruta base por defecto del proyecto.

    Si se ejecuta como .EXE (PyInstaller), la carpeta del ejecutable es la base
    (p. ej. SniesManager/ con ref/, models/, outputs/, docs/, config.json junto al .exe).

    Si se ejecuta como script, usa la carpeta del repositorio (padre de etl/).
    """
    if getattr(sys, 'frozen', False):
        # Ejecutándose como .EXE empaquetado con PyInstaller.
        # La carpeta del exe (SniesManager/) ES la raíz del proyecto:
        # contiene ref/, models/, outputs/, docs/, config.json junto al exe.
        # No hay que subir niveles — usamos directamente la carpeta del exe.
        base_path = Path(sys.executable).resolve().parent
    else:
        # Ejecutándose como script de Python
        base_path = Path(__file__).resolve().parents[1]
    
    return base_path


def _get_config_file_path() -> Path:
    """Obtiene la ruta del archivo config.json."""
    default_base = _get_default_base_path()
    return default_base / "config.json"


# Ruta pública a config.json (misma que usa _load_config / GUI).
CONFIG_PATH: Path = _get_config_file_path()


# Caché de configuración para evitar lecturas repetidas del disco
_config_cache: dict | None = None
_config_cache_mtime: float | None = None


def _load_config() -> dict:
    """
    Carga la configuración desde config.json si existe.
    Usa caché en memoria; se invalida al guardar o si cambia la fecha del archivo.
    
    Returns:
        Diccionario con la configuración, o diccionario vacío si no existe.
    """
    global _config_cache, _config_cache_mtime
    config_file = _get_config_file_path()
    if not config_file.exists():
        _config_cache = {}
        _config_cache_mtime = None
        return {}
    try:
        mtime = config_file.stat().st_mtime
        if _config_cache is not None and _config_cache_mtime == mtime:
            return _config_cache
    except OSError:
        pass
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            _config_cache = json.load(f)
            _config_cache_mtime = config_file.stat().st_mtime if config_file.exists() else None
            return _config_cache
    except (json.JSONDecodeError, IOError) as e:
        print(f"[WARN] No se pudo cargar config.json: {e}. Usando valores por defecto.")
        _config_cache = {}
        _config_cache_mtime = None
        return {}


def _save_config(config: dict) -> bool:
    """
    Guarda la configuración en config.json.
    Invalida la caché para que la próxima lectura use el archivo actualizado.
    
    Args:
        config: Diccionario con la configuración a guardar
        
    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    global _config_cache, _config_cache_mtime
    config_file = _get_config_file_path()
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        _config_cache = None
        _config_cache_mtime = None
        return True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar config.json: {e}")
        return False


def get_base_dir() -> Path:
    """
    Obtiene el directorio base del proyecto siempre de forma relativa
    al ejecutable (o al script en desarrollo).

    No lee base_dir de config.json. Esto permite que el mismo proyecto
    compartido en OneDrive/SharePoint funcione automáticamente en cualquier
    computador, sin importar la ruta local del usuario, porque cada quien
    tiene su propia carpeta de usuario de Windows pero el exe siempre
    está junto a las carpetas del proyecto (ref/, outputs/, models/).
    """
    return _get_default_base_path()


def set_base_dir(base_dir: Path) -> bool:
    """
    Mantenida por compatibilidad con el resto del código que la llama.
    Ya NO guarda base_dir en config.json porque la ruta se calcula
    automáticamente desde la ubicación del ejecutable.

    Retorna True si la carpeta existe (comportamiento esperado por los
    llamadores), False si no existe.
    """
    return base_dir.exists() and base_dir.is_dir()


# Ruta base del proyecto (se obtiene dinámicamente)
_BASE_PATH = None

def _refresh_base_path():
    """Actualiza la ruta base del proyecto."""
    global _BASE_PATH
    _BASE_PATH = get_base_dir()

_refresh_base_path()

# ========= RUTAS DE DIRECTORIOS =========
# Todas las rutas son relativas a BASE_PATH (que puede ser configurada por el usuario)
def _get_path(config_key: str, default_relative: str) -> Path:
    """
    Obtiene una ruta desde config o usa la ruta relativa por defecto.
    
    Si existe una ruta absoluta en config.json, la usa.
    Si no, construye la ruta relativa a BASE_PATH.
    """
    config = _load_config()
    config_value = config.get(config_key, "").strip()
    if config_value:
        return Path(config_value)
    return _BASE_PATH / default_relative

OUTPUTS_DIR = _get_path("outputs_dir", "outputs")
HISTORIC_DIR = OUTPUTS_DIR / "historico"
REF_DIR = _get_path("ref_dir", "ref")
MODELS_DIR = _get_path("models_dir", "models")
DOCS_DIR = _get_path("docs_dir", "docs")
LOGS_DIR = _get_path("logs_dir", "logs")

# Crear directorios si no existen
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORIC_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ========= RUTAS DE ARCHIVOS =========
ARCHIVO_PROGRAMAS = OUTPUTS_DIR / "Programas.xlsx"
ARCHIVO_HISTORICO = OUTPUTS_DIR / "HistoricoProgramasNuevos.xlsx"

# ========= PIPELINE MERCADO (Fase 1+) =========
TEMP_DIR = _get_temp_dir_local(_BASE_PATH)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ESTUDIO_MERCADO_DIR = OUTPUTS_DIR / "estudio_de_mercado"
ESTUDIO_MERCADO_DIR.mkdir(parents=True, exist_ok=True)

HISTORICO_ESTUDIO_MERCADO_DIR = ESTUDIO_MERCADO_DIR / "historico_estudio_de_mercado"
HISTORICO_ESTUDIO_MERCADO_DIR.mkdir(parents=True, exist_ok=True)
RAW_HISTORIC_DIR = HISTORIC_DIR / "raw"
RAW_HISTORIC_DIR.mkdir(parents=True, exist_ok=True)
HOJA_REFERENTE_CATEGORIAS = "1_Consolidado"
CHECKPOINT_BASE_MAESTRA = TEMP_DIR / "base_maestra.parquet"
MODELO_CLASIFICADOR_MERCADO = MODELS_DIR / "clasificador_mercado.pkl"

# URLs y selectores para scrapers (Fase 2). Confirmar con cliente.
SNIES_URLS = {
    "matriculados": {"url": "", "selectors": {}},
    "inscritos": {"url": "", "selectors": {}},
}
OLE_URLS = {"url": "", "selectors": {}}

# Fase 4: scoring y comparación de costos por nivel de formación
# Cada nivel tiene su propio benchmark; se sobrescriben desde config.json.
# Especialización / maestría calibrados con p75 de costos reales (SNIES, Esp+Maestría).
BENCHMARK_COSTO_PREGRADO = 10_000_000  # sin datos suficientes en pipeline — sin cambio
BENCHMARK_COSTO_ESPECIALIZACION = 11_910_000  # p75 real Esp (antes 13_400_000)
BENCHMARK_COSTO_MAESTRIA = 13_686_800  # p75 real Maestría (antes 18_000_000)
BENCHMARK_COSTO_DOCTORADO = 25_000_000  # sin datos en pipeline — sin cambio
# Sub-niveles de especialización con mercados de costo muy distinto:
BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA = 31_895_490  # p75 real Esp. Médico Quirúrgica
BENCHMARK_COSTO_ESP_TECNOLOGICA = 5_336_000  # p75 real Esp. Tecnológica
BENCHMARK_COSTO_ESP_TECNICO_PROF = 5_433_677  # p75 real Esp. Técnico Profesional
# Valor genérico (fallback para niveles no clasificados)
BENCHMARK_COSTO = 11_910_000  # alineado con Especialización
SMLMV = 1_300_000  # O el valor dinámico que tenga

# ── Filtro de niveles de formación ──────────────────────────────────────────
NIVELES_MERCADO: list[str] = [
    "ESPECIALIZACIÓN",
    "MAESTRÍA",
    "ESPECIALIZACIÓN MÉDICO QUIRÚRGICA",
    "ESPECIALIZACIÓN TECNOLÓGICA",
    "ESPECIALIZACIÓN TÉCNICO PROFESIONAL",
    "UNIVERSITARIO",
]

# ── Universos separados para scoring diferenciado (Fase 4 / Fase 5) ──────────
# Filtran PROGRAMAS (no categorías). Al filtrar por NIVELES_POSGRADO dentro de
# run_fase4_desde_sabana, el groupby por CATEGORIA_FINAL produce 288 categorías
# (todas tienen al menos un programa ESP/MAE). Por NIVELES_PREGRADO produce 144.
NIVELES_POSGRADO: frozenset[str] = frozenset({
    "ESPECIALIZACIÓN",
    "MAESTRÍA",
    "ESPECIALIZACIÓN MÉDICO QUIRÚRGICA",
    "ESPECIALIZACIÓN TECNOLÓGICA",
    "ESPECIALIZACIÓN TÉCNICO PROFESIONAL",
})
NIVELES_PREGRADO: frozenset[str] = frozenset({"UNIVERSITARIO"})

# Sanity check: la unión debe coincidir con NIVELES_MERCADO.
assert (
    NIVELES_POSGRADO | NIVELES_PREGRADO
) == frozenset(NIVELES_MERCADO), (
    "NIVELES_POSGRADO ∪ NIVELES_PREGRADO debe igualar NIVELES_MERCADO. "
    "Si añades un nivel en NIVELES_MERCADO, asígnalo también a uno de los dos universos."
)

# ── Análisis regional ────────────────────────────────────────────────────────
# Umbral mínimo de nuevos matriculados (primer_curso) que debe tener una celda
# (categoría × departamento) en el último año para que se calculen sus métricas
# de crecimiento regional. Celdas por debajo se marcan DATOS_INSUFICIENTES=True.
# Calibración: el umbral anterior era 50 sobre matrícula total. Primer_curso es
# ~5× menor que matrícula total en posgrado → equivalente ≈ 10.
# Se usa 10 para balancear cobertura vs. ruido estadístico en celdas pequeñas.
UMBRAL_REGIONAL_PRIMER_CURSO: int = 10

# Valores históricos conocidos del SMLMV por año calendario.
# El año corriente usará obtener_smlmv_vigente(), salvo que se haga override en sesión.
SMLMV_POR_ANO: dict[int, int] = {
    2023: 1_160_000,
    2024: 1_300_000,
    2025: 1_423_500,
}

# ── Rango de años con datos SNIES disponibles ─────────────────────────────────
# AÑO_INICIO_HISTORICO    : primer año en ref/backup/ (matrícula, inscritos, etc.)
# AÑO_FIN_DATOS           : último año con datos completos → ACTUALIZAR cuando lleguen datos nuevos
# AÑO_INICIO_PRIMER_CURSO : primer año disponible en archivos primer_curso_*.xlsx
#
# Para soportar p. ej. 2025: añadir "AÑO_FIN_DATOS": 2025 en config.json
# (no requiere modificar ningún archivo Python).
AÑO_INICIO_HISTORICO: int = 2019
AÑO_FIN_DATOS: int = 2024
AÑO_INICIO_PRIMER_CURSO: int = 2014

try:
    _cfg = _load_config()
    if isinstance(_cfg.get("BENCHMARK_COSTO_PREGRADO"), (int, float)) and _cfg["BENCHMARK_COSTO_PREGRADO"] > 0:
        BENCHMARK_COSTO_PREGRADO = float(_cfg["BENCHMARK_COSTO_PREGRADO"])
    if isinstance(_cfg.get("BENCHMARK_COSTO_ESPECIALIZACION"), (int, float)) and _cfg["BENCHMARK_COSTO_ESPECIALIZACION"] > 0:
        BENCHMARK_COSTO_ESPECIALIZACION = float(_cfg["BENCHMARK_COSTO_ESPECIALIZACION"])
    if isinstance(_cfg.get("BENCHMARK_COSTO_MAESTRIA"), (int, float)) and _cfg["BENCHMARK_COSTO_MAESTRIA"] > 0:
        BENCHMARK_COSTO_MAESTRIA = float(_cfg["BENCHMARK_COSTO_MAESTRIA"])
    if isinstance(_cfg.get("BENCHMARK_COSTO_DOCTORADO"), (int, float)) and _cfg["BENCHMARK_COSTO_DOCTORADO"] > 0:
        BENCHMARK_COSTO_DOCTORADO = float(_cfg["BENCHMARK_COSTO_DOCTORADO"])
    if isinstance(_cfg.get("BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA"), (int, float)) and _cfg["BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA"] > 0:
        BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA = float(_cfg["BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA"])
    if isinstance(_cfg.get("BENCHMARK_COSTO_ESP_TECNOLOGICA"), (int, float)) and _cfg["BENCHMARK_COSTO_ESP_TECNOLOGICA"] > 0:
        BENCHMARK_COSTO_ESP_TECNOLOGICA = float(_cfg["BENCHMARK_COSTO_ESP_TECNOLOGICA"])
    if isinstance(_cfg.get("BENCHMARK_COSTO_ESP_TECNICO_PROF"), (int, float)) and _cfg["BENCHMARK_COSTO_ESP_TECNICO_PROF"] > 0:
        BENCHMARK_COSTO_ESP_TECNICO_PROF = float(_cfg["BENCHMARK_COSTO_ESP_TECNICO_PROF"])
    # Fallback genérico (mantener compatibilidad)
    if isinstance(_cfg.get("BENCHMARK_COSTO"), (int, float)) and _cfg["BENCHMARK_COSTO"] > 0:
        BENCHMARK_COSTO = float(_cfg["BENCHMARK_COSTO"])
    # ── Override de rango de años desde config.json ────────────────────────
    if isinstance(_cfg.get("AÑO_FIN_DATOS"), int) and _cfg["AÑO_FIN_DATOS"] > 2019:
        AÑO_FIN_DATOS = int(_cfg["AÑO_FIN_DATOS"])
    if isinstance(_cfg.get("AÑO_INICIO_HISTORICO"), int) and _cfg["AÑO_INICIO_HISTORICO"] >= 2015:
        AÑO_INICIO_HISTORICO = int(_cfg["AÑO_INICIO_HISTORICO"])
    if isinstance(_cfg.get("AÑO_INICIO_PRIMER_CURSO"), int) and _cfg["AÑO_INICIO_PRIMER_CURSO"] >= 2010:
        AÑO_INICIO_PRIMER_CURSO = int(_cfg["AÑO_INICIO_PRIMER_CURSO"])
    if isinstance(_cfg.get("SMLMV_POR_ANO"), dict):
        for _k, _v in _cfg["SMLMV_POR_ANO"].items():
            try:
                SMLMV_POR_ANO[int(_k)] = int(_v)
            except (TypeError, ValueError, KeyError):
                pass
except Exception:
    pass


def invalidate_config_cache() -> None:
    """Invalida la caché en memoria de config.json para forzar la próxima lectura desde disco."""
    global _config_cache, _config_cache_mtime
    _config_cache = None
    _config_cache_mtime = None


def reload_year_and_smlmv_from_config_file() -> None:
    """
    Vuelve a leer config.json y actualiza AÑO_* y SMLMV_POR_ANO en el proceso actual.
    Útil tras guardar cambios desde la GUI sin reiniciar la aplicación.
    """
    global AÑO_FIN_DATOS, AÑO_INICIO_HISTORICO, AÑO_INICIO_PRIMER_CURSO, SMLMV_POR_ANO
    invalidate_config_cache()
    c = _load_config()
    if isinstance(c.get("AÑO_FIN_DATOS"), int) and c["AÑO_FIN_DATOS"] > 2019:
        AÑO_FIN_DATOS = int(c["AÑO_FIN_DATOS"])
    if isinstance(c.get("AÑO_INICIO_HISTORICO"), int) and c["AÑO_INICIO_HISTORICO"] >= 2015:
        AÑO_INICIO_HISTORICO = int(c["AÑO_INICIO_HISTORICO"])
    if isinstance(c.get("AÑO_INICIO_PRIMER_CURSO"), int) and c["AÑO_INICIO_PRIMER_CURSO"] >= 2010:
        AÑO_INICIO_PRIMER_CURSO = int(c["AÑO_INICIO_PRIMER_CURSO"])
    if isinstance(c.get("SMLMV_POR_ANO"), dict):
        for _k, _v in c["SMLMV_POR_ANO"].items():
            try:
                SMLMV_POR_ANO[int(_k)] = int(_v)
            except (TypeError, ValueError, KeyError):
                pass


def obtener_smlmv_vigente() -> int:
    """
    Retorna el SMLMV vigente según el año actual.

    Si el año no está configurado en SMLMV_POR_ANO, devuelve la constante SMLMV como fallback.
    """
    year = datetime.datetime.now().year
    return SMLMV_POR_ANO.get(year, SMLMV)


def set_smlmv_sesion(valor: float) -> bool:
    """
    Persiste el SMLMV efectivo en config.json (clave 'SMLMV').

    Returns:
        True si se guardó correctamente, False en caso contrario.
    """
    try:
        c = _load_config()
        c["SMLMV"] = float(valor)
        return _save_config(c)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar SMLMV en config.json: {e}")
        return False


def get_smlmv_sesion() -> float:
    """
    Obtiene el SMLMV efectivo desde config.json (clave 'SMLMV').

    Si no existe la clave o el archivo, retorna el valor vigente según el año
    actual (usando obtener_smlmv_vigente()).
    """
    try:
        c = _load_config()
        val = c.get("SMLMV")
        if isinstance(val, (int, float)):
            return float(val)
    except Exception:
        pass
    return float(obtener_smlmv_vigente())

# Fase 5: exportación estudio de mercado
ARCHIVO_ESTUDIO_MERCADO = ESTUDIO_MERCADO_DIR / "Estudio_Mercado_Colombia.xlsx"


def _resolve_referencia_path(ref_dir: Path, nombre_base: str) -> Path:
    """Resuelve ruta a referentesUnificados o catalogoOfertasEAFIT (.xlsx o .csv).
    Busca primero en ref_dir/backup (donde suelen estar los archivos), luego en ref_dir.
    """
    for carpeta in (ref_dir / "backup", ref_dir):
        if not carpeta.exists():
            continue
        for ext in [".xlsx", ".csv"]:
            p = carpeta / f"{nombre_base}{ext}"
            if p.exists():
                return p
    return ref_dir / f"{nombre_base}.xlsx"


ARCHIVO_REFERENTES = _resolve_referencia_path(REF_DIR, "referentesUnificados")
ARCHIVO_CATALOGO_EAFIT = _resolve_referencia_path(REF_DIR, "catalogoOfertasEAFIT")
ARCHIVO_NORMALIZACION = DOCS_DIR / "normalizacionFinal.xlsx"
ARCHIVO_REFERENTE_CATEGORIAS = _resolve_referencia_path(REF_DIR, "Referente_Categorias")

# Fase 6 (opcional): archivo de programas EAFIT en proceso de valorización (EAFIT vs Mercado)
PROGRAMAS_EAFIT: Path = REF_DIR / "backup" / "programas_para_valorizacion.xlsx"

# ========= CONFIGURACIÓN DE DESCARGA SNIES =========
SNIES_URL = "https://hecaa.mineducacion.gov.co/consultaspublicas/programas"
DOWNLOAD_DIR = OUTPUTS_DIR  # Usar el mismo directorio de outputs
RENAME_TO = "Programas"
_CONFIG = _load_config()  # Cargar configuración
HEADLESS = _CONFIG.get("headless", False)
# Validación de config con valores por defecto y advertencias
_raw_max_wait = _CONFIG.get("max_wait_download_sec", 180)
MAX_WAIT_DOWNLOAD_SEC = max(60, int(_raw_max_wait)) if isinstance(_raw_max_wait, (int, float)) else 180
if isinstance(_raw_max_wait, (int, float)) and int(_raw_max_wait) < 60:
    print("[WARN] config: max_wait_download_sec debe ser >= 60. Usando 60.")
_raw_log = str(_CONFIG.get("log_level", "INFO")).upper()
LOG_LEVEL = _raw_log if _raw_log in ("DEBUG", "INFO", "WARNING", "ERROR") else "INFO"
if _raw_log != LOG_LEVEL:
    print("[WARN] config: log_level inválido. Usando INFO.")
LOG_MAX_BYTES = max(1024 * 1024, int(_CONFIG.get("log_max_bytes", 2 * 1024 * 1024)))
DOWNLOAD_RETRIES = max(1, int(_CONFIG.get("download_retries", 2)))
_raw_timeout = _CONFIG.get("selenium_page_load_timeout_sec", 120)
SELENIUM_PAGE_LOAD_TIMEOUT_SEC = max(30, int(_raw_timeout)) if isinstance(_raw_timeout, (int, float)) else 120
try:
    _raw_umbral = float(_CONFIG.get("umbral_referente", 0.70))
    UMBRAL_REFERENTE = max(0.0, min(1.0, _raw_umbral))
except (TypeError, ValueError):
    UMBRAL_REFERENTE = 0.70

# Limpieza de históricos: umbral configurable
MAX_ARCHIVOS_HISTORICOS = max(5, int(_CONFIG.get("max_archivos_historicos", 20)))

# ========= CONFIGURACIÓN DE HOJAS =========
HOJA_PROGRAMAS = "Programas"
HOJA_HISTORICO = "ProgramasNuevos"

# ========= FUNCIÓN PARA ACTUALIZAR RUTAS =========
def update_paths_for_base_dir(base_dir: Path) -> None:
    """
    Actualiza todas las rutas para usar un nuevo directorio base.
    
    Esta función debe llamarse antes de ejecutar el pipeline si se quiere
    cambiar el directorio base en tiempo de ejecución.
    
    Args:
        base_dir: Nuevo directorio base a usar
    """
    global _BASE_PATH, OUTPUTS_DIR, HISTORIC_DIR, REF_DIR, MODELS_DIR, DOCS_DIR, LOGS_DIR
    global ARCHIVO_PROGRAMAS, ARCHIVO_HISTORICO, ARCHIVO_REFERENTES, ARCHIVO_CATALOGO_EAFIT, ARCHIVO_NORMALIZACION
    global TEMP_DIR, ESTUDIO_MERCADO_DIR, HISTORICO_ESTUDIO_MERCADO_DIR
    global RAW_HISTORIC_DIR, CHECKPOINT_BASE_MAESTRA, MODELO_CLASIFICADOR_MERCADO, ARCHIVO_REFERENTE_CATEGORIAS, ARCHIVO_ESTUDIO_MERCADO

    if not set_base_dir(base_dir):
        raise ValueError(f"No se pudo establecer el directorio base: {base_dir}")
    
    _BASE_PATH = base_dir
    
    # Recalcular todas las rutas
    OUTPUTS_DIR = _get_path("outputs_dir", "outputs")
    HISTORIC_DIR = OUTPUTS_DIR / "historico"
    REF_DIR = _get_path("ref_dir", "ref")
    MODELS_DIR = _get_path("models_dir", "models")
    DOCS_DIR = _get_path("docs_dir", "docs")
    LOGS_DIR = _get_path("logs_dir", "logs")

    ESTUDIO_MERCADO_DIR = OUTPUTS_DIR / "estudio_de_mercado"
    ESTUDIO_MERCADO_DIR.mkdir(parents=True, exist_ok=True)

    HISTORICO_ESTUDIO_MERCADO_DIR = ESTUDIO_MERCADO_DIR / "historico_estudio_de_mercado"
    HISTORICO_ESTUDIO_MERCADO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Recalcular rutas de archivos
    ARCHIVO_PROGRAMAS = OUTPUTS_DIR / "Programas.xlsx"
    ARCHIVO_HISTORICO = OUTPUTS_DIR / "HistoricoProgramasNuevos.xlsx"
    ARCHIVO_NORMALIZACION = DOCS_DIR / "normalizacionFinal.xlsx"
    TEMP_DIR = _get_temp_dir_local(base_dir)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    if _esta_en_onedrive(_BASE_PATH):
        print(
            f"[INFO] App detectada en OneDrive/SharePoint. "
            f"Archivos temporales redirigidos a: {TEMP_DIR}"
        )
    RAW_HISTORIC_DIR = HISTORIC_DIR / "raw"
    RAW_HISTORIC_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_BASE_MAESTRA = TEMP_DIR / "base_maestra.parquet"
    MODELO_CLASIFICADOR_MERCADO = MODELS_DIR / "clasificador_mercado.pkl"
    ARCHIVO_ESTUDIO_MERCADO = ESTUDIO_MERCADO_DIR / "Estudio_Mercado_Colombia.xlsx"

    # Funciones para detección automática de formato en archivos de referencia
    def _cargar_archivo_referencia(base_path: Path, nombre_base: str) -> Path:
        """Busca .xlsx o .csv primero en base_path/backup (donde suelen estar los archivos), luego en base_path."""
        for carpeta in (base_path / "backup", base_path):
            if not carpeta.exists():
                continue
            for ext in ['.xlsx', '.csv']:
                archivo = carpeta / f"{nombre_base}{ext}"
                if archivo.exists():
                    return archivo
        raise FileNotFoundError(
            f"No se encontró {nombre_base}.xlsx ni {nombre_base}.csv en {base_path} ni en {base_path / 'backup'}"
        )
    
    def _leer_datos_flexible(ruta: Path, **kwargs) -> pd.DataFrame:
        """
        Lee Excel o CSV automáticamente con manejo robusto.
        """
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")
        
        suffix = ruta.suffix.lower()
        
        if suffix == '.csv':
            # Manejo robusto de CSV
            encodings = kwargs.pop('encoding', ['utf-8', 'latin-1', 'cp1252'])
            separators = kwargs.pop('sep', [',', ';'])
            
            if isinstance(encodings, str):
                encodings = [encodings]
            if isinstance(separators, str):
                separators = [separators]
            
            # Intentar diferentes combinaciones
            for encoding in encodings:
                for sep in separators:
                    try:
                        return pd.read_csv(ruta, encoding=encoding, sep=sep, **kwargs)
                    except Exception:
                        continue
            
            # Último intento con defaults de pandas
            return pd.read_csv(ruta, **kwargs)
        
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(ruta, **kwargs)
        
        else:
            raise ValueError(f"Formato no soportado: {suffix}")
    
    # Rutas dinámicas para archivos de referencia
    try:
        ARCHIVO_REFERENTES = _cargar_archivo_referencia(REF_DIR, "referentesUnificados")
        ARCHIVO_CATALOGO_EAFIT = _cargar_archivo_referencia(REF_DIR, "catalogoOfertasEAFIT")
        ARCHIVO_REFERENTE_CATEGORIAS = _cargar_archivo_referencia(REF_DIR, "Referente_Categorias")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        # Fallback a rutas por defecto (para compatibilidad)
        ARCHIVO_REFERENTES = REF_DIR / "referentesUnificados.xlsx"
        ARCHIVO_CATALOGO_EAFIT = REF_DIR / "catalogoOfertasEAFIT.xlsx"
        ARCHIVO_REFERENTE_CATEGORIAS = REF_DIR / "Referente_Categorias.xlsx"

    # Crear directorios si no existen
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIC_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Exponer funciones de utilidad para uso en otros módulos
def cargar_archivo_referencia(base_path: Path, nombre_base: str) -> Path:
    """Busca .xlsx o .csv en base_path/backup primero, luego en base_path."""
    # PRIORIDAD: Buscar primero en backup/ (donde suelen estar los archivos)
    carpetas_busqueda = [base_path / "backup", base_path]
    
    for carpeta in carpetas_busqueda:
        if not carpeta.exists():
            continue
        for ext in ['.xlsx', '.csv']:
            archivo = carpeta / f"{nombre_base}{ext}"
            if archivo.exists():
                print(f"[INFO] Archivo {nombre_base} encontrado en: {archivo}")
                return archivo
    
    # Si no se encuentra, mostrar información de diagnóstico
    print(f"[ERROR] No se encontró {nombre_base}.xlsx ni {nombre_base}.csv")
    print(f"  Buscado en: {base_path / 'backup'}")
    print(f"  Buscado en: {base_path}")
    if base_path.exists():
        print(f"  Archivos en {base_path}: {list(base_path.glob('*'))[:5]}")
        if (base_path / "backup").exists():
            print(f"  Archivos en {base_path / 'backup'}: {list((base_path / 'backup').glob('*'))[:5]}")
    
    raise FileNotFoundError(
        f"No se encontró {nombre_base}.xlsx ni {nombre_base}.csv en {base_path / 'backup'} ni en {base_path}"
    )

def leer_datos_flexible(ruta: Path, **kwargs) -> pd.DataFrame:
    """Lee Excel o CSV automáticamente con manejo robusto."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    
    suffix = ruta.suffix.lower()
    
    if suffix == '.csv':
        # Manejo robusto de CSV
        encodings = kwargs.pop('encoding', ['utf-8', 'latin-1', 'cp1252'])
        separators = kwargs.pop('sep', [',', ';'])
        
        if isinstance(encodings, str):
            encodings = [encodings]
        if isinstance(separators, str):
            separators = [separators]
        
        # Intentar diferentes combinaciones
        for encoding in encodings:
            for sep in separators:
                try:
                    return pd.read_csv(ruta, encoding=encoding, sep=sep, **kwargs)
                except Exception:
                    continue
        
        # Último intento con defaults de pandas
        return pd.read_csv(ruta, **kwargs)
    
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(ruta, **kwargs)
    
    else:
        raise ValueError(f"Formato no soportado: {suffix}")

# Funciones actualizadas para obtener rutas dinámicas
def get_archivo_referentes() -> Path:
    """Obtiene la ruta al archivo de referentes (xlsx o csv)."""
    return cargar_archivo_referencia(REF_DIR, "referentesUnificados")

def get_archivo_catalogo_eafit() -> Path:
    """Obtiene la ruta al archivo del catálogo EAFIT (xlsx o csv)."""
    return cargar_archivo_referencia(REF_DIR, "catalogoOfertasEAFIT")


# Exponer rutas de config para la GUI (evitar duplicar lógica exe/script)
def get_config_file_path() -> Path:
    """Ruta del archivo config.json (según se ejecute como script o .EXE)."""
    return _get_config_file_path()

def get_default_base_path() -> Path:
    """Ruta base por defecto (carpeta del .exe o del proyecto)."""
    return _get_default_base_path()


# Última ejecución exitosa (guardada en config.json para la GUI)
def get_last_success() -> tuple[str | None, float | None]:
    """
    Obtiene fecha y duración de la última ejecución exitosa del pipeline.
    Returns:
        (iso_timestamp o None, duracion_minutos o None)
    """
    c = _load_config()
    return (c.get("last_success_iso"), c.get("last_success_duration_min"))

def set_last_success(iso_timestamp: str, duration_minutes: float) -> bool:
    """Guarda en config.json la última ejecución exitosa."""
    c = _load_config()
    c["last_success_iso"] = iso_timestamp
    c["last_success_duration_min"] = round(duration_minutes, 2)
    return _save_config(c)


def get_benchmark_costo(nivel: str = "general") -> float:
    nivel_norm = str(nivel).lower().strip()
    # Sub-niveles de especialización — detectar ANTES de "especial" genérico
    if "médico" in nivel_norm or "medico" in nivel_norm or "quirúrg" in nivel_norm or "quirurg" in nivel_norm:
        return BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA
    if "tecnológ" in nivel_norm or "tecnolog" in nivel_norm:
        return BENCHMARK_COSTO_ESP_TECNOLOGICA
    if "técnico" in nivel_norm or "tecnico" in nivel_norm:
        return BENCHMARK_COSTO_ESP_TECNICO_PROF
    # Niveles principales
    if "especial" in nivel_norm:
        return BENCHMARK_COSTO_ESPECIALIZACION
    if "maestr" in nivel_norm or "magist" in nivel_norm:
        return BENCHMARK_COSTO_MAESTRIA
    if "doctor" in nivel_norm:
        return BENCHMARK_COSTO_DOCTORADO
    if "universit" in nivel_norm or "pregrad" in nivel_norm or "tecno" in nivel_norm or "tecni" in nivel_norm:
        return BENCHMARK_COSTO_PREGRADO
    return BENCHMARK_COSTO


def set_benchmark_costo(valor: float, nivel: str = "general") -> bool:
    global BENCHMARK_COSTO, BENCHMARK_COSTO_PREGRADO, BENCHMARK_COSTO_ESPECIALIZACION
    global BENCHMARK_COSTO_MAESTRIA, BENCHMARK_COSTO_DOCTORADO
    global BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA, BENCHMARK_COSTO_ESP_TECNOLOGICA, BENCHMARK_COSTO_ESP_TECNICO_PROF
    try:
        c = _load_config()
        nivel_norm = str(nivel).lower().strip()
        # Sub-niveles de especialización — detectar ANTES del genérico
        if "médico" in nivel_norm or "medico" in nivel_norm or "quirúrg" in nivel_norm or "quirurg" in nivel_norm:
            c["BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA"] = float(valor)
            BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA = float(valor)
        elif "tecnológ" in nivel_norm or "tecnolog" in nivel_norm:
            c["BENCHMARK_COSTO_ESP_TECNOLOGICA"] = float(valor)
            BENCHMARK_COSTO_ESP_TECNOLOGICA = float(valor)
        elif "técnico" in nivel_norm or "tecnico" in nivel_norm:
            c["BENCHMARK_COSTO_ESP_TECNICO_PROF"] = float(valor)
            BENCHMARK_COSTO_ESP_TECNICO_PROF = float(valor)
        elif "especial" in nivel_norm:
            c["BENCHMARK_COSTO_ESPECIALIZACION"] = float(valor)
            BENCHMARK_COSTO_ESPECIALIZACION = float(valor)
        elif "maestr" in nivel_norm or "magist" in nivel_norm:
            c["BENCHMARK_COSTO_MAESTRIA"] = float(valor)
            BENCHMARK_COSTO_MAESTRIA = float(valor)
        elif "doctor" in nivel_norm:
            c["BENCHMARK_COSTO_DOCTORADO"] = float(valor)
            BENCHMARK_COSTO_DOCTORADO = float(valor)
        elif "universit" in nivel_norm or "pregrad" in nivel_norm:
            c["BENCHMARK_COSTO_PREGRADO"] = float(valor)
            BENCHMARK_COSTO_PREGRADO = float(valor)
        else:
            c["BENCHMARK_COSTO"] = float(valor)
            BENCHMARK_COSTO = float(valor)
        return _save_config(c)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar benchmark en config.json: {e}")
        return False


def get_todos_benchmarks() -> dict[str, float]:
    return {
        "pregrado": BENCHMARK_COSTO_PREGRADO,
        "especializacion": BENCHMARK_COSTO_ESPECIALIZACION,
        "maestria": BENCHMARK_COSTO_MAESTRIA,
        "doctorado": BENCHMARK_COSTO_DOCTORADO,
        "esp_medico_quirurgica": BENCHMARK_COSTO_ESP_MEDICO_QUIRURGICA,
        "esp_tecnologica": BENCHMARK_COSTO_ESP_TECNOLOGICA,
        "esp_tecnico_prof": BENCHMARK_COSTO_ESP_TECNICO_PROF,
    }

# ========= INFORMACIÓN DE DEBUG =========
def print_config_info() -> None:
    """Imprime información de configuración para debugging."""
    print("=== CONFIGURACIÓN DE RUTAS ===")
    print(f"Base Path: {_BASE_PATH}")
    print(f"Outputs Dir: {OUTPUTS_DIR}")
    print(f"Historic Dir: {HISTORIC_DIR}")
    print(f"Ref Dir: {REF_DIR}")
    print(f"Models Dir: {MODELS_DIR}")
    print(f"Docs Dir: {DOCS_DIR}")
    print(f"Logs Dir: {LOGS_DIR}")
    print(f"Archivo Programas: {ARCHIVO_PROGRAMAS}")
    print(f"Ejecutándose como .EXE: {getattr(sys, 'frozen', False)}")
    print("=============================")


if __name__ == "__main__":
    # Ejecutar este script directamente para ver la configuración
    print_config_info()

