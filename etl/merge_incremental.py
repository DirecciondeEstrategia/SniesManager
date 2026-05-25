"""
Merge incremental para Estudio_Mercado_Colombia.xlsx.

Reglas:
  - Nunca borra filas del Excel existente.
  - Programas nuevos (en Programas.xlsx pero no en Excel): se agregan.
  - Programas comunes: se actualizan columnas calculadas; se respetan manuales.
  - Programas desaparecidos (en Excel pero no en Programas.xlsx): ACTIVO_PIPELINE=False.
  - Antes de modificar, guarda snapshot con fecha en historico_estudio_de_mercado/snapshots/
    (Estudio_Mercado_Colombia.xlsx y Base_Programas_Categoria_F1_*.xlsx).
"""

from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etl.config import ARCHIVO_ESTUDIO_MERCADO, ESTUDIO_MERCADO_DIR, HISTORICO_ESTUDIO_MERCADO_DIR
from etl.pipeline_logger import log_info, log_warning

# ── Rutas ─────────────────────────────────────────────────────────────────────
SNAPSHOTS_DIR = HISTORICO_ESTUDIO_MERCADO_DIR / "snapshots"
ESTUDIO_PATH = ARCHIVO_ESTUDIO_MERCADO
BASE_PROGRAMAS_GLOB = "Base_Programas_Categoria_F1_*.xlsx"

# Número de días que se conservan los snapshots. Snapshots más viejos se eliminan.
SNAPSHOT_RETENTION_DAYS = 365

# ── Identificador único de programa ───────────────────────────────────────────
ID_COL = "CÓDIGO_SNIES_DEL_PROGRAMA"

# ── Columnas que el pipeline recalcula siempre ────────────────────────────────
COLS_CALCULADAS = [
    # Matrículas SNIES (semestral + anual)
    *[f"matricula_{y}_{s}" for y in range(2019, 2026) for s in (1, 2)],
    *[f"matricula_{y}" for y in range(2019, 2026)],
    *[f"inscritos_{y}" for y in range(2019, 2026)],
    *[f"primer_curso_{y}" for y in range(2019, 2026)],
    *[f"graduados_{y}" for y in range(2019, 2026)],
    "ACREDITADA_ALTA_CALIDAD",
    "tiene_matricula_2024",
    # OLE
    "SALARIO_OLE",
    "TASA_COTIZANTES",
    "FUENTE_OLE",
    # Scoring
    "score_matricula",
    "score_participacion",
    "score_AAGR",
    "score_salario",
    "score_pct_no_matriculados",
    "score_num_programas",
    "score_costo",
    "calificacion_final",
    # Flags derivados
    "ACTIVO_PIPELINE",
    "nuevo_en_snies_3a",
    "nuevo_vs_snapshot_anterior",
    "PROBABILIDAD",
    "COSTO_MATRÍCULA_ESTUD_NUEVOS",
    "COSTO_IMPUTADO_MEDIANA",
]

# ── Columnas descriptivas SNIES (se actualizan cuando el programa reaparece) ──
COLS_DESCRIPTIVAS_SNIES = [
    "NOMBRE_DEL_PROGRAMA",
    "TITULO_OTORGADO",
    "ESTADO_PROGRAMA",
    "NIVEL_ACADÉMICO",
    "NIVEL_DE_FORMACIÓN",
    "MODALIDAD",
    "ÁREA_DE_CONOCIMIENTO",
    "NÚCLEO_BÁSICO_DEL_CONOCIMIENTO",
    "DEPARTAMENTO_OFERTA_PROGRAMA",
    "MUNICIPIO_OFERTA_PROGRAMA",
    "NOMBRE_INSTITUCIÓN",
    "CARÁCTER_ACADÉMICO",
    "SECTOR",
    "PERIODICIDAD",
    "NÚMERO_CRÉDITOS",
    "NÚMERO_PERIODOS_DE_DURACIÓN",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _guardar_snapshot(path: Path) -> None:
    """
    Copia el estudio actual y los Excel Base_Programas F1 a historico/snapshots/.
    Luego elimina snapshots con más de SNAPSHOT_RETENTION_DAYS días de antigüedad.
    """
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    fecha = date.today().isoformat()
    hora: str | None = None

    def _destino_unico(nombre: str) -> Path:
        nonlocal hora
        dest = SNAPSHOTS_DIR / nombre
        if dest.exists():
            if hora is None:
                hora = datetime.now().strftime("%H%M")
            stem, suffix = dest.stem, dest.suffix
            dest = SNAPSHOTS_DIR / f"{stem}_{hora}{suffix}"
        return dest

    # Estudio_Mercado_Colombia.xlsx
    if path.exists():
        dest = _destino_unico(f"Estudio_Mercado_{fecha}.xlsx")
        shutil.copy2(path, dest)
        log_info(f"[Merge] Snapshot guardado: {dest.name}")

    # Base_Programas_Categoria_F1_*.xlsx (export Fase 1 en estudio_de_mercado/)
    n_base = 0
    for src in sorted(ESTUDIO_MERCADO_DIR.glob(BASE_PROGRAMAS_GLOB)):
        dest = _destino_unico(src.name)
        shutil.copy2(src, dest)
        log_info(f"[Merge] Snapshot Base F1: {dest.name}")
        n_base += 1
    if n_base == 0:
        log_info("[Merge] No hay archivos Base_Programas_Categoria_F1_*.xlsx para archivar.")
    else:
        log_info(f"[Merge] {n_base} archivo(s) Base_Programas archivados en snapshots/.")

    _limpiar_snapshots_antiguos()


def _limpiar_snapshots_antiguos() -> None:
    """Elimina snapshots con más de SNAPSHOT_RETENTION_DAYS días de antigüedad."""
    if not SNAPSHOTS_DIR.exists():
        return

    hoy = date.today()
    eliminados = 0
    errores = 0

    for patron in ("Estudio_Mercado_*.xlsx", BASE_PROGRAMAS_GLOB):
        for f in SNAPSHOTS_DIR.glob(patron):
            try:
                edad_dias = (hoy - date.fromtimestamp(f.stat().st_mtime)).days
                if edad_dias > SNAPSHOT_RETENTION_DAYS:
                    f.unlink()
                    eliminados += 1
            except Exception as e:
                log_warning(f"[Merge] No se pudo eliminar snapshot {f.name}: {e}")
                errores += 1

    if eliminados > 0:
        log_info(
            f"[Merge] Retención de snapshots: {eliminados} archivo(s) eliminado(s) "
            f"(>{SNAPSHOT_RETENTION_DAYS} días). Errores: {errores}."
        )


def _calcular_activo_pipeline(df: pd.DataFrame) -> pd.Series:
    """
    ACTIVO_PIPELINE = True si tiene matriculados > 0 en el año más reciente
    disponible, O si ESTADO_PROGRAMA == 'activo'.
    Corrige el problema de programas 'inactivos' con matrícula real.
    """
    mat_col = next(
        (f"matricula_{y}" for y in range(2025, 2018, -1) if f"matricula_{y}" in df.columns),
        None,
    )
    tiene_mat = (df[mat_col].fillna(0) > 0) if mat_col else pd.Series(False, index=df.index)
    estado_ok = (
        df["ESTADO_PROGRAMA"].astype(str).str.lower().str.strip() == "activo"
        if "ESTADO_PROGRAMA" in df.columns
        else pd.Series(False, index=df.index)
    )
    return (tiene_mat | estado_ok).astype(bool)


def _calcular_nuevo_en_snies_3a(fecha_serie: pd.Series) -> pd.Series:
    """nuevo_en_snies_3a = True si FECHA_PRIMERA_VEZ >= hoy - 3 años."""
    hoy = pd.Timestamp.today().normalize()
    hace_3 = hoy - pd.DateOffset(years=3)
    return pd.to_datetime(fecha_serie, errors="coerce") >= hace_3


def _ensure_columns(dst: pd.DataFrame, src: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Asegura que dst tenga las columnas listadas si existen en src."""
    for c in cols:
        if c in src.columns and c not in dst.columns:
            dst[c] = pd.NA
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def merge_incremental(
    nuevo: pd.DataFrame,
    nuevo_total: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | None]:
    """
    Aplica merge incremental entre Excel existente y resultado nuevo del pipeline.

    Parámetros:
      nuevo       : DataFrame de programas_detalle calculado en esta ejecución.
      nuevo_total : DataFrame de la hoja total calculado en esta ejecución.

    Retorna:
      {'programas_detalle': df_merged, 'total': df_total_ajustado}
    """
    hoy = pd.Timestamp.today().normalize()

    # 1. Snapshot ──────────────────────────────────────────────────────────────
    _guardar_snapshot(ESTUDIO_PATH)

    # 2. Normalizar ID en nuevo ────────────────────────────────────────────────
    nuevo = nuevo.copy()
    # Renombres de columnas (compatibilidad + claridad semántica)
    if "ES_PROGRAMA_NUEVO" in nuevo.columns and "nuevo_vs_snapshot_anterior" not in nuevo.columns:
        nuevo = nuevo.rename(columns={"ES_PROGRAMA_NUEVO": "nuevo_vs_snapshot_anterior"})
    if ID_COL not in nuevo.columns:
        raise ValueError(f"[Merge] Falta columna ID '{ID_COL}' en DataFrame nuevo.")
    nuevo[ID_COL] = nuevo[ID_COL].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # Protección contra SNIES duplicados: si el mismo código aparece más de una vez
    # en el nuevo DataFrame (ej. programa en convenio duplicado), keep='last'
    # conserva la fila más reciente. Sin esto, .update() lanza ValueError.
    n_antes = len(nuevo)
    nuevo = nuevo.drop_duplicates(subset=[ID_COL], keep="last")
    if len(nuevo) < n_antes:
        log_warning(
            f"[Merge] {n_antes - len(nuevo)} filas duplicadas por {ID_COL} eliminadas del nuevo "
            f"(keep='last'). Verificar Programas.xlsx si el número es alto."
        )

    # 3. Cargar Excel existente ────────────────────────────────────────────────
    existente = pd.DataFrame()
    if ESTUDIO_PATH.exists():
        try:
            existente = pd.read_excel(ESTUDIO_PATH, sheet_name="programas_detalle", dtype={ID_COL: str})
            existente[ID_COL] = (
                existente[ID_COL].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            )
            # Renombres para Excel legado
            if "ES_PROGRAMA_NUEVO" in existente.columns and "nuevo_vs_snapshot_anterior" not in existente.columns:
                existente = existente.rename(columns={"ES_PROGRAMA_NUEVO": "nuevo_vs_snapshot_anterior"})
            log_info(f"[Merge] Excel existente: {len(existente):,} programas")
        except Exception as e:
            log_warning(f"[Merge] No se pudo leer Excel existente ({e}). Creando desde cero.")
            existente = pd.DataFrame()

    # 4. Primera ejecución ─────────────────────────────────────────────────────
    if existente.empty:
        log_info("[Merge] Primera ejecución — inicializando columnas de control.")
        nuevo["FECHA_PRIMERA_VEZ"] = (
            pd.to_datetime(nuevo["FECHA_DE_REGISTRO_EN_SNIES"], errors="coerce")
            if "FECHA_DE_REGISTRO_EN_SNIES" in nuevo.columns
            else pd.Series(hoy, index=nuevo.index)
        ).fillna(hoy)
        nuevo["FECHA_ULTIMO_ACTIVO"] = hoy
        nuevo["ACTIVO_PIPELINE"] = _calcular_activo_pipeline(nuevo)
        nuevo["nuevo_en_snies_3a"] = _calcular_nuevo_en_snies_3a(nuevo["FECHA_PRIMERA_VEZ"])
        if "nuevo_vs_snapshot_anterior" not in nuevo.columns:
            nuevo["nuevo_vs_snapshot_anterior"] = False
        total_final = _ajustar_totales(nuevo, nuevo_total)
        _log_resumen(nuevo)
        return {"programas_detalle": nuevo, "total": total_final}

    # Asegurar que existente tenga columnas del nuevo (para no perder columnas nuevas)
    existente = _ensure_columns(
        existente,
        nuevo,
        cols=(
            COLS_CALCULADAS
            + COLS_DESCRIPTIVAS_SNIES
            + ["CATEGORIA_FINAL", "FUENTE_CATEGORIA", "REQUIERE_REVISION", "schema_version"]
        ),
    )

    # 5. Inicializar columnas de control si no existen en existente ────────────
    for col, default in [
        ("FECHA_PRIMERA_VEZ", hoy),
        ("FECHA_ULTIMO_ACTIVO", hoy),
        ("ACTIVO_PIPELINE", True),
        ("nuevo_en_snies_3a", False),
        ("nuevo_vs_snapshot_anterior", False),
    ]:
        if col not in existente.columns:
            existente[col] = default

    # 6. Calcular diferencias ──────────────────────────────────────────────────
    ids_exist = set(existente[ID_COL].dropna().astype(str))
    ids_nuevos = set(nuevo[ID_COL].dropna().astype(str))

    nuevos_ids = ids_nuevos - ids_exist
    comunes = ids_nuevos & ids_exist
    desaparecidos = ids_exist - ids_nuevos

    log_info(
        f"[Merge] Nuevos={len(nuevos_ids):,} | Comunes={len(comunes):,} | Desaparecidos={len(desaparecidos):,}"
    )

    # 7. Desaparecidos → inactivos (NO borrar) ─────────────────────────────────
    mask_desap = existente[ID_COL].isin(desaparecidos)
    existente.loc[mask_desap, "ACTIVO_PIPELINE"] = False
    existente.loc[mask_desap, "FECHA_ULTIMO_ACTIVO"] = hoy
    log_info(f"[Merge] {int(mask_desap.sum()):,} programas marcados ACTIVO_PIPELINE=False")

    # 8. Comunes → actualizar columnas calculadas (vectorizado) ───────────────
    cols_actualizar = [
        c
        for c in COLS_CALCULADAS + COLS_DESCRIPTIVAS_SNIES
        if c in nuevo.columns and c in existente.columns
    ]

    # Proteger CATEGORIA_FINAL y FUENTE_CATEGORIA de programas con edición manual
    # Los programas MANUAL se excluyen poniendo NaN en esas columnas del nuevo,
    # de modo que DataFrame.update() los salte (solo actualiza con non-NaN).
    snies_manuales: set[str] = set()
    if "FUENTE_CATEGORIA" in existente.columns:
        snies_manuales = set(
            existente.loc[
                existente["FUENTE_CATEGORIA"].astype(str).str.upper().str.strip() == "MANUAL",
                ID_COL,
            ].astype(str)
        )

    nuevo_para_update = nuevo[
        [ID_COL]
        + cols_actualizar
        + [c for c in ["CATEGORIA_FINAL", "FUENTE_CATEGORIA"] if c in nuevo.columns]
    ].copy()

    if snies_manuales and "CATEGORIA_FINAL" in nuevo_para_update.columns:
        mask_manual = nuevo_para_update[ID_COL].astype(str).isin(snies_manuales)
        nuevo_para_update.loc[mask_manual, "CATEGORIA_FINAL"] = np.nan
        if "FUENTE_CATEGORIA" in nuevo_para_update.columns:
            nuevo_para_update.loc[mask_manual, "FUENTE_CATEGORIA"] = np.nan

    nuevo_para_update = nuevo_para_update.set_index(ID_COL)

    existente = existente.set_index(ID_COL)
    existente.update(nuevo_para_update)  # sobreescribe solo donde nuevo tiene non-NaN
    existente = existente.reset_index()

    # Recalcular ACTIVO_PIPELINE y FECHA_ULTIMO_ACTIVO para los programas comunes
    mask_comunes = existente[ID_COL].isin(comunes)
    existente.loc[mask_comunes, "ACTIVO_PIPELINE"] = _calcular_activo_pipeline(
        existente.loc[mask_comunes]
    ).values

    # Actualizar FECHA_ULTIMO_ACTIVO solo para los que quedaron activos
    mask_ahora_activos = mask_comunes & existente["ACTIVO_PIPELINE"].astype(bool)
    existente.loc[mask_ahora_activos, "FECHA_ULTIMO_ACTIVO"] = hoy

    log_info(f"[Merge] {len(comunes):,} programas comunes actualizados")

    # 9. Nuevos → agregar ──────────────────────────────────────────────────────
    if nuevos_ids:
        df_nuevos = nuevo[nuevo[ID_COL].isin(nuevos_ids)].copy()
        df_nuevos["FECHA_PRIMERA_VEZ"] = (
            pd.to_datetime(df_nuevos["FECHA_DE_REGISTRO_EN_SNIES"], errors="coerce")
            if "FECHA_DE_REGISTRO_EN_SNIES" in df_nuevos.columns
            else pd.Series(hoy, index=df_nuevos.index)
        ).fillna(hoy)
        df_nuevos["FECHA_ULTIMO_ACTIVO"] = hoy
        df_nuevos["ACTIVO_PIPELINE"] = _calcular_activo_pipeline(df_nuevos)
        df_nuevos["nuevo_en_snies_3a"] = True
        if "nuevo_vs_snapshot_anterior" not in df_nuevos.columns:
            df_nuevos["nuevo_vs_snapshot_anterior"] = True
        if "FUENTE_CATEGORIA" not in df_nuevos.columns:
            df_nuevos["FUENTE_CATEGORIA"] = "PIPELINE"
        existente = pd.concat([existente, df_nuevos], ignore_index=True)
        log_info(f"[Merge] {len(df_nuevos):,} programas nuevos agregados")

    # 10. Recalcular nuevo_en_snies_3a para todos ──────────────────────────────
    existente["nuevo_en_snies_3a"] = _calcular_nuevo_en_snies_3a(existente["FECHA_PRIMERA_VEZ"])
    if "nuevo_vs_snapshot_anterior" not in existente.columns:
        existente["nuevo_vs_snapshot_anterior"] = False

    # 11. Ajustar hoja total ───────────────────────────────────────────────────
    total_final = _ajustar_totales(existente, nuevo_total)

    _log_resumen(existente)
    return {"programas_detalle": existente, "total": total_final}


def _ajustar_totales(detalle: pd.DataFrame, total_nuevo: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Recalcula programas_activos, programas_inactivos y programas_nuevos_3a
    en la hoja total usando el detalle merged (incluye inactivos históricos).
    """
    if total_nuevo is None:
        return None

    # Edge case: total_nuevo vacío y sin columnas (no hay llave para mergear).
    # Devolver tal cual para no romper el pipeline.
    if isinstance(total_nuevo, pd.DataFrame) and len(total_nuevo.columns) == 0:
        return total_nuevo

    if "ACTIVO_PIPELINE" not in detalle.columns or "CATEGORIA_FINAL" not in detalle.columns:
        return total_nuevo

    # Validación defensiva: columnas de "nuevos" pueden no existir en Excel legado o en detalles parciales.
    if "nuevo_en_snies_3a" not in detalle.columns:
        detalle = detalle.copy()
        detalle["nuevo_en_snies_3a"] = False
    if "nuevo_vs_snapshot_anterior" not in detalle.columns:
        detalle = detalle.copy()
        detalle["nuevo_vs_snapshot_anterior"] = False

    cat_col = str(total_nuevo.columns[0]) if len(total_nuevo.columns) else "CATEGORIA_FINAL"
    if cat_col not in total_nuevo.columns:
        log_warning(
            f"[Merge] No se pudo ajustar totales: la hoja 'total' no tiene columna llave '{cat_col}'. "
            "Se deja la hoja total sin ajustes de activos/inactivos/nuevos."
        )
        return total_nuevo
    conteos = (
        detalle.groupby("CATEGORIA_FINAL")
        .agg(
            _activos=("ACTIVO_PIPELINE", lambda s: s.astype(bool).sum()),
            _inactivos=("ACTIVO_PIPELINE", lambda s: (~s.astype(bool)).sum()),
            _nuevos_snies=(
                "nuevo_en_snies_3a",
                lambda s: s.astype(bool).sum() if "nuevo_en_snies_3a" in detalle.columns else 0,
            ),
            _nuevos_snap=(
                "nuevo_vs_snapshot_anterior",
                lambda s: s.astype(bool).sum() if "nuevo_vs_snapshot_anterior" in detalle.columns else 0,
            ),
        )
        .reset_index()
        .rename(columns={"CATEGORIA_FINAL": cat_col})
    )

    total_upd = total_nuevo.merge(conteos, on=cat_col, how="left")

    for dest, src in [
        ("programas_activos", "_activos"),
        ("programas_inactivos", "_inactivos"),
        ("programas_nuevos_3a", "_nuevos_snies"),
        ("nuevos_vs_snapshot", "_nuevos_snap"),
    ]:
        if dest in total_upd.columns:
            total_upd[dest] = total_upd[src].fillna(0).astype(int)

    return total_upd.drop(
        columns=["_activos", "_inactivos", "_nuevos_snies", "_nuevos_snap"],
        errors="ignore",
    )


def _log_resumen(df: pd.DataFrame) -> None:
    if "ACTIVO_PIPELINE" in df.columns:
        activos = int(df["ACTIVO_PIPELINE"].astype(bool).sum())
        inactivos = int((~df["ACTIVO_PIPELINE"].astype(bool)).sum())
    else:
        activos = -1
        inactivos = -1
    nuevos = int(df["nuevo_en_snies_3a"].astype(bool).sum()) if "nuevo_en_snies_3a" in df.columns else -1
    log_info(
        f"[Merge] Resultado final: {len(df):,} programas totales | "
        f"Activos={activos:,} | Inactivos={inactivos:,} | Nuevos (3a)={nuevos:,}"
    )

