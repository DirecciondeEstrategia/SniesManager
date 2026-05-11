"""
Lógica de calificación ponderada por categoría (Fase 4 pipeline mercado).

Cada variable se mapea a score 1-5 según umbrales; calificacion_final = suma(score_i × peso_i).

score_matricula: modo_local=False → quintiles fijos nacionales (P20/P40/P60/P80) sobre Colombia.
                 modo_local=True  → quintiles dinámicos del segmento (regional/modal).

score_participacion: quintiles por **cuantiles del segmento** actual (``participacion_2024``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from etl.pipeline_logger import log_info

# Pesos de las dos métricas con lógica propia (deben coincidir con el resto de SCORING_CONFIG).
_SCORE_MATRICULA_PESO = 0.30
_SCORE_PARTICIPACION_PESO = 0.15

# score_matricula — percentiles reales de prom_primer_curso_2024 sobre universo
# POSGRADO-only (ESP+MAE, 288 categorías Colombia). Recalibrado tras excluir
# programas UNIVERSITARIO del cómputo por categoría.
# Distribución resultante: ~57/58/57/58/58 categorías por score (quintílica).
_SCORE_MAT_P20: float = 3.0
_SCORE_MAT_P40: float = 5.4
_SCORE_MAT_P60: float = 8.5
_SCORE_MAT_P80: float = 13.8

# score_matricula — PREGRADO (prom_primer_curso_2024 por categoría, 144 cats Colombia)
# Universo: programas UNIVERSITARIO dentro de las 144 categorías que los contienen.
# Distribución real: P20=22 P40=34 P60=49 P80=83 (~10× mayor que posgrado)
_SCORE_MAT_PRE_P20: float = 22.0
_SCORE_MAT_PRE_P40: float = 34.0
_SCORE_MAT_PRE_P60: float = 49.5
_SCORE_MAT_PRE_P80: float = 83.0

# score_AAGR — árbol de decisión para POSGRADO ─────────────────────────────────
# ESP (categorías con NIVEL_MAYORIT ∈ {ESPECIALIZACIÓN, ESP.MED.QUIR, ESP.TEC, ESP.TEC.PRO})
# Distribución AAGR_ROBUSTO ESP: P20=0.8% P40=5.8% P60=10.3% P80=18.0%
_AAGR_ESP_THRESHOLDS: list[tuple[float, int]] = [
    (0.008, 1), (0.058, 2), (0.103, 3), (0.180, 4)
]
# MAE: más volátil, 33% con AAGR negativo
# Distribución: P20=-2.1% P40=3.2% P60=7.4% P80=16.3%
_AAGR_MAE_THRESHOLDS: list[tuple[float, int]] = [
    (-0.021, 1), (0.032, 2), (0.074, 3), (0.163, 4)
]
# PREGRADO: mercado maduro, distribución centrada en crecimiento bajo
# Distribución: P20=-1.9% P40=0.8% P60=3.4% P80=8.2%
_AAGR_PRE_THRESHOLDS: list[tuple[float, int]] = [
    (-0.019, 1), (0.008, 2), (0.034, 3), (0.082, 4)
]

# Umbrales: lista de (límite_superior_inclusivo, score). Valores por encima del último → score 5 (o 1 si inverse).
# Para "inverse" (menor es mejor), se usa score 5 para el rango más bajo.
SCORING_CONFIG = [
    {
        "col": "AAGR_ROBUSTO",
        "out": "score_AAGR",
        "peso": 0.20,
        # Percentiles P20/P40/P60/P80 de AAGR_ROBUSTO (primer_curso) Colombia 288 cats
        # Antes (total mat): 0% / 4% / 19% / 30%  ← calibrado para stock, no flujo
        # Ahora (primer_curso): P80 real = 16.5%  (crecimiento de flujo más volátil)
        "thresholds": [(0.00, 1), (0.038, 2), (0.084, 3), (0.165, 4)],
        "inverse": False,
    },
    {
        "col": "salario_promedio_smlmv",
        "out": "score_salario",
        "peso": 0.15,
        # Alineado con Excel referencia manual: score 5 = >= 8 SMLMV (médicos, TI senior)
        # Rango real Colombia: P20=3.2 / P40=4.7 / P60=5.5 / P80=7.0 SMLMV
        "thresholds": [(2, 1), (3, 2), (5, 3), (8, 4)],
        "inverse": False,
    },
    {
        "col": "pct_no_matriculados_2024",
        "out": "score_pct_no_matriculados",
        "peso": 0.10,
        "thresholds": [(0.10, 5), (0.20, 4), (0.30, 3), (0.50, 2)],
        "inverse": True,
    },
    {
        "col": "num_programas_2024",
        "out": "score_num_programas",
        "peso": 0.05,
        # Percentiles reales posgrado-only (ESP+MAE, 288 cats):
        # P20=4 · P40=10 · P60=18 · P80=32 (media=21 progs/cat)
        # P60 actualizado de 25→18, P80 actualizado de 55→32
        "thresholds": [(4, 5), (10, 4), (18, 3), (32, 2)],
        "inverse": True,
    },
    {
        "col": "distancia_costo_pct",
        "out": "score_costo",
        "peso": 0.05,
        "thresholds": [(-60, 1), (-40, 2), (-15, 3), (20, 4)],
        "inverse": False,
    },
]


# ── Configuración de scoring para universo PREGRADO ──────────────────────────
# Todos los thresholds calibrados sobre 144 categorías con programas UNIVERSITARIO.
# Pesos idénticos al posgrado para mantener comparabilidad de estructura.
SCORING_CONFIG_PREGRADO: list[dict] = [
    {
        "col": "AAGR_ROBUSTO",
        "out": "score_AAGR",
        "peso": 0.20,
        # Mercado maduro: P20=-1.9% P40=0.8% P60=3.4% P80=8.2%
        "thresholds": _AAGR_PRE_THRESHOLDS,
        "inverse": False,
    },
    {
        "col": "salario_promedio_smlmv",
        "out": "score_salario",
        "peso": 0.15,
        # P20=2.76 P40=3.30 P60=3.90 P80=5.28 SMLMV (escala entrada laboral)
        "thresholds": [(2.0, 1), (2.76, 2), (3.30, 3), (3.90, 4)],
        "inverse": False,
    },
    {
        "col": "pct_no_matriculados_2024",
        "out": "score_pct_no_matriculados",
        "peso": 0.10,
        # P20=0.27 P40=0.34 P60=0.41 P80=0.48 (similar a posgrado, escala relativa)
        "thresholds": [(0.27, 5), (0.34, 4), (0.41, 3), (0.48, 2)],
        "inverse": True,
    },
    {
        "col": "num_programas_2024",
        "out": "score_num_programas",
        "peso": 0.05,
        # P20=2 P40=7 P60=25 P80=57 (distribución más polarizada que posgrado)
        "thresholds": [(2, 5), (7, 4), (25, 3), (57, 2)],
        "inverse": True,
    },
    {
        "col": "distancia_costo_pct",
        "out": "score_costo",
        "peso": 0.05,
        # Mismo threshold: distancia relativa ya ajusta por nivel via BENCHMARK_COSTO_PREGRADO
        "thresholds": [(-60, 1), (-40, 2), (-15, 3), (20, 4)],
        "inverse": False,
    },
]


def _value_to_score(value: float, thresholds: list[tuple[float, int]], inverse: bool) -> float:
    """
    Mapea un valor numérico a score 1-5 según umbrales.
    thresholds: list of (upper_bound_inclusive, score).
    inverse: si True, por debajo del primer umbral se asigna el score del primer umbral (mejor).
    """
    if pd.isna(value) or (isinstance(value, float) and np.isinf(value)):
        return 1.0
    if inverse:
        for bound, score in thresholds:
            if value <= bound:
                return float(score)
        return 1.0
    for bound, score in thresholds:
        if value <= bound:
            return float(score)
    return 5.0


def apply_scoring(
    df: pd.DataFrame,
    modo_local: bool = False,
    universo: str = "posgrado",
) -> pd.DataFrame:
    """
    Aplica la calificación ponderada a un DataFrame con las columnas esperadas por SCORING_CONFIG.
    Añade columnas score_* y calificacion_final. NaN en alguna variable no rompe; se usa score 1.
    calificacion_final está siempre entre 1.0 y 5.0.

    Args:
        df: DataFrame con las columnas de métricas agregadas por categoría.
        modo_local: Si True, score_matricula usa quintiles calculados sobre el propio
                    DataFrame (para segmentos regionales/modales). Si False (default),
                    usa los thresholds nacionales fijos _SCORE_MAT_P20.._P80,
                    garantizando comparabilidad cross-segmento a nivel Colombia.
        universo:   "posgrado" (default) usa SCORING_CONFIG y thresholds de matrícula
                    posgrado. "pregrado" usa SCORING_CONFIG_PREGRADO y thresholds de
                    matrícula pregrado (~10× mayor). En posgrado, score_AAGR aplica
                    árbol de decisión por NIVEL_MAYORIT (ESP vs MAE).
    """
    out = df.copy()
    scoring_config = SCORING_CONFIG_PREGRADO if universo == "pregrado" else SCORING_CONFIG
    total_peso = (
        _SCORE_MATRICULA_PESO
        + _SCORE_PARTICIPACION_PESO
        + sum(c["peso"] for c in scoring_config)
    )
    if abs(total_peso - 1.0) > 1e-9:
        raise ValueError(
            f"Los pesos deben sumar 1.0, suman {total_peso} (universo={universo})"
        )

    # ── DIAGNÓSTICO DE DISTRIBUCIÓN ───────────────────────────────────────────
    _cols_diag = {
        "prom_primer_curso_2024": "PRIMER_CURSO",
        "prom_matricula_por_programa_2024": "MAT",
        "prom_matricula_2024": "MAT_PROM",
        "participacion_2024": "PART",
        "AAGR_ROBUSTO": "AAGR",
        "salario_promedio_smlmv": "SAL",
    }
    for _col, _lbl in _cols_diag.items():
        if _col in out.columns:
            _s = pd.to_numeric(out[_col], errors="coerce").dropna()
            if len(_s) > 0:
                _pcts = np.percentile(_s, [10, 20, 40, 60, 80, 90])
                log_info(
                    f"[Scoring diag] {_lbl} n={len(_s)} | "
                    f"P10={_pcts[0]:.4g} P20={_pcts[1]:.4g} P40={_pcts[2]:.4g} "
                    f"P60={_pcts[3]:.4g} P80={_pcts[4]:.4g} P90={_pcts[5]:.4g}"
                )

    # Con inscritos SNIES (fuente primaria desde 2025), la cobertura es >80%.
    PCT_NAN_FILL = 0.25
    if "pct_no_matriculados_2024" in out.columns:
        out["pct_no_matriculados_2024"] = out["pct_no_matriculados_2024"].fillna(PCT_NAN_FILL)

    # score_matricula — thresholds nacionales fijos (modo_local=False)
    #                   o quintiles locales dinámicos (modo_local=True).
    if "prom_matricula_por_programa_2024" in out.columns:
        _s_mat = pd.to_numeric(out["prom_matricula_por_programa_2024"], errors="coerce")
    elif "prom_matricula_2024" in out.columns:
        _s_mat = pd.to_numeric(out["prom_matricula_2024"], errors="coerce")
    else:
        _s_mat = None

    if _s_mat is not None:
        if modo_local:
            # Quintiles dinámicos sobre el segmento — misma lógica que score_participacion
            _mat_valid = _s_mat.dropna()
            _lp20 = float(_mat_valid.quantile(0.20))
            _lp40 = float(_mat_valid.quantile(0.40))
            _lp60 = float(_mat_valid.quantile(0.60))
            _lp80 = float(_mat_valid.quantile(0.80))
            # Guard contra bins duplicados cuando el segmento es pequeño
            _lqs = [_lp20, _lp40, _lp60, _lp80]
            for _i in range(1, len(_lqs)):
                if _lqs[_i] <= _lqs[_i - 1]:
                    _lqs[_i] = _lqs[_i - 1] + 1e-9
            _lp20, _lp40, _lp60, _lp80 = _lqs
            _bins = [-np.inf, _lp20, _lp40, _lp60, _lp80, np.inf]
            log_info(
                f"[Scoring] score_matricula LOCAL → "
                f"P20={_lp20:.2f} P40={_lp40:.2f} P60={_lp60:.2f} P80={_lp80:.2f}"
            )
        else:
            # Thresholds nacionales fijos calibrados sobre Colombia, separados por universo
            if universo == "pregrado":
                _p20, _p40, _p60, _p80 = (
                    _SCORE_MAT_PRE_P20,
                    _SCORE_MAT_PRE_P40,
                    _SCORE_MAT_PRE_P60,
                    _SCORE_MAT_PRE_P80,
                )
                log_info(
                    f"[Scoring] score_matricula PREGRADO → "
                    f"P20={_p20} P40={_p40} P60={_p60} P80={_p80}"
                )
            else:
                _p20, _p40, _p60, _p80 = (
                    _SCORE_MAT_P20,
                    _SCORE_MAT_P40,
                    _SCORE_MAT_P60,
                    _SCORE_MAT_P80,
                )
            _bins = [-np.inf, _p20, _p40, _p60, _p80, np.inf]

        _cat_mat = pd.cut(
            _s_mat,
            bins=_bins,
            labels=[1, 2, 3, 4, 5],
            right=True,
        )
        out["score_matricula"] = (
            pd.to_numeric(_cat_mat, errors="coerce").fillna(1.0).astype(int)
        )
    else:
        out["score_matricula"] = 1

    # score_participacion — cuantiles del segmento en curso (mercado regional relativo)
    if "participacion_2024" in out.columns:
        _part_series = pd.to_numeric(out["participacion_2024"], errors="coerce")
        _p20 = float(_part_series.quantile(0.20))
        _p40 = float(_part_series.quantile(0.40))
        _p60 = float(_part_series.quantile(0.60))
        _p80 = float(_part_series.quantile(0.80))
        if any(map(pd.isna, (_p20, _p40, _p60, _p80))):
            out["score_participacion"] = 1
        else:
            _qs = [_p20, _p40, _p60, _p80]
            for _i in range(1, len(_qs)):
                if _qs[_i] <= _qs[_i - 1]:
                    _qs[_i] = _qs[_i - 1] + 1e-15
            _p20, _p40, _p60, _p80 = _qs
            _bins_p = [-np.inf, _p20, _p40, _p60, _p80, np.inf]
            _cat_p = pd.cut(
                _part_series,
                bins=_bins_p,
                labels=[1, 2, 3, 4, 5],
                right=True,
            )
            out["score_participacion"] = (
                pd.to_numeric(_cat_p, errors="coerce").fillna(1.0).astype(int)
            )
    else:
        out["score_participacion"] = 1

    for cfg in scoring_config:
        col = cfg["col"]
        out_col = cfg["out"]
        thresholds = cfg["thresholds"]
        inverse = cfg.get("inverse", False)

        if col not in out.columns:
            out[out_col] = 1.0
            continue

        # Árbol de decisión AAGR: thresholds distintos para ESP vs MAE (solo posgrado).
        # Pregrado cae al else estándar usando _AAGR_PRE_THRESHOLDS dentro de SCORING_CONFIG_PREGRADO.
        if (
            col == "AAGR_ROBUSTO"
            and universo == "posgrado"
            and "NIVEL_MAYORIT" in out.columns
        ):
            def _score_aagr_nivel(
                row,
                _esp=_AAGR_ESP_THRESHOLDS,
                _mae=_AAGR_MAE_THRESHOLDS,
            ):
                nivel = str(row.get("NIVEL_MAYORIT", "")).upper()
                val = row.get("AAGR_ROBUSTO", np.nan)
                # MAE o ESP (todos los sub-tipos: ESPECIALIZACIÓN, ESP.MED.QUIR, ESP.TEC, ESP.TEC.PRO)
                thrs = _mae if "MAEST" in nivel else _esp
                v = float(val) if pd.notna(val) else np.nan
                return _value_to_score(v, thrs, inverse=False)

            out[out_col] = out.apply(_score_aagr_nivel, axis=1)
        else:
            out[out_col] = out[col].apply(
                lambda v: _value_to_score(float(v) if pd.notna(v) else np.nan, thresholds, inverse)
            )

    out["calificacion_final"] = 0.0
    out["calificacion_final"] += out["score_matricula"].astype(float) * _SCORE_MATRICULA_PESO
    out["calificacion_final"] += out["score_participacion"].astype(float) * _SCORE_PARTICIPACION_PESO
    for cfg in scoring_config:
        out["calificacion_final"] += out[cfg["out"]] * cfg["peso"]
    out["calificacion_final"] = out["calificacion_final"].clip(1.0, 5.0).round(4)

    for _sc in ["score_matricula", "score_participacion", "score_AAGR", "score_salario"]:
        if _sc in out.columns:
            _dist = out[_sc].value_counts().sort_index().to_dict()
            log_info(f"[Scoring val/{universo}] {_sc}: {_dist}")

    return out
