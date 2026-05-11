"""
Modelo de clasificación supervisado para identificar programas nuevos que son referentes
de EAFIT o competencia directa.

El modelo se entrena con referentesUnificados.xlsx y clasifica programas nuevos del
archivo Programas.xlsx que tienen PROGRAMA_NUEVO = Sí.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importación diferida de sentence_transformers para evitar errores al importar el módulo
_SentenceTransformer = None

def _get_sentence_transformer():
    """
    Obtiene SentenceTransformer con importación diferida.
    
    Returns:
        Clase SentenceTransformer
        
    Raises:
        ImportError: Si sentence_transformers no está disponible o está corrupto
    """
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
        except ImportError as e:
            error_msg = (
                f"No se pudo importar sentence_transformers. "
                f"Esto puede deberse a una instalación corrupta. "
                f"Error: {e}. "
                f"Intente reinstalar: pip install --no-cache-dir sentence-transformers"
            )
            log_error(error_msg)
            raise ImportError(error_msg) from e
    return _SentenceTransformer

from etl.pipeline_logger import log_error, log_info, log_resultado, log_warning
from etl.config import (
    REF_DIR,
    MODELS_DIR,
    ARCHIVO_REFERENTES,
    ARCHIVO_CATALOGO_EAFIT,
    ARCHIVO_PROGRAMAS,
    HOJA_PROGRAMAS,
    UMBRAL_REFERENTE,
    leer_datos_flexible,
    get_archivo_referentes,
    get_archivo_catalogo_eafit,
)

# ========= CONFIG =========
# Modelo de embeddings (multilingüe para español)
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

# Rutas de modelos guardados
MODELO_CLASIFICADOR = MODELS_DIR / "clasificador_referentes.pkl"
MODELO_EMBEDDINGS_OBJ = MODELS_DIR / "modelo_embeddings.pkl"
ENCODER_PROGRAMAS_EAFIT = MODELS_DIR / "encoder_programas_eafit.pkl"
# =========================


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparación: minúsculas, sin tildes, sin caracteres especiales.
    
    Args:
        texto: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    if pd.isna(texto):
        return ""
    
    texto = str(texto).lower().strip()
    # Eliminar espacios múltiples
    texto = " ".join(texto.split())
    return texto


def normalizar_nivel_formacion(nivel: str) -> str:
    if pd.isna(nivel):
        return ""
    nivel_norm = normalizar_texto(str(nivel))
    if any(x in nivel_norm for x in ['especial', 'especializ']):
        return 'especializacion universitaria'
    if any(x in nivel_norm for x in ['universit', 'pregrad', 'pregra']):
        return 'universitario'
    if any(x in nivel_norm for x in ['maestr', 'magist', 'master']):
        return 'maestria'
    if any(x in nivel_norm for x in ['doctor', 'phd']):
        return 'doctorado'
    return ""


def niveles_coinciden(nivel1: str, nivel2: str) -> bool:
    """
    Verifica si dos niveles de formación coinciden.
    
    Args:
        nivel1: Nivel 1 (normalizado)
        nivel2: Nivel 2 (normalizado)
        
    Returns:
        True si coinciden, False en caso contrario
    """
    nivel1_norm = normalizar_nivel_formacion(nivel1)
    nivel2_norm = normalizar_nivel_formacion(nivel2)
    
    if not nivel1_norm or not nivel2_norm:
        return False
    
    return nivel1_norm == nivel2_norm


def cargar_referentes(archivo: Path = None) -> pd.DataFrame:
    if archivo is None:
        archivo = get_archivo_referentes()

    print(f"Cargando referentes desde: {archivo}")
    df = leer_datos_flexible(archivo)
    df = df[df['label'] == 1].copy()
    df = df.dropna(subset=['NOMBRE_DEL_PROGRAMA', 'NombrePrograma EAFIT'])

    # Cargar pares positivos adicionales si existen en la misma carpeta
    carpeta = archivo.parent
    archivos_pares = {
        'posParesPositivos': carpeta / 'posParesPositivos.csv',
        'preParesPositivos': carpeta / 'preParesPositivos.csv',
    }
    dfs_extra = []
    for nombre, ruta_pares in archivos_pares.items():
        if not ruta_pares.exists():
            # También buscar en subcarpeta backup/
            ruta_pares = carpeta / 'backup' / ruta_pares.name
        if ruta_pares.exists():
            try:
                df_pares = leer_datos_flexible(ruta_pares)
                # Normalizar columnas al esquema de referentesUnificados
                rename_map = {}
                for col in df_pares.columns:
                    col_l = str(col).strip().lower()
                    if 'nombre' in col_l and 'instituci' in col_l:
                        rename_map[col] = 'NOMBRE_INSTITUCIÓN'
                    elif 'nombre' in col_l and 'programa' in col_l and 'eafit' not in col_l:
                        rename_map[col] = 'NOMBRE_DEL_PROGRAMA'
                    elif 'nombre' in col_l and 'eafit' in col_l:
                        rename_map[col] = 'NombrePrograma EAFIT'
                    elif 'campo' in col_l and 'eafit' not in col_l and 'amplio' in col_l:
                        rename_map[col] = 'CAMPO_AMPLIO'
                    elif 'campo' in col_l and 'eafit' in col_l:
                        rename_map[col] = 'CAMPO_AMPLIO_EAFIT'
                    elif 'nivel' in col_l and 'eafit' not in col_l:
                        rename_map[col] = 'NIVEL_DE_FORMACIÓN'
                    elif 'nivel' in col_l and 'eafit' in col_l:
                        rename_map[col] = 'NIVEL_DE_FORMACIÓN EAFIT'
                    elif 'codigo' in col_l and 'eafit' in col_l:
                        rename_map[col] = 'Codigo EAFIT'
                if rename_map:
                    df_pares = df_pares.rename(columns=rename_map)
                df_pares['label'] = 1
                # Conservar solo filas con los campos mínimos necesarios
                required = ['NOMBRE_DEL_PROGRAMA', 'NombrePrograma EAFIT']
                if all(c in df_pares.columns for c in required):
                    df_pares = df_pares.dropna(subset=required)
                    dfs_extra.append(df_pares)
                    print(f"  Pares adicionales cargados desde {ruta_pares.name}: {len(df_pares)} filas")
                else:
                    print(f"  Advertencia: {ruta_pares.name} no tiene columnas mínimas requeridas, se omite")
            except Exception as e:
                print(f"  Advertencia: no se pudo cargar {ruta_pares.name}: {e}")
        else:
            print(f"  Archivo opcional no encontrado, se omite: {ruta_pares.name}")

    if dfs_extra:
        # Guardar el df original del referente para obtener niveles de los pares
        df_ref_original = df.copy()
        df = pd.concat([df] + dfs_extra, ignore_index=True)

        # Enriquecer filas de pares que no tienen NIVEL con datos del referente original
        if 'CÓDIGO_SNIES_DEL_PROGRAMA' in df.columns and 'CÓDIGO_SNIES_DEL_PROGRAMA' in df_ref_original.columns:
            df_ref_original['_snies_str'] = df_ref_original['CÓDIGO_SNIES_DEL_PROGRAMA'].astype(str).str.strip()
            df['_snies_str'] = df['CÓDIGO_SNIES_DEL_PROGRAMA'].astype(str).str.strip()
            nivel_lookup = df_ref_original.drop_duplicates(subset=['_snies_str']).set_index('_snies_str')
            for col_nivel in ['NIVEL_DE_FORMACIÓN', 'NIVEL_DE_FORMACIÓN EAFIT']:
                if col_nivel in nivel_lookup.columns:
                    mask_vacio = df[col_nivel].isna() | (df[col_nivel].astype(str).str.strip() == '')
                    df.loc[mask_vacio, col_nivel] = df.loc[mask_vacio, '_snies_str'].map(nivel_lookup[col_nivel])
            df = df.drop(columns=['_snies_str'], errors='ignore')

        # Eliminar duplicados conservando referente original (keep='first')
        if 'CÓDIGO_SNIES_DEL_PROGRAMA' in df.columns:
            df = df.drop_duplicates(subset=['CÓDIGO_SNIES_DEL_PROGRAMA'], keep='first')
        else:
            df = df.drop_duplicates(subset=['NOMBRE_DEL_PROGRAMA', 'NombrePrograma EAFIT'], keep='first')
        print(f"Total referentes combinados (referentes + pares): {len(df)}")

    # Asegurar columnas opcionales con fallback vacío
    for col_opt in ['CAMPO_AMPLIO', 'CAMPO_AMPLIO_EAFIT', 'NIVEL_DE_FORMACIÓN', 'NIVEL_DE_FORMACIÓN EAFIT']:
        if col_opt not in df.columns:
            df[col_opt] = ''

    df['NOMBRE_DEL_PROGRAMA_norm'] = df['NOMBRE_DEL_PROGRAMA'].apply(normalizar_texto)
    df['NombrePrograma EAFIT_norm'] = df['NombrePrograma EAFIT'].apply(normalizar_texto)
    df['CAMPO_AMPLIO_norm'] = df['CAMPO_AMPLIO'].fillna('').apply(normalizar_texto)
    df['CAMPO_AMPLIO_EAFIT_norm'] = df['CAMPO_AMPLIO_EAFIT'].fillna('').apply(normalizar_texto)

    if 'NIVEL_DE_FORMACIÓN' in df.columns:
        df['NIVEL_DE_FORMACIÓN_norm'] = df['NIVEL_DE_FORMACIÓN'].fillna('').apply(normalizar_nivel_formacion)
    else:
        df['NIVEL_DE_FORMACIÓN_norm'] = ''

    if 'NIVEL_DE_FORMACIÓN EAFIT' in df.columns:
        df['NIVEL_DE_FORMACIÓN_EAFIT_norm'] = df['NIVEL_DE_FORMACIÓN EAFIT'].fillna('').apply(normalizar_nivel_formacion)
    else:
        df['NIVEL_DE_FORMACIÓN_EAFIT_norm'] = ''

    antes_filtro = len(df)
    df = df[
        (df['NIVEL_DE_FORMACIÓN_norm'] != '') &
        (df['NIVEL_DE_FORMACIÓN_EAFIT_norm'] != '') &
        (df['NIVEL_DE_FORMACIÓN_norm'] == df['NIVEL_DE_FORMACIÓN_EAFIT_norm'])
    ].copy()
    despues_filtro = len(df)

    if antes_filtro != despues_filtro:
        print(f"Filtrados {antes_filtro - despues_filtro} referentes donde los niveles no coinciden")

    print(f"Total de referentes con label=1 y niveles coincidentes: {len(df)}")
    return df


def cargar_catalogo_eafit(archivo: Path = None) -> pd.DataFrame:
    """
    Carga el catálogo de programas EAFIT y filtra solo los programas con estado "activo".
    
    Args:
        archivo: Ruta al archivo del catálogo (Excel o CSV). Si es None, usa detección automática.
        
    Returns:
        DataFrame con programas EAFIT activos (filtrados por ESTADO_PROGRAMA = "activo")
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no tiene las columnas requeridas
    """
    if archivo is None:
        try:
            archivo = get_archivo_catalogo_eafit()
            print(f"[INFO] Ruta del catálogo EAFIT resuelta: {archivo}")
        except FileNotFoundError as e:
            print(f"[ERROR] No se pudo encontrar el archivo del catálogo EAFIT")
            print(f"  Error: {e}")
            raise FileNotFoundError(
                f"No se encontró el archivo del catálogo EAFIT.\n\n"
                f"Buscado en: ref/backup/ y ref/\n"
                f"Archivo esperado: catalogoOfertasEAFIT.xlsx o catalogoOfertasEAFIT.csv\n\n"
                f"Error original: {e}"
            ) from e
    
    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del catálogo EAFIT: {archivo}\n\n"
            "Verifica que el archivo exista en la carpeta ref/backup/ o ref/ o configura la ruta en config.json"
        )
    
    print(f"Cargando catálogo EAFIT desde: {archivo}")
    print(f"  Ruta absoluta: {archivo.resolve()}")
    try:
        df = leer_datos_flexible(archivo)
    except Exception as e:
        raise RuntimeError(
            f"Error al leer el catálogo EAFIT desde {archivo.name}: {e}\n\n"
            "Verifica que el archivo sea un Excel o CSV válido y que no esté corrupto."
        ) from e
    
    # Validar columnas requeridas
    columnas_requeridas = ['Nombre Programa EAFIT', 'CAMPO_AMPLIO', 'ESTADO_PROGRAMA']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    # Log de columnas disponibles para diagnóstico
    print(f"  Columnas disponibles en catálogo EAFIT ({len(df.columns)} total):")
    for i, col in enumerate(df.columns[:15], 1):  # Mostrar primeras 15 columnas
        print(f"    {i}. {col}")
    if len(df.columns) > 15:
        print(f"    ... y {len(df.columns) - 15} más")
    
    # Verificar si existe columna de código EAFIT
    posibles_columnas_codigo = ['Codigo EAFIT', 'Código Programa', 'CODIGO_PROGRAMA', 'Codigo Programa']
    columna_codigo_encontrada = None
    for col in posibles_columnas_codigo:
        if col in df.columns:
            columna_codigo_encontrada = col
            print(f"  ✓ Columna de código encontrada: '{col}'")
            break
    
    if not columna_codigo_encontrada:
        print(f"  ⚠ ADVERTENCIA: No se encontró columna de código EAFIT. Buscadas: {posibles_columnas_codigo}")
        print(f"    Esto puede causar problemas al buscar programas EAFIT por código.")
    
    if columnas_faltantes:
        raise ValueError(
            f"El catálogo EAFIT no tiene las columnas requeridas: {', '.join(columnas_faltantes)}\n\n"
            f"Columnas encontradas: {', '.join(df.columns[:10])}...\n\n"
            "Verifica que el archivo del catálogo tenga el formato correcto. La columna ESTADO_PROGRAMA es obligatoria."
        )
    
    # Normalizar nombres
    df['Nombre Programa EAFIT_norm'] = df['Nombre Programa EAFIT'].apply(normalizar_texto)
    df['CAMPO_AMPLIO_norm'] = df['CAMPO_AMPLIO'].fillna('').apply(normalizar_texto)
    
    # Normalizar niveles de formación
    # El catálogo EAFIT tiene la columna 'NIVEL_DE_FORMACIÓN'
    if 'NIVEL_DE_FORMACIÓN' in df.columns:
        df['NIVEL_DE_FORMACIÓN_norm'] = df['NIVEL_DE_FORMACIÓN'].fillna('').apply(normalizar_nivel_formacion)
    elif 'Nivel Programas' in df.columns:
        # Fallback: si no está NIVEL_DE_FORMACIÓN, usar Nivel Programas
        df['NIVEL_DE_FORMACIÓN_norm'] = df['Nivel Programas'].fillna('').apply(normalizar_nivel_formacion)
    else:
        print("ADVERTENCIA: No se encontró columna de nivel en catálogo EAFIT")
        df['NIVEL_DE_FORMACIÓN_norm'] = ''
    
    # CRÍTICO: Filtrar solo programas con estado "activo"
    # Solo se pueden asignar como referentes programas EAFIT que estén activos
    # Esta es una condición obligatoria: los programas inactivos NO pueden ser referentes
    antes_filtro_estado = len(df)
    # Normalizar estado para comparación (case-insensitive)
    df['ESTADO_PROGRAMA_norm'] = df['ESTADO_PROGRAMA'].fillna('').astype(str).str.lower().str.strip()
    df = df[df['ESTADO_PROGRAMA_norm'] == 'activo'].copy()
    despues_filtro_estado = len(df)
    if antes_filtro_estado != despues_filtro_estado:
        print(f"Filtrados {antes_filtro_estado - despues_filtro_estado} programas EAFIT inactivos (solo se consideran programas activos como referentes)")
    print(f"Total de programas EAFIT activos: {len(df)}")
    
    return df


def generar_features_embeddings(
    df: pd.DataFrame,
    modelo: Any,  # SentenceTransformer
    columna_nombre: str = 'NOMBRE_DEL_PROGRAMA_norm'
) -> np.ndarray:
    """
    Genera embeddings semánticos para los nombres de programas.
    
    Args:
        df: DataFrame con los programas
        modelo: Modelo de sentence-transformers
        columna_nombre: Nombre de la columna con los nombres de programas
        
    Returns:
        Array con embeddings (n_samples, embedding_dim)
    """
    print(f"Generando embeddings para {len(df)} programas...")
    textos = df[columna_nombre].astype(str).tolist()
    
    embeddings = modelo.encode(
        textos,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    return embeddings


def calcular_similitud_campo_amplio(
    campo1: str,
    campo2: str
) -> float:
    """
    Calcula similitud entre campos amplios (1 si son iguales, 0 si no).
    
    Args:
        campo1: Campo amplio 1
        campo2: Campo amplio 2
        
    Returns:
        Similitud (0.0 o 1.0)
    """
    if not campo1 or not campo2:
        return 0.0
    
    return 1.0 if campo1 == campo2 else 0.0


def calcular_similitud_nivel(
    nivel1: str,
    nivel2: str
) -> float:
    """
    Calcula similitud entre niveles de formación (1 si coinciden, 0 si no).
    CRÍTICO: Si los niveles no coinciden, no pueden ser referentes.
    
    Args:
        nivel1: Nivel 1
        nivel2: Nivel 2
        
    Returns:
        Similitud (0.0 o 1.0)
    """
    return 1.0 if niveles_coinciden(nivel1, nivel2) else 0.0


def preparar_features_entrenamiento(
    df_referentes: pd.DataFrame,
    modelo_embeddings: SentenceTransformer
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Prepara las features para entrenamiento del modelo.
    
    Features incluyen:
    - Embeddings del nombre del programa externo
    - Embeddings del nombre del programa EAFIT
    - Similitud coseno entre embeddings
    - Similitud de campo amplio (binaria)
    
    Args:
        df_referentes: DataFrame con referentes
        modelo_embeddings: Modelo de embeddings
        
    Returns:
        Tupla con (features, labels, encoder_programas_eafit)
    """
    print("Preparando features de entrenamiento...")
    
    # Generar embeddings para programas externos
    embeddings_externos = generar_features_embeddings(
        df_referentes,
        modelo_embeddings,
        'NOMBRE_DEL_PROGRAMA_norm'
    )
    
    # Generar embeddings para programas EAFIT
    embeddings_eafit = generar_features_embeddings(
        df_referentes,
        modelo_embeddings,
        'NombrePrograma EAFIT_norm'
    )
    
    # Calcular similitud coseno entre embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    similitudes_embeddings = np.array([
        cosine_similarity([emb_ext], [emb_eaf])[0][0]
        for emb_ext, emb_eaf in zip(embeddings_externos, embeddings_eafit)
    ])
    
    # OPTIMIZACIÓN: Calcular similitudes usando operaciones vectorizadas en lugar de iterrows()
    # Calcular similitud de campo amplio (vectorizado)
    campo_amplio_norm = df_referentes['CAMPO_AMPLIO_norm'].fillna('').astype(str)
    campo_amplio_eafit_norm = df_referentes['CAMPO_AMPLIO_EAFIT_norm'].fillna('').astype(str)
    similitudes_campo = (campo_amplio_norm == campo_amplio_eafit_norm).astype(float).values
    
    # Calcular similitud de nivel de formación (vectorizado)
    nivel_norm = df_referentes['NIVEL_DE_FORMACIÓN_norm'].fillna('').astype(str)
    nivel_eafit_norm = df_referentes['NIVEL_DE_FORMACIÓN_EAFIT_norm'].fillna('').astype(str)
    similitudes_nivel = (nivel_norm == nivel_eafit_norm).astype(float).values
    
    # Combinar features: embeddings externos + similitud coseno + similitud campo + similitud nivel
    # Usamos solo los embeddings del programa externo y las similitudes como features adicionales
    embedding_dim = embeddings_externos.shape[1]
    
    # Features: embeddings del programa externo + similitud coseno + similitud campo + similitud nivel
    features = np.hstack([
        embeddings_externos,  # embedding_dim features
        similitudes_embeddings.reshape(-1, 1),  # 1 feature
        similitudes_campo.reshape(-1, 1),  # 1 feature
        similitudes_nivel.reshape(-1, 1)  # 1 feature
    ])
    
    # Labels: código del programa EAFIT (usaremos el nombre normalizado como label)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df_referentes['NombrePrograma EAFIT_norm'])
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Total de programas EAFIT únicos: {len(encoder.classes_)}")
    
    return features, labels, encoder


def entrenar_modelo(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[RandomForestClassifier, dict]:
    """
    Entrena un modelo RandomForest para clasificar programas.
    
    Args:
        features: Array de features (n_samples, n_features)
        labels: Array de labels (n_samples,)
        test_size: Proporción de datos para test
        random_state: Semilla aleatoria
        
    Returns:
        Tupla con (modelo entrenado, métricas)
    """
    print("Dividiendo datos en entrenamiento y prueba...")

    # Si alguna clase tiene solo 1 muestra, stratify falla. En ese caso usamos split sin estratificar.
    counts_por_clase = np.bincount(labels)
    min_count = counts_por_clase.min()
    usar_stratify = min_count >= 2

    if not usar_stratify:
        print(
            f"[WARN] Al menos una clase tiene solo {min_count} muestra. "
            "Se realizará el split sin estratificar para evitar errores."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if usar_stratify else None,
    )
    
    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba: {len(X_test)} muestras")
    
    print("Entrenando modelo RandomForest...")
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    modelo.fit(X_train, y_train)
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    y_pred = modelo.predict(X_test)
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calcular accuracy
    accuracy = modelo.score(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    metricas = {
        'accuracy': accuracy,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    return modelo, metricas


def obtener_siguiente_version_modelo() -> int:
    """
    Obtiene el siguiente número de versión disponible para los modelos.
    
    Returns:
        Número de versión siguiente (ej: si existe v1 y v2, retorna 3)
    """
    versiones_existentes = []
    
    # Buscar todas las versiones existentes
    for archivo in MODELS_DIR.glob("clasificador_referentes_v*.pkl"):
        try:
            # Extraer número de versión del nombre: clasificador_referentes_v2.pkl -> 2
            nombre = archivo.stem  # Sin extensión
            if "_v" in nombre:
                num_str = nombre.split("_v")[-1]
                versiones_existentes.append(int(num_str))
        except (ValueError, IndexError):
            continue
    
    # Si no hay versiones, empezar en v1
    if not versiones_existentes:
        return 1
    
    return max(versiones_existentes) + 1


def listar_versiones_modelos() -> list[int]:
    """
    Lista todas las versiones de modelos disponibles.
    
    Returns:
        Lista de números de versión ordenados (ej: [1, 2, 3])
    """
    versiones = []
    
    for archivo in MODELS_DIR.glob("clasificador_referentes_v*.pkl"):
        try:
            nombre = archivo.stem
            if "_v" in nombre:
                num_str = nombre.split("_v")[-1]
                versiones.append(int(num_str))
        except (ValueError, IndexError):
            continue
    
    return sorted(versiones)


def obtener_rutas_modelo_version(version: int | None = None) -> tuple[Path, Path, Path]:
    """
    Obtiene las rutas de los archivos de modelo para una versión específica.
    
    Args:
        version: Número de versión (None = versión actual sin sufijo)
        
    Returns:
        Tupla (ruta_clasificador, ruta_embeddings, ruta_encoder)
    """
    if version is None:
        # Versión actual (sin sufijo)
        return MODELO_CLASIFICADOR, MODELO_EMBEDDINGS_OBJ, ENCODER_PROGRAMAS_EAFIT
    else:
        # Versión numerada
        return (
            MODELS_DIR / f"clasificador_referentes_v{version}.pkl",
            MODELS_DIR / f"modelo_embeddings_v{version}.pkl",
            MODELS_DIR / f"encoder_programas_eafit_v{version}.pkl"
        )


def guardar_modelos(
    modelo_clasificador: RandomForestClassifier,
    modelo_embeddings: SentenceTransformer,
    encoder: LabelEncoder,
    crear_version: bool = True
) -> int:
    """
    Guarda los modelos entrenados en disco con versionado.
    
    Args:
        modelo_clasificador: Modelo RandomForest entrenado
        modelo_embeddings: Modelo de embeddings
        encoder: Encoder de labels
        crear_version: Si True, guarda como nueva versión (vN). Si False, sobrescribe versión actual.
        
    Returns:
        Número de versión guardada
    """
    if crear_version:
        # Guardar como nueva versión
        version = obtener_siguiente_version_modelo()
        ruta_clasificador, ruta_embeddings, ruta_encoder = obtener_rutas_modelo_version(version)
        print(f"Guardando modelos versión {version} en {MODELS_DIR}...")
    else:
        # Guardar como versión actual (sin sufijo)
        version = None
        ruta_clasificador, ruta_embeddings, ruta_encoder = obtener_rutas_modelo_version(None)
        print(f"Guardando modelos (versión actual) en {MODELS_DIR}...")
    
    with open(ruta_clasificador, 'wb') as f:
        pickle.dump(modelo_clasificador, f)
    
    with open(ruta_embeddings, 'wb') as f:
        pickle.dump(modelo_embeddings, f)
    
    with open(ruta_encoder, 'wb') as f:
        pickle.dump(encoder, f)
    
    # Si es nueva versión, también crear enlaces simbólicos a versión "actual"
    if crear_version:
        try:
            import shutil
            shutil.copy2(ruta_clasificador, MODELO_CLASIFICADOR)
            shutil.copy2(ruta_embeddings, MODELO_EMBEDDINGS_OBJ)
            shutil.copy2(ruta_encoder, ENCODER_PROGRAMAS_EAFIT)
            print(f"Versión {version} también establecida como versión actual")
        except Exception as e:
            print(f"Advertencia: No se pudo crear enlace a versión actual: {e}")
    
    print("Modelos guardados exitosamente.")
    return version if crear_version else 0


def cargar_modelos(version: int | None = None) -> tuple[RandomForestClassifier, Any, LabelEncoder]:
    """
    Carga los modelos entrenados desde disco.
    
    Args:
        version: Número de versión a cargar (None = versión actual)
    
    Returns:
        Tupla con (modelo_clasificador, modelo_embeddings, encoder)
        
    Raises:
        FileNotFoundError: Si faltan archivos de modelos
        pickle.UnpicklingError: Si los archivos están corruptos
        MemoryError: Si no hay suficiente memoria para cargar los modelos
    """
    ruta_clasificador, ruta_embeddings, ruta_encoder = obtener_rutas_modelo_version(version)
    
    if version:
        print(f"Cargando modelos versión {version} desde disco...")
    else:
        print("Cargando modelos (versión actual) desde disco...")
    
    if not ruta_clasificador.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo clasificador{' versión ' + str(version) if version else ''}. Ejecute primero el entrenamiento.\n"
            f"Ruta esperada: {ruta_clasificador}"
        )
    if not ruta_embeddings.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de embeddings{' versión ' + str(version) if version else ''}. Ejecute primero el entrenamiento.\n"
            f"Ruta esperada: {ruta_embeddings}"
        )
    if not ruta_encoder.exists():
        raise FileNotFoundError(
            f"No se encontró el encoder{' versión ' + str(version) if version else ''}. Ejecute primero el entrenamiento.\n"
            f"Ruta esperada: {ruta_encoder}"
        )
    
    try:
        with open(ruta_clasificador, 'rb') as f:
            modelo_clasificador = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"El archivo {ruta_clasificador.name} está corrupto o no es un modelo válido.\n\n"
            "Solución: Reentrena el modelo ejecutando el entrenamiento nuevamente."
        ) from e
    except MemoryError as e:
        raise MemoryError(
            f"No hay suficiente memoria para cargar el modelo {ruta_clasificador.name}.\n\n"
            "Solución: Cierra otras aplicaciones y vuelve a intentar."
        ) from e
    
    try:
        with open(ruta_embeddings, 'rb') as f:
            modelo_embeddings = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"El archivo {ruta_embeddings.name} está corrupto o no es un modelo válido.\n\n"
            "Solución: Reentrena el modelo ejecutando el entrenamiento nuevamente."
        ) from e
    except MemoryError as e:
        raise MemoryError(
            f"No hay suficiente memoria para cargar el modelo {ruta_embeddings.name}.\n\n"
            "Solución: Cierra otras aplicaciones y vuelve a intentar."
        ) from e
    
    try:
        with open(ruta_encoder, 'rb') as f:
            encoder = pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"El archivo {ruta_encoder.name} está corrupto o no es un encoder válido.\n\n"
            "Solución: Reentrena el modelo ejecutando el entrenamiento nuevamente."
        ) from e
    except MemoryError as e:
        raise MemoryError(
            f"No hay suficiente memoria para cargar el encoder {ruta_encoder.name}.\n\n"
            "Solución: Cierra otras aplicaciones y vuelve a intentar."
        ) from e
    
    print("Modelos cargados exitosamente.")
    return modelo_clasificador, modelo_embeddings, encoder


def clasificar_programa_nuevo(
    nombre_programa: str,
    campo_amplio: str | None,
    nivel_formacion: str | None,
    modelo_clasificador: RandomForestClassifier,
    modelo_embeddings: Any,  # SentenceTransformer
    encoder: LabelEncoder,
    df_catalogo_eafit: pd.DataFrame,
    top_k_candidatos: int = 20,
    embeddings_catalogo: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Clasifica un programa nuevo y determina si es referente y a qué programa EAFIT pertenece.
    
    Estrategia:
    1. Normaliza el nivel de formación del programa nuevo
    2. FILTRA el catálogo EAFIT solo a programas con el mismo nivel Y estado "activo" (CRÍTICO)
    3. Genera embedding del programa nuevo
    4. Encuentra los top K candidatos por similitud de embedding (solo entre los del mismo nivel y activos)
    5. Para cada candidato, calcula features completas y usa el modelo para predecir
    6. Selecciona el mejor match
    7. Valida que el programa EAFIT asignado esté activo
    
    Args:
        nombre_programa: Nombre del programa nuevo
        campo_amplio: Campo amplio del programa (opcional)
        nivel_formacion: Nivel de formación del programa nuevo (CRÍTICO)
        modelo_clasificador: Modelo RandomForest entrenado
        modelo_embeddings: Modelo de embeddings
        encoder: Encoder de labels
        df_catalogo_eafit: DataFrame con catálogo EAFIT
        top_k_candidatos: Número de candidatos a evaluar con el modelo completo
        
    Returns:
        Diccionario con resultados de clasificación
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Normalizar nombre y nivel
    nombre_norm = normalizar_texto(nombre_programa)
    campo_norm = normalizar_texto(campo_amplio) if campo_amplio else ""
    nivel_norm = normalizar_nivel_formacion(nivel_formacion) if nivel_formacion else ""
    
    # CRÍTICO: Filtrar catálogo EAFIT por nivel de formación Y estado activo
    # Solo se pueden asignar programas EAFIT que estén activos
    mask_nivel = (
        df_catalogo_eafit['NIVEL_DE_FORMACIÓN_norm'] == nivel_norm
        if nivel_norm
        else np.ones(len(df_catalogo_eafit), dtype=bool)
    )
    
    # CRÍTICO: Filtrar por estado activo (condición obligatoria)
    # Solo se pueden asignar programas EAFIT que estén activos
    # El catálogo ya debería estar filtrado por estado activo, pero validamos nuevamente por seguridad
    if 'ESTADO_PROGRAMA_norm' in df_catalogo_eafit.columns:
        mask_activo = df_catalogo_eafit['ESTADO_PROGRAMA_norm'] == 'activo'
        mask_filtro = mask_nivel & mask_activo
    else:
        # Si no hay columna de estado normalizada, intentar con la original
        if 'ESTADO_PROGRAMA' in df_catalogo_eafit.columns:
            mask_activo = df_catalogo_eafit['ESTADO_PROGRAMA'].fillna('').astype(str).str.lower().str.strip() == 'activo'
            mask_filtro = mask_nivel & mask_activo
        else:
            # Si no hay columna de estado, solo filtrar por nivel (no debería pasar si el catálogo está bien)
            print("ADVERTENCIA: No se encontró columna ESTADO_PROGRAMA en catálogo. Filtrando solo por nivel.")
            mask_filtro = mask_nivel
    
    if nivel_norm and not mask_filtro.any():
        razon = f'Nivel de formación "{nivel_norm}" no coincide con ningún programa EAFIT activo'
        if 'ESTADO_PROGRAMA_norm' in df_catalogo_eafit.columns:
            # Verificar si hay programas con el nivel pero inactivos
            programas_nivel = df_catalogo_eafit.loc[mask_nivel]
            if len(programas_nivel) > 0:
                razon = f'Nivel de formación "{nivel_norm}" coincide pero no hay programas EAFIT activos con ese nivel'
        return {
            'es_referente': False,
            'probabilidad': 0.0,
            'programa_eafit_codigo': None,
            'programa_eafit_nombre': None,
            'similitud_embedding': 0.0,
            'similitud_campo': 0.0,
            'similitud_nivel': 0.0,
            'razon_no_referente': razon,
            'top_5_matches': []
        }
    df_candidatos = df_catalogo_eafit.loc[mask_filtro].copy()
    candidato_iloc = np.where(mask_filtro)[0]
    if not nivel_norm:
        print("ADVERTENCIA: No se proporcionó nivel de formación. Evaluando todos los candidatos activos.")

    # Generar embedding del programa nuevo
    embedding_programa = modelo_embeddings.encode(
        [nombre_norm],
        convert_to_numpy=True
    )

    # Usar embeddings precalculados del catálogo si están disponibles; si no, calcular
    if embeddings_catalogo is not None and len(embeddings_catalogo) == len(df_catalogo_eafit):
        embeddings_eafit = embeddings_catalogo[candidato_iloc]
    else:
        nombres_eafit = df_candidatos['Nombre Programa EAFIT_norm'].tolist()
        embeddings_eafit = modelo_embeddings.encode(
            nombres_eafit,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    # Calcular similitudes coseno con candidatos EAFIT
    similitudes = cosine_similarity(embedding_programa, embeddings_eafit)[0]
    
    # Obtener índices de los top K candidatos por similitud
    top_k_indices = np.argsort(similitudes)[::-1][:top_k_candidatos]
    
    mejores_matches = []
    
    # Evaluar cada candidato con el modelo completo
    for idx in top_k_indices:
        row_eafit = df_candidatos.iloc[idx]
        nombre_eafit_norm = row_eafit['Nombre Programa EAFIT_norm']
        campo_eafit_norm = row_eafit['CAMPO_AMPLIO_norm']
        nivel_eafit_norm = row_eafit['NIVEL_DE_FORMACIÓN_norm']
        embedding_eafit = embeddings_eafit[idx]
        similitud_emb = similitudes[idx]
        
        # Calcular similitud de campo amplio
        similitud_campo = calcular_similitud_campo_amplio(campo_norm, campo_eafit_norm)
        
        # Calcular similitud de nivel (debe ser 1.0 porque ya filtramos por nivel)
        similitud_nivel = calcular_similitud_nivel(nivel_norm, nivel_eafit_norm)
        
        # Preparar features para el modelo (igual que en entrenamiento)
        features = np.hstack([
            embedding_programa[0],  # embedding del programa nuevo
            np.array([similitud_emb]),  # similitud coseno
            np.array([similitud_campo]),  # similitud campo
            np.array([similitud_nivel])  # similitud nivel
        ]).reshape(1, -1)
        
        # Predecir con el modelo
        # El modelo predice qué programa EAFIT corresponde
        try:
            if nombre_eafit_norm in encoder.classes_:
                label_eafit = encoder.transform([nombre_eafit_norm])[0]
                proba = modelo_clasificador.predict_proba(features)[0]
                probabilidad_modelo = proba[label_eafit] if label_eafit < len(proba) else 0.0
                # Score combinado: modelo conoce este programa
                score_final = 0.6 * probabilidad_modelo + 0.4 * similitud_emb
            else:
                # Programa EAFIT no visto en entrenamiento: usar solo similitud semántica
                # sin penalizar, porque el modelo simplemente no tiene ejemplos de él
                probabilidad_modelo = similitud_emb
                score_final = similitud_emb
        except Exception:
            probabilidad_modelo = similitud_emb * 0.8
            score_final = similitud_emb * 0.8
        
        # Obtener estado del programa EAFIT
        estado_programa_eafit = ''
        if 'ESTADO_PROGRAMA_norm' in row_eafit.index:
            estado_programa_eafit = row_eafit['ESTADO_PROGRAMA_norm']
        elif 'ESTADO_PROGRAMA' in row_eafit.index:
            estado_programa_eafit = str(row_eafit['ESTADO_PROGRAMA']).lower().strip()
        
        mejores_matches.append({
            'Codigo EAFIT': row_eafit['Codigo EAFIT'],
            'NombrePrograma EAFIT': row_eafit['Nombre Programa EAFIT'],
            'Nivel EAFIT': nivel_eafit_norm,
            'Estado EAFIT': estado_programa_eafit,
            'probabilidad': score_final,
            'probabilidad_modelo': probabilidad_modelo,
            'similitud_embedding': similitud_emb,
            'similitud_campo': similitud_campo,
            'similitud_nivel': similitud_nivel
        })
    
    # Ordenar por score final descendente
    mejores_matches.sort(key=lambda x: x['probabilidad'], reverse=True)
    
    mejor_match = mejores_matches[0] if mejores_matches else None
    
    # Determinar si es referente (umbral ajustable)
    # VALIDACIÓN CRÍTICA: El nivel DEBE coincidir obligatoriamente Y el programa EAFIT DEBE estar activo
    # Si el nivel no coincide o el programa no está activo, automáticamente NO es referente, sin importar la probabilidad
    umbral_referente = UMBRAL_REFERENTE  # Ajustable desde config.json ("umbral_referente")
    
    if mejor_match:
        nivel_coincide = mejor_match.get('similitud_nivel', 0.0) == 1.0
        probabilidad_suficiente = mejor_match['probabilidad'] >= umbral_referente
        
        # CRÍTICO: Validar que el programa EAFIT esté activo (condición obligatoria)
        # Primero intentar usar el estado del match (más eficiente)
        estado_programa_eafit = mejor_match.get('Estado EAFIT', '')
        if estado_programa_eafit:
            programa_activo = estado_programa_eafit == 'activo'
        else:
            # Fallback: consultar el DataFrame si no está en el match
            programa_activo = False  # Por defecto, asumir inactivo si no se puede verificar
            if mejor_match.get('Codigo EAFIT'):
                programa_eafit_info = df_catalogo_eafit[
                    df_catalogo_eafit['Codigo EAFIT'] == mejor_match['Codigo EAFIT']
                ]
                if not programa_eafit_info.empty:
                    if 'ESTADO_PROGRAMA_norm' in programa_eafit_info.columns:
                        estado_programa = programa_eafit_info.iloc[0].get('ESTADO_PROGRAMA_norm', '')
                        programa_activo = estado_programa == 'activo'
                    elif 'ESTADO_PROGRAMA' in programa_eafit_info.columns:
                        estado_programa = str(programa_eafit_info.iloc[0].get('ESTADO_PROGRAMA', '')).lower().strip()
                        programa_activo = estado_programa == 'activo'
        
        # VALIDACIÓN ESTRICTA: Si el nivel NO coincide o el programa NO está activo, automáticamente NO es referente
        if not nivel_coincide or not programa_activo:
            es_referente = False
        else:
            # Solo si el nivel coincide Y el programa está activo, verificar la probabilidad
            es_referente = probabilidad_suficiente
    else:
        es_referente = False
    
    return {
        'es_referente': es_referente,
        'probabilidad': mejor_match['probabilidad'] if mejor_match else 0.0,
        'programa_eafit_codigo': mejor_match['Codigo EAFIT'] if mejor_match else None,
        'programa_eafit_nombre': mejor_match['NombrePrograma EAFIT'] if mejor_match else None,
        'similitud_embedding': mejor_match['similitud_embedding'] if mejor_match else 0.0,
        'similitud_campo': mejor_match['similitud_campo'] if mejor_match else 0.0,
        'similitud_nivel': mejor_match.get('similitud_nivel', 0.0) if mejor_match else 0.0,
        'top_5_matches': mejores_matches[:5]
    }


def clasificar_programas_nuevos(
    archivo_programas: Path | None = None,
    hoja: str = HOJA_PROGRAMAS,
    progress_callback: Callable[[int, int, str], None] | None = None,
    df_programas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Clasifica todos los programas nuevos del archivo Programas.xlsx.

    Args:
        archivo_programas: Ruta al archivo Excel con programas (si df_programas es None)
        hoja: Nombre de la hoja
        progress_callback: Opcional. Se llama con (actual, total, nombre_programa) cada 10 programas.
        df_programas: DataFrame opcional. Si se proporciona, se clasifica en memoria sin leer archivo.

    Returns:
        DataFrame con resultados de clasificación
    """
    # Si se proporciona DataFrame, trabajar en memoria
    if df_programas is not None:
        df_programas = df_programas.copy()
        log_info(f"Clasificando DataFrame en memoria ({len(df_programas)} filas)")
    else:
        # Modo tradicional: leer desde archivo
        archivo_programas = archivo_programas or ARCHIVO_PROGRAMAS
        print(f"Cargando programas desde: {archivo_programas}")
        df_programas = pd.read_excel(archivo_programas, sheet_name=hoja)
        log_info(f"Archivo de programas cargado: {archivo_programas.name}")
    
    # Filtrar solo programas nuevos
    if 'PROGRAMA_NUEVO' not in df_programas.columns:
        error_msg = (
            "No se encontró la columna 'PROGRAMA_NUEVO'. "
            "Ejecute primero procesamientoSNIES.py"
        )
        log_error(error_msg)
        raise ValueError(error_msg)
    
    df_nuevos = df_programas[df_programas['PROGRAMA_NUEVO'] == 'Sí'].copy()

    # ── BACKFILL: poblar ES_REFERENTE para programas ya conocidos en referentesUnificados ──
    # Propósito: los 4,846 programas que están en referentesUnificados.csv con label=1
    # nunca tuvieron ES_REFERENTE asignado porque ya eran PROGRAMA_NUEVO='No' cuando se
    # instaló el pipeline automatizado. Este bloque los corrige en cada ejecución.
    # Es idempotente: actúa solo sobre filas con ES_REFERENTE=NaN; en re-ejecuciones
    # detecta que ya están asignados y no hace nada (log: "0 programas pendientes").
    try:
        _ruta_ref = get_archivo_referentes()
        _df_ref = leer_datos_flexible(_ruta_ref)

        # Filtrar solo label=1 si la columna existe
        if 'label' in _df_ref.columns:
            _df_ref = _df_ref[_df_ref['label'] == 1].copy()

        _COL_SNIES = 'CÓDIGO_SNIES_DEL_PROGRAMA'
        _COL_NOM   = 'NombrePrograma EAFIT'
        _COL_COD   = 'Codigo EAFIT'

        if _COL_SNIES in _df_ref.columns and _COL_NOM in _df_ref.columns:
            # Normalizar códigos SNIES en referentesUnificados
            _df_ref['_snorm'] = (
                _df_ref[_COL_SNIES].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            )
            # Una sola entrada por código (keep='first' respeta el referente original)
            _ref_dedup = _df_ref.drop_duplicates(subset=['_snorm'], keep='first').set_index('_snorm')

            # Normalizar códigos en Programas.xlsx
            _snies_prog = (
                df_programas[_COL_SNIES]
                .astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            )

            # Solo actuar sobre filas sin ES_REFERENTE aún asignado
            _col_es_ref = 'ES_REFERENTE'
            if _col_es_ref not in df_programas.columns:
                df_programas[_col_es_ref] = None

            _mask_pendiente = df_programas[_col_es_ref].isna()
            _mask_en_ref    = _snies_prog.isin(_ref_dedup.index)
            _mask_backfill  = _mask_pendiente & _mask_en_ref
            _n              = _mask_backfill.sum()

            if _n > 0:
                # Asegurar columnas de clasificación
                for _c, _default in [
                    ('PROBABILIDAD',         0.0),
                    ('PROGRAMA_EAFIT_CODIGO', None),
                    ('PROGRAMA_EAFIT_NOMBRE', None),
                ]:
                    if _c not in df_programas.columns:
                        df_programas[_c] = _default

                _snies_bf = _snies_prog[_mask_backfill]

                df_programas.loc[_mask_backfill, _col_es_ref]          = 'Sí'
                df_programas.loc[_mask_backfill, 'PROBABILIDAD']        = 1.0
                df_programas.loc[_mask_backfill, 'PROGRAMA_EAFIT_NOMBRE'] = (
                    _snies_bf.map(_ref_dedup[_COL_NOM]).values
                )
                if _COL_COD in _ref_dedup.columns:
                    df_programas.loc[_mask_backfill, 'PROGRAMA_EAFIT_CODIGO'] = (
                        _snies_bf.map(_ref_dedup[_COL_COD]).values
                    )

                # Marcar como 'No' los que siguen sin asignar y son PROGRAMA_NUEVO='No'
                # (para no dejarlos en NaN indefinidamente)
                _mask_resto = (
                    df_programas[_col_es_ref].isna() &
                    (df_programas.get('PROGRAMA_NUEVO', 'No') == 'No')
                )
                df_programas.loc[_mask_resto, _col_es_ref] = 'No'
                df_programas.loc[_mask_resto, 'PROBABILIDAD'] = (
                    df_programas.loc[_mask_resto, 'PROBABILIDAD'].fillna(0.0)
                )

                log_info(
                    f"[Backfill] {_n:,} programas marcados ES_REFERENTE='Sí' "
                    f"por lookup en referentesUnificados "
                    f"({_mask_resto.sum():,} marcados 'No')."
                )
                print(f"[Backfill] {_n:,} referentes conocidos asignados correctamente.")
            else:
                log_info("[Backfill] Sin programas pendientes — ES_REFERENTE ya poblado.")
        else:
            log_warning(
                f"[Backfill] referentesUnificados no tiene columnas '{_COL_SNIES}' "
                f"o '{_COL_NOM}'. Backfill omitido."
            )
    except Exception as _e_bf:
        log_warning(
            f"[Backfill] No se pudo aplicar: {_e_bf}. "
            "Continuando con clasificación normal."
        )
    # ── FIN BACKFILL ──────────────────────────────────────────────────────────

    if len(df_nuevos) == 0:
        info_msg = "No hay programas nuevos para clasificar. Retornando DataFrame completo con todos los programas."
        print(info_msg)
        log_info(info_msg)
        # Asegurar que las columnas de clasificación existan en el DataFrame retornado
        # Esto evita errores cuando el pipeline espera estas columnas más adelante
        columnas_clasificacion = ['ES_REFERENTE', 'PROBABILIDAD', 'PROGRAMA_EAFIT_CODIGO', 'PROGRAMA_EAFIT_NOMBRE']
        for col in columnas_clasificacion:
            if col not in df_programas.columns:
                # Inicializar columnas con valores por defecto si no existen
                if col == 'ES_REFERENTE':
                    # fillna para no pisar los 'Sí' que el backfill ya asignó
                    df_programas[col] = df_programas[col].fillna('No')
                elif col == 'PROBABILIDAD':
                    df_programas[col] = 0.0
                else:  # PROGRAMA_EAFIT_CODIGO o PROGRAMA_EAFIT_NOMBRE
                    df_programas[col] = None
        # Retornar el DataFrame completo en lugar de uno vacío para que el pipeline pueda continuar
        return df_programas
    
    print(f"Clasificando {len(df_nuevos)} programas nuevos...")
    log_info(f"Iniciando clasificación de {len(df_nuevos)} programas nuevos")
    
    # Cargar modelos
    modelo_clasificador, modelo_embeddings, encoder = cargar_modelos()
    
    # Cargar catálogo EAFIT
    print("=" * 60)
    print("Cargando catálogo EAFIT para clasificación...")
    try:
        df_catalogo_eafit = cargar_catalogo_eafit()
        print(f"✓ Catálogo EAFIT cargado: {len(df_catalogo_eafit)} programas activos")
        if len(df_catalogo_eafit) == 0:
            error_msg = (
                "CRÍTICO: El catálogo EAFIT está vacío después del filtro de programas activos.\n\n"
                "Esto significa que no hay programas EAFIT activos para comparar, por lo que NO se pueden "
                "identificar referentes. Verifica el archivo del catálogo EAFIT."
            )
            print(f"[ERROR] {error_msg}")
            log_error(error_msg)
            raise ValueError(error_msg)
    except FileNotFoundError as e:
        error_msg = (
            f"CRÍTICO: No se pudo cargar el catálogo EAFIT: {e}\n\n"
            "El sistema necesita el archivo 'catalogoOfertasEAFIT.xlsx' o 'catalogoOfertasEAFIT.csv' "
            "en la carpeta ref/backup/ o ref/ para poder identificar referentes.\n\n"
            "Sin este archivo, NO se pueden clasificar programas como referentes."
        )
        print(f"[ERROR] {error_msg}")
        log_error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error al cargar catálogo EAFIT: {e}"
        print(f"[ERROR] {error_msg}")
        log_error(error_msg)
        raise
    
    # Precalcular embeddings del catálogo una sola vez (evita recalcular por cada programa nuevo)
    print("Precalculando embeddings del catálogo EAFIT...")
    textos_catalogo = df_catalogo_eafit['Nombre Programa EAFIT_norm'].astype(str).tolist()
    print(f"  Total de programas en catálogo para comparación: {len(textos_catalogo)}")
    try:
        embeddings_catalogo = modelo_embeddings.encode(
            textos_catalogo,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        log_info("Embeddings del catálogo EAFIT precalculados.")
    except MemoryError as e:
        error_msg = (
            f"No hay suficiente memoria para precalcular embeddings del catálogo ({len(textos_catalogo)} programas).\n\n"
            "Solución:\n"
            "1. Cierra otras aplicaciones que consuman memoria\n"
            "2. Reduce el tamaño del catálogo si es posible\n"
            "3. Procesa en lotes más pequeños (requiere modificar el código)"
        )
        log_error(error_msg)
        raise MemoryError(error_msg) from e

    # OPTIMIZACIÓN: Usar itertuples() en lugar de iterrows() para mejor rendimiento
    # Clasificar cada programa nuevo
    resultados = []
    total_nuevos = len(df_nuevos)
    
    # Filtrar filas con nombre válido antes del loop (optimización)
    mask_nombres_validos = (
        df_nuevos['NOMBRE_DEL_PROGRAMA'].notna() & 
        (df_nuevos['NOMBRE_DEL_PROGRAMA'].astype(str).str.strip() != '')
    )
    df_nuevos_validos = df_nuevos[mask_nombres_validos].copy()
    
    # Usar itertuples() que es más rápido que iterrows()
    for num, row_tuple in enumerate(df_nuevos_validos.itertuples(), start=1):
        idx = row_tuple.Index
        nombre_programa = getattr(row_tuple, 'NOMBRE_DEL_PROGRAMA', '')
        campo_amplio = getattr(row_tuple, 'CINE_F_2013_AC_CAMPO_AMPLIO', None)
        nivel_formacion = getattr(row_tuple, 'NIVEL_DE_FORMACIÓN', None)
        
        resultado = clasificar_programa_nuevo(
            str(nombre_programa),
            str(campo_amplio) if campo_amplio else None,
            str(nivel_formacion) if nivel_formacion else None,
            modelo_clasificador,
            modelo_embeddings,
            encoder,
            df_catalogo_eafit,
            embeddings_catalogo=embeddings_catalogo,
        )
        
        # VALIDACIÓN CRÍTICA: Comparar directamente los campos NIVEL_DE_FORMACIÓN Y ESTADO_PROGRAMA
        # Regla de negocio (TODAS las condiciones son obligatorias):
        # 1. Si el NIVEL_DE_FORMACIÓN del programa nuevo (Programas.xlsx) 
        #    NO coincide exactamente con el NIVEL_DE_FORMACIÓN del programa EAFIT (catalogoOfertasEAFIT.xlsx),
        #    automáticamente NO son referentes (ES_REFERENTE = "No")
        # 2. Si el programa EAFIT NO está en estado "activo", automáticamente NO es referente
        # 3. El score final debe ser >= umbral_referente (configurable en config.json)
        
        # Normalizar el nivel del programa nuevo desde Programas.xlsx
        nivel_programa_nuevo_norm = normalizar_nivel_formacion(nivel_formacion) if nivel_formacion else ""
        
        # Inicializar como NO referente si no hay nivel válido
        es_referente_final = False
        razon_no_referente = []
        
        # Solo puede ser referente si se cumplen TODAS estas condiciones:
        # 1. Hay un programa EAFIT asignado
        # 2. El nivel del programa nuevo es válido
        # 3. El nivel del programa EAFIT es válido
        # 4. Ambos niveles coinciden exactamente
        # 5. El programa EAFIT está en estado "activo" (OBLIGATORIO)
        # 6. El score final >= umbral_referente
        if resultado['programa_eafit_codigo'] is not None and nivel_programa_nuevo_norm:
            # Obtener el nivel y estado del programa EAFIT desde catalogoOfertasEAFIT.xlsx
            programa_eafit_codigo = resultado['programa_eafit_codigo']
            
            # Intentar buscar por diferentes nombres de columna de código
            programa_eafit_info = pd.DataFrame()
            posibles_columnas_codigo = ['Codigo EAFIT', 'Código Programa', 'CODIGO_PROGRAMA', 'Codigo Programa']
            
            for col_codigo in posibles_columnas_codigo:
                if col_codigo in df_catalogo_eafit.columns:
                    programa_eafit_info = df_catalogo_eafit[
                        df_catalogo_eafit[col_codigo].astype(str) == str(programa_eafit_codigo)
                    ]
                    if not programa_eafit_info.empty:
                        break
            
            if not programa_eafit_info.empty:
                nivel_eafit_norm = programa_eafit_info.iloc[0].get('NIVEL_DE_FORMACIÓN_norm', '')
                estado_programa_eafit = programa_eafit_info.iloc[0].get('ESTADO_PROGRAMA_norm', '')
                estado_original = programa_eafit_info.iloc[0].get('ESTADO_PROGRAMA', '')
                
                if 'ESTADO_PROGRAMA' in programa_eafit_info.columns:
                    # Si no hay columna normalizada, usar la original
                    if not estado_programa_eafit:
                        estado_programa_eafit = programa_eafit_info.iloc[0].get('ESTADO_PROGRAMA', '').lower().strip()
                else:
                    # Si no hay columna ESTADO_PROGRAMA, no se puede validar → NO es referente
                    estado_programa_eafit = 'inactivo'  # Por seguridad, asumir inactivo si no se puede verificar
                    razon_no_referente.append("No se encontró columna ESTADO_PROGRAMA en catálogo EAFIT")
                
                # VALIDACIÓN DIRECTA: Comparar los niveles normalizados Y verificar estado activo
                # Ambas condiciones son OBLIGATORIAS: nivel debe coincidir Y programa debe estar activo
                if nivel_eafit_norm:
                    if nivel_programa_nuevo_norm == nivel_eafit_norm:
                        # Los niveles coinciden → verificar estado activo (condición obligatoria)
                        if estado_programa_eafit == 'activo':
                            # Los niveles coinciden Y el programa está activo → puede ser referente (depende de probabilidad)
                            es_referente_final = resultado['es_referente']
                            if not es_referente_final:
                                razon_no_referente.append(f"Probabilidad ({resultado.get('probabilidad', 0.0):.4f}) no alcanza el umbral")
                        else:
                            # Los niveles coinciden PERO el programa NO está activo → NO es referente
                            # Esta es una condición obligatoria: solo programas activos pueden ser referentes
                            es_referente_final = False
                            razon_no_referente.append(f"Programa EAFIT no está activo (estado: '{estado_original}')")
                    else:
                        # Los niveles NO coinciden → NO es referente
                        es_referente_final = False
                        razon_no_referente.append(f"Niveles no coinciden: '{nivel_programa_nuevo_norm}' != '{nivel_eafit_norm}'")
                else:
                    # El programa EAFIT no tiene nivel válido
                    es_referente_final = False
                    razon_no_referente.append("Programa EAFIT no tiene NIVEL_DE_FORMACIÓN válido")
            else:
                # No se encontró el programa EAFIT en el catálogo
                es_referente_final = False
                razon_no_referente.append(f"No se encontró programa EAFIT con código '{programa_eafit_codigo}' en catálogo")
        elif resultado['programa_eafit_codigo'] is None:
            es_referente_final = False
            razon_no_referente.append("No se encontró ningún programa EAFIT candidato")
        elif not nivel_programa_nuevo_norm:
            # El programa nuevo no tiene nivel válido → NO puede ser referente
            es_referente_final = False
            razon_no_referente.append(f"Nivel de formación inválido: '{nivel_formacion}'")
        
        if progress_callback and (num % 10 == 0 or num == total_nuevos):
            try:
                progress_callback(num, total_nuevos, str(nombre_programa)[:80])
            except Exception:
                pass
        resultados.append({
            'CÓDIGO_SNIES_DEL_PROGRAMA': getattr(row_tuple, 'CÓDIGO_SNIES_DEL_PROGRAMA', None),
            'NOMBRE_DEL_PROGRAMA': nombre_programa,
            'NIVEL_FORMACION': nivel_formacion,
            'ES_REFERENTE': 'Sí' if es_referente_final else 'No',
            'PROBABILIDAD': resultado['probabilidad'],
            'PROGRAMA_EAFIT_CODIGO': resultado['programa_eafit_codigo'],
            'PROGRAMA_EAFIT_NOMBRE': resultado['programa_eafit_nombre'],
            'SIMILITUD_EMBEDDING': resultado['similitud_embedding'],
            'SIMILITUD_CAMPO': resultado['similitud_campo'],
            'SIMILITUD_NIVEL': resultado.get('similitud_nivel', 0.0)
        })
        
        if (len(resultados) % 10) == 0:
            print(f"Procesados {len(resultados)}/{total_nuevos} programas...")
    
    df_resultados = pd.DataFrame(resultados)
    
    # Agregar columnas al archivo original (si no existen)
    print("Agregando resultados al archivo original...")
    columnas_clasificacion = ['ES_REFERENTE', 'PROBABILIDAD', 'PROGRAMA_EAFIT_CODIGO', 
                              'PROGRAMA_EAFIT_NOMBRE']
    for col in columnas_clasificacion:
        if col not in df_programas.columns:
            df_programas[col] = None
            print(f"  Columna '{col}' creada (no existía)")
    
    # Normalizar códigos SNIES para comparación robusta
    def _normalizar_codigo_snies(valor: object) -> str:
        """Normaliza un código SNIES para comparación (maneja float, string, etc.)."""
        if pd.isna(valor):
            return ""
        # Convertir a string y limpiar
        s = str(valor).strip()
        # Remover .0 si es float convertido a string
        if s.endswith(".0"):
            s = s[:-2]
        return s
    
    # Crear columna normalizada temporal para comparación
    df_programas['_CODIGO_SNIES_NORM'] = df_programas['CÓDIGO_SNIES_DEL_PROGRAMA'].apply(_normalizar_codigo_snies)
    
    # Mapear resultados al DataFrame original
    programas_mapeados = 0
    programas_no_encontrados = []
    
    for resultado in resultados:
        codigo_snies = resultado['CÓDIGO_SNIES_DEL_PROGRAMA']
        codigo_norm = _normalizar_codigo_snies(codigo_snies)
        
        # Buscar usando la columna normalizada
        mask = df_programas['_CODIGO_SNIES_NORM'] == codigo_norm
        
        if mask.any():
            df_programas.loc[mask, 'ES_REFERENTE'] = resultado['ES_REFERENTE']
            df_programas.loc[mask, 'PROBABILIDAD'] = resultado['PROBABILIDAD']
            df_programas.loc[mask, 'PROGRAMA_EAFIT_CODIGO'] = resultado['PROGRAMA_EAFIT_CODIGO']
            df_programas.loc[mask, 'PROGRAMA_EAFIT_NOMBRE'] = resultado['PROGRAMA_EAFIT_NOMBRE']
            programas_mapeados += mask.sum()
        else:
            programas_no_encontrados.append(codigo_snies)
    
    # Eliminar columna temporal
    df_programas = df_programas.drop(columns=['_CODIGO_SNIES_NORM'])
    
    # Log de diagnóstico
    if programas_no_encontrados:
        print(f"[WARN] {len(programas_no_encontrados)} programas no se encontraron en el DataFrame para mapear resultados:")
        for codigo in programas_no_encontrados[:5]:  # Mostrar solo los primeros 5
            print(f"  - {codigo}")
        if len(programas_no_encontrados) > 5:
            print(f"  ... y {len(programas_no_encontrados) - 5} más")
    
    print(f"Mapeados {programas_mapeados} registros con resultados de clasificación.")
    
    # Validación: verificar que los valores se mapearon correctamente
    if programas_mapeados > 0:
        df_validacion = df_programas[df_programas['ES_REFERENTE'].notna()].copy()
        referentes_mapeados = (df_validacion['ES_REFERENTE'] == 'Sí').sum()
        no_referentes_mapeados = (df_validacion['ES_REFERENTE'] == 'No').sum()
        print(f"Validación: {referentes_mapeados} referentes y {no_referentes_mapeados} no referentes mapeados correctamente.")
    
    # Si se proporcionó df_programas, solo retornar (sin escribir)
    if df_programas is not None and archivo_programas is None:
        total_nuevos = len(df_nuevos)
        referentes = df_resultados['ES_REFERENTE'].value_counts().get('Sí', 0)
        no_referentes = df_resultados['ES_REFERENTE'].value_counts().get('No', 0)
        
        print(f"\nClasificación completada:")
        print(f"Total de programas nuevos: {total_nuevos}")
        print(f"Programas clasificados como referentes: {referentes}")
        print(f"Programas no referentes: {no_referentes}")
        
        log_resultado(f"Total de programas nuevos clasificados: {total_nuevos}")
        log_resultado(f"Programas clasificados como referentes: {referentes}")
        log_resultado(f"Programas no referentes: {no_referentes}")
        
        return df_programas
    
    # Modo tradicional: escribir de vuelta al archivo
    archivo_programas = archivo_programas or ARCHIVO_PROGRAMAS
    print(f"Guardando resultados en {archivo_programas}...")
    with pd.ExcelWriter(
        archivo_programas,
        mode="a",
        if_sheet_exists="replace",
        engine="openpyxl",
    ) as writer:
        df_programas.to_excel(writer, sheet_name=hoja, index=False)
    
    total_nuevos = len(df_nuevos)
    referentes = df_resultados['ES_REFERENTE'].value_counts().get('Sí', 0)
    no_referentes = df_resultados['ES_REFERENTE'].value_counts().get('No', 0)
    
    print(f"\nClasificación completada:")
    print(f"Total de programas nuevos: {total_nuevos}")
    print(f"Programas clasificados como referentes: {referentes}")
    print(f"Programas no referentes: {no_referentes}")
    
    log_resultado(f"Total de programas nuevos clasificados: {total_nuevos}")
    log_resultado(f"Programas clasificados como referentes: {referentes}")
    log_resultado(f"Programas no referentes: {no_referentes}")
    
    return df_programas


def entrenar_y_guardar_modelo() -> None:
    """
    Función principal para entrenar el modelo desde cero.
    """
    print("=== ENTRENAMIENTO DEL MODELO DE CLASIFICACIÓN ===\n")
    
    # Cargar datos de entrenamiento
    df_referentes = cargar_referentes()
    
    # Cargar modelo de embeddings
    print(f"\nCargando modelo de embeddings: {MODELO_EMBEDDINGS}")
    SentenceTransformer = _get_sentence_transformer()
    modelo_embeddings = SentenceTransformer(MODELO_EMBEDDINGS)
    
    # Preparar features
    features, labels, encoder = preparar_features_entrenamiento(
        df_referentes,
        modelo_embeddings
    )
    
    # Entrenar modelo
    modelo_clasificador, metricas = entrenar_modelo(features, labels)
    
    # Guardar modelos (crear nueva versión)
    version = guardar_modelos(modelo_clasificador, modelo_embeddings, encoder, crear_version=True)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"Modelos guardados en: {MODELS_DIR} (versión {version})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "entrenar":
        # Entrenar modelo
        entrenar_y_guardar_modelo()
    else:
        # Clasificar programas nuevos
        clasificar_programas_nuevos()

