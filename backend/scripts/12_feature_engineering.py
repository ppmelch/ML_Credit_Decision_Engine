"""
Feature engineering para el modelo individual de PD.

Construye el dataset final de modelado consolidando los tres pilares de
features definidos en el alcance del proyecto:

- Pilar A: patrón temporal del flujo mensual de remesas (la innovación
  metodológica del proyecto).
- Pilar B: características demográficas del receptor del padrón
  sintético.
- Pilar C: variables contextuales del municipio de residencia,
  incluyendo el score regional del Componente 1.

Estructura del dataset resultante: una fila por hogar, compatible
directamente con modelos tabulares (LogReg, XGBoost) sin transformaciones
adicionales. Para el LSTM de Fase 8, las features tabulares se
concatenarán al output de la red recurrente que procesará la serie de 24
meses por separado.

Cuota propuesta del crédito (para `pa_ratio_remesa_cuota`)
----------------------------------------------------------
Se asume un crédito de mejoramiento de vivienda con monto igual a 36 veces
el ingreso mensual de remesas del hogar (en pesos), tasa fija de 12.0%
anual (representativa de las tasas en pesos para crédito hipotecario y
mejoramiento de vivienda en México según CONDUSEF, 2024) y plazo de 60
meses. La cuota mensual se calcula con la fórmula de anualidad estándar
(PMT). El tipo de cambio asumido es 18 MXN/USD.

Insumos
-------
data/processed/series_mensuales_hogares.csv     (Fase 5)
data/processed/padron_hogares_sinteticos.csv    (Fase 4)
data/processed/jalisco_municipal_features.csv   (Fase 2)
data/processed/jalisco_score_regional.csv       (Fase 2)

Salida
------
data/processed/dataset_modelado_individual.csv
    Una fila por hogar con id_hogar, features de los tres pilares.
data/processed/dataset_modelado_individual_metadata.json
    Diccionario de variables, parámetros, fuentes y diagnósticos.

Uso
---
    python scripts/12_feature_engineering.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Import condicional de statsmodels: si está disponible se usa STL para
# `pa_fuerza_estacional`; si no, se aplica una aproximación robusta basada
# en la varianza explicada por el patrón estacional empírico.
try:
    from statsmodels.tsa.seasonal import STL  # type: ignore

    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERIES_CSV = (
    PROJECT_ROOT / "data" / "processed" / "series_mensuales_hogares.csv"
)
PADRON_CSV = (
    PROJECT_ROOT / "data" / "processed" / "padron_hogares_sinteticos.csv"
)
MUNI_FEATURES_CSV = (
    PROJECT_ROOT / "data" / "processed" / "jalisco_municipal_features.csv"
)
SCORE_REGIONAL_CSV = (
    PROJECT_ROOT / "data" / "processed" / "jalisco_score_regional.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_modelado_individual.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_modelado_individual_metadata.json"
)

# ---------------------------------------------------------------------------
# Parámetros del crédito hipotético
# ---------------------------------------------------------------------------
TIPO_CAMBIO_MXN_POR_USD = 18.0
MULTIPLO_INGRESO_CREDITO = 36  # crédito = 36 × remesa mensual
TASA_ANUAL_CREDITO = 0.12
PLAZO_MESES_CREDITO = 60

# ---------------------------------------------------------------------------
# Codificación ordinal de escolaridad
# ---------------------------------------------------------------------------
ESCOLARIDAD_ORD = {
    "sin_estudios": 0,
    "primaria": 1,
    "secundaria": 2,
    "media_superior": 3,
    "superior": 4,
}


# ---------------------------------------------------------------------------
# Pilar A — utilidades para series mensuales
# ---------------------------------------------------------------------------
def calcular_features_pilar_a(
    serie: np.ndarray, antiguedad_meses: int, monto_base_usd: float
) -> dict[str, float]:
    """
    Calcula las features del Pilar A para una serie individual.

    Parameters
    ----------
    serie : np.ndarray
        Vector de 24 montos mensuales en USD (con ceros en meses sin
        envío).
    antiguedad_meses : int
        Antigüedad del flujo de remesas reportada por el hogar (Fase 4).
    monto_base_usd : float
        Monto base esperado del hogar (Fase 4). Se usa para calcular la
        cuota propuesta del crédito hipotético.

    Returns
    -------
    dict[str, float]
        Diccionario con las features del Pilar A.
    """
    n = len(serie)
    no_cero = serie[serie > 0]
    n_no_cero = len(no_cero)

    # --- Estadísticos de monto en meses con envío --------------------
    if n_no_cero > 0:
        pa_mediana_remesa = float(np.median(no_cero))
        pa_media_remesa = float(np.mean(no_cero))
        pa_desv_remesa = float(np.std(no_cero, ddof=1)) if n_no_cero > 1 else 0.0
        pa_cv_remesa = (
            pa_desv_remesa / pa_media_remesa if pa_media_remesa > 0 else 0.0
        )
    else:
        pa_mediana_remesa = 0.0
        pa_media_remesa = 0.0
        pa_desv_remesa = 0.0
        pa_cv_remesa = 0.0

    # --- Tendencia: pendiente OLS sobre la serie completa -----------
    # Usamos la serie con ceros para que las interrupciones afecten la
    # estimación de la tendencia (un hogar con varias interrupciones
    # tendrá pendiente más volátil/sesgada).
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    s_mean = serie.mean()
    var_t = float(np.sum((t - t_mean) ** 2))
    if var_t > 0:
        cov_ts = float(np.sum((t - t_mean) * (serie - s_mean)))
        pa_pendiente_24m = cov_ts / var_t  # USD/mes
    else:
        pa_pendiente_24m = 0.0

    # Pendiente normalizada por mediana (% mensual). Si mediana es 0 se
    # imputa 0.
    if pa_mediana_remesa > 0:
        pa_pendiente_norm = pa_pendiente_24m / pa_mediana_remesa
    else:
        pa_pendiente_norm = 0.0

    # --- Interrupciones ---------------------------------------------
    pa_n_meses_sin_envio = int(np.sum(serie == 0))

    # Mayor racha consecutiva de ceros.
    max_racha = 0
    racha_actual = 0
    for v in serie:
        if v == 0:
            racha_actual += 1
            if racha_actual > max_racha:
                max_racha = racha_actual
        else:
            racha_actual = 0
    pa_max_racha_sin_envio = int(max_racha)

    # --- Fuerza estacional ------------------------------------------
    # Definida como Var(componente_estacional) / Var(serie). Rango [0, 1]:
    # 0 = sin estacionalidad, valores altos = patrón estacional dominante.
    pa_fuerza_estacional = _calcular_fuerza_estacional(serie)

    # --- Ratio remesa / cuota ---------------------------------------
    # Crédito hipotético: 36 × monto_base_usd × tipo_cambio (en MXN).
    monto_credito_mxn = (
        MULTIPLO_INGRESO_CREDITO * monto_base_usd * TIPO_CAMBIO_MXN_POR_USD
    )
    cuota_mxn = _calcular_pmt(
        principal=monto_credito_mxn,
        tasa_anual=TASA_ANUAL_CREDITO,
        plazo_meses=PLAZO_MESES_CREDITO,
    )
    cuota_usd = cuota_mxn / TIPO_CAMBIO_MXN_POR_USD

    # Ratio = mediana realizada del flujo (más conservador que monto base).
    if cuota_usd > 0:
        pa_ratio_remesa_cuota = pa_mediana_remesa / cuota_usd
    else:
        pa_ratio_remesa_cuota = 0.0

    return {
        "pa_mediana_remesa": pa_mediana_remesa,
        "pa_media_remesa": pa_media_remesa,
        "pa_desv_remesa": pa_desv_remesa,
        "pa_cv_remesa": pa_cv_remesa,
        "pa_pendiente_24m": pa_pendiente_24m,
        "pa_pendiente_norm": pa_pendiente_norm,
        "pa_n_meses_sin_envio": pa_n_meses_sin_envio,
        "pa_max_racha_sin_envio": pa_max_racha_sin_envio,
        "pa_fuerza_estacional": pa_fuerza_estacional,
        "pa_antiguedad_meses": antiguedad_meses,
        "pa_cuota_propuesta_mxn": cuota_mxn,
        "pa_ratio_remesa_cuota": pa_ratio_remesa_cuota,
    }


def _calcular_pmt(
    principal: float, tasa_anual: float, plazo_meses: int
) -> float:
    """
    Calcula la cuota mensual fija de un crédito tipo francés (PMT).

    PMT = P × i / (1 - (1+i)^(-n))

    Parameters
    ----------
    principal : float
        Monto del crédito.
    tasa_anual : float
        Tasa anual (e.g. 0.12 para 12%).
    plazo_meses : int
        Número de mensualidades.

    Returns
    -------
    float
        Cuota mensual en la misma moneda que principal.
    """
    if plazo_meses <= 0:
        return 0.0
    i = tasa_anual / 12.0
    if i == 0:
        return principal / plazo_meses
    return principal * i / (1.0 - (1.0 + i) ** (-plazo_meses))


def _calcular_fuerza_estacional(serie: np.ndarray) -> float:
    """
    Calcula la fuerza del componente estacional de una serie.

    Si statsmodels está disponible, se aplica STL con período 12. La
    fuerza se define como max(0, 1 - Var(residuo)/Var(residuo+estacional))
    según Hyndman & Athanasopoulos (FPP3).

    Si no, se aplica una aproximación: Var(componente_mensual_promedio)
    sobre Var(serie completa), donde el componente_mensual_promedio se
    construye replicando los promedios por mes calendario en posición
    correspondiente. Esto es robusto y no requiere dependencias externas.

    Parameters
    ----------
    serie : np.ndarray
        Vector de 24 montos.

    Returns
    -------
    float
        Fuerza estacional en [0, 1].
    """
    n = len(serie)
    var_serie = float(np.var(serie))
    if var_serie <= 0:
        return 0.0

    # Solo intenta STL si la serie tiene suficiente variabilidad y no es
    # casi-constante (STL falla con series degeneradas).
    if _STATSMODELS_OK:
        try:
            # STL requiere al menos 2 ciclos completos. n=24 con period=12
            # cumple. robust=True maneja outliers (envíos cero).
            stl = STL(serie, period=12, robust=True).fit()
            var_resid = float(np.var(stl.resid))
            var_seas_resid = float(np.var(stl.seasonal + stl.resid))
            if var_seas_resid > 0:
                fuerza = max(0.0, 1.0 - var_resid / var_seas_resid)
                return float(min(fuerza, 1.0))
        except Exception:
            pass  # fallback abajo

    # Fallback: varianza explicada por promedios mensuales empíricos.
    if n != 24:
        return 0.0
    promedios_por_mes = np.zeros(12)
    for m in range(12):
        promedios_por_mes[m] = (serie[m] + serie[m + 12]) / 2.0
    componente_estacional = np.tile(promedios_por_mes, 2)
    var_estacional = float(np.var(componente_estacional))
    fuerza = var_estacional / var_serie
    return float(min(max(fuerza, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Pilar B — features demográficas
# ---------------------------------------------------------------------------
def calcular_features_pilar_b(hogar: dict[str, object]) -> dict[str, float]:
    """Construye features demográficas del receptor."""
    return {
        "pb_edad": float(hogar["edad_receptor"]),
        "pb_genero_F": 1.0 if hogar["genero_receptor"] == "F" else 0.0,
        "pb_n_dependientes": float(hogar["n_dependientes"]),
        "pb_escolaridad_ord": float(
            ESCOLARIDAD_ORD[str(hogar["escolaridad"])]
        ),
        "pb_vivienda_propia": (
            1.0 if hogar["tipo_vivienda_actual"] == "propia" else 0.0
        ),
        "pb_vivienda_rentada": (
            1.0 if hogar["tipo_vivienda_actual"] == "rentada" else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Pilar C — features contextuales del municipio
# ---------------------------------------------------------------------------
def cargar_features_municipales() -> dict[str, dict[str, float]]:
    """
    Carga features municipales y los une con el score regional.

    Returns
    -------
    dict[str, dict[str, float]]
        cve_municipio -> dict con features del Pilar C.
    """
    municipal: dict[str, dict[str, float]] = {}
    with open(MUNI_FEATURES_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            municipal[cve] = {
                "pc_iim_dp2": _safe_float(row.get("iim_dp2")),
                "pc_pct_viv_remesas": _safe_float(row.get("pct_viv_remesas")),
                "pc_pct_viv_emigrantes": _safe_float(
                    row.get("pct_viv_emigrantes")
                ),
                "pc_imn_2020": _safe_float(row.get("imn_2020")),
                "pc_escolaridad_promedio_mun": _safe_float(
                    row.get("escolaridad_promedio")
                ),
                "pc_pct_pob_ocupada_2sm": _safe_float(
                    row.get("pct_pob_ocupada_hasta_2sm")
                ),
            }

    # Anexar score regional.
    with open(SCORE_REGIONAL_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            if cve in municipal:
                municipal[cve]["pc_score_regional"] = _safe_float(
                    row.get("score_regional")
                )
                municipal[cve]["pc_sub_capacidad_pago"] = _safe_float(
                    row.get("sub_score_capacidad_pago")
                )
                municipal[cve]["pc_sub_intensidad_migratoria"] = _safe_float(
                    row.get("sub_score_intensidad_migratoria")
                )

    return municipal


def _safe_float(v: object) -> float:
    """Convierte a float; vacíos/inválidos -> 0.0."""
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Carga del padrón y series
# ---------------------------------------------------------------------------
def cargar_padron() -> dict[str, dict[str, object]]:
    """Carga el padrón indexado por id_hogar."""
    padron: dict[str, dict[str, object]] = {}
    with open(PADRON_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["edad_receptor"] = int(row["edad_receptor"])
            row["n_dependientes"] = int(row["n_dependientes"])
            row["antiguedad_recepcion_meses"] = int(
                row["antiguedad_recepcion_meses"]
            )
            row["remesa_mediana_esperada_usd"] = float(
                row["remesa_mediana_esperada_usd"]
            )
            row["cve_municipio"] = row["cve_municipio"].zfill(5)
            padron[row["id_hogar"]] = row
    return padron


def cargar_series_wide() -> tuple[list[str], list[str], np.ndarray]:
    """
    Carga las series mensuales en formato matricial.

    Returns
    -------
    ids : list[str]
        id_hogar por fila.
    cves : list[str]
        cve_municipio por fila.
    matriz : np.ndarray (n, 24)
        Series mensuales.
    """
    ids: list[str] = []
    cves: list[str] = []
    series: list[list[float]] = []
    with open(SERIES_CSV, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        mes_cols = [c for c in reader.fieldnames if c.startswith("m_")]
        for row in reader:
            ids.append(row["id_hogar"])
            cves.append(row["cve_municipio"].zfill(5))
            serie = [float(row[c]) for c in mes_cols]
            series.append(serie)
    return ids, cves, np.array(series, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Construye el dataset final de modelado individual."""
    for path in [
        SERIES_CSV,
        PADRON_CSV,
        MUNI_FEATURES_CSV,
        SCORE_REGIONAL_CSV,
    ]:
        if not path.exists():
            print(f"ERROR: no se encontró {path}", file=sys.stderr)
            sys.exit(1)

    print("[1/5] Cargando padrón y series...")
    padron = cargar_padron()
    ids, cves, M = cargar_series_wide()
    print(f"      Hogares: {len(ids):,}")
    print(f"      Tensor mensual: {M.shape}")

    print("[2/5] Cargando features municipales y score regional...")
    municipal = cargar_features_municipales()
    print(f"      Municipios con features: {len(municipal)}")
    n_con_score = sum(1 for v in municipal.values() if "pc_score_regional" in v)
    print(f"      Municipios con score regional: {n_con_score}")

    if not _STATSMODELS_OK:
        print(
            "      AVISO: statsmodels no disponible. `pa_fuerza_estacional` "
            "se calculará con la aproximación de varianza explicada por "
            "promedios mensuales (fallback)."
        )

    # Pre-calcular cuota propuesta para diagnóstico (usando monto base
    # típico). El valor por hogar se calcula dentro del Pilar A.
    print(
        f"[3/5] Crédito hipotético: {MULTIPLO_INGRESO_CREDITO}× ingreso × "
        f"{TIPO_CAMBIO_MXN_POR_USD} MXN/USD, "
        f"{TASA_ANUAL_CREDITO*100:.1f}% anual a {PLAZO_MESES_CREDITO} meses."
    )
    cuota_para_500usd = _calcular_pmt(
        principal=MULTIPLO_INGRESO_CREDITO * 500.0 * TIPO_CAMBIO_MXN_POR_USD,
        tasa_anual=TASA_ANUAL_CREDITO,
        plazo_meses=PLAZO_MESES_CREDITO,
    )
    print(
        f"      Ejemplo: hogar con remesa USD 500 → cuota MXN "
        f"{cuota_para_500usd:,.2f} (USD "
        f"{cuota_para_500usd / TIPO_CAMBIO_MXN_POR_USD:,.2f})"
    )

    print("[4/5] Calculando features de los 3 pilares...")
    registros: list[dict[str, object]] = []
    municipios_sin_features = 0

    for i, id_hogar in enumerate(ids):
        cve = cves[i]
        hogar = padron[id_hogar]

        # Pilar A
        pa = calcular_features_pilar_a(
            serie=M[i, :],
            antiguedad_meses=hogar["antiguedad_recepcion_meses"],
            monto_base_usd=hogar["remesa_mediana_esperada_usd"],
        )

        # Pilar B
        pb = calcular_features_pilar_b(hogar)

        # Pilar C
        if cve in municipal:
            pc = municipal[cve]
        else:
            pc = {}
            municipios_sin_features += 1

        # Imputar 0 para columnas faltantes (raro, pero defensivo).
        pc_default = {
            "pc_iim_dp2": 0.0,
            "pc_pct_viv_remesas": 0.0,
            "pc_pct_viv_emigrantes": 0.0,
            "pc_imn_2020": 0.0,
            "pc_escolaridad_promedio_mun": 0.0,
            "pc_pct_pob_ocupada_2sm": 0.0,
            "pc_score_regional": 0.0,
            "pc_sub_capacidad_pago": 0.0,
            "pc_sub_intensidad_migratoria": 0.0,
        }
        for k, default in pc_default.items():
            if k not in pc:
                pc[k] = default

        registro = {
            "id_hogar": id_hogar,
            "cve_municipio": cve,
            "municipio": hogar["municipio"],
            **pa,
            **pb,
            **pc,
        }
        registros.append(registro)

    if municipios_sin_features > 0:
        print(
            f"      AVISO: {municipios_sin_features} hogares en municipios "
            f"sin features Pilar C; se imputaron 0."
        )

    print(f"      Registros generados: {len(registros):,}")

    # --- Persistencia ---------------------------------------------------
    print("[5/5] Persistiendo dataset y metadatos...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(registros[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registros)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"      Filas: {len(registros):,} × Columnas: {len(fieldnames)}")

    # --- Diagnóstico estadístico ----------------------------------------
    diag = _diagnostico(registros)
    metadata = {
        "descripcion": (
            "Dataset final de modelado individual con features de los "
            "tres pilares: A (patrón temporal del flujo de remesas), "
            "B (demográficas del receptor) y C (contextuales del "
            "municipio incluyendo score regional del Componente 1)."
        ),
        "n_hogares": len(registros),
        "n_columnas_total": len(fieldnames),
        "n_features_pilar_a": sum(1 for c in fieldnames if c.startswith("pa_")),
        "n_features_pilar_b": sum(1 for c in fieldnames if c.startswith("pb_")),
        "n_features_pilar_c": sum(1 for c in fieldnames if c.startswith("pc_")),
        "credito_hipotetico": {
            "multiplo_ingreso_mensual": MULTIPLO_INGRESO_CREDITO,
            "tasa_anual": TASA_ANUAL_CREDITO,
            "plazo_meses": PLAZO_MESES_CREDITO,
            "tipo_cambio_mxn_usd": TIPO_CAMBIO_MXN_POR_USD,
            "fuente_tasa": (
                "CONDUSEF (2024) — tasas representativas de crédito "
                "hipotecario y de mejoramiento de vivienda en pesos."
            ),
            "interpretacion_ratio": (
                "pa_ratio_remesa_cuota = mediana_remesa_USD / cuota_USD. "
                "Ratio < 1.0 → la remesa mediana no cubre la cuota; "
                "ratio ≥ 2.5 → margen holgado."
            ),
        },
        "codificacion_categoricas": {
            "pb_genero_F": "1 si receptor mujer; 0 si hombre.",
            "pb_escolaridad_ord": (
                "Ordinal 0-4 con el orden natural: 0=sin_estudios, "
                "1=primaria, 2=secundaria, 3=media_superior, 4=superior. "
                "Justificada por la monotonía esperada del efecto sobre PD."
            ),
            "pb_vivienda_propia": "1/0 (referencia: prestada).",
            "pb_vivienda_rentada": "1/0 (referencia: prestada).",
        },
        "fuerza_estacional_metodo": (
            "STL (statsmodels) con period=12 y robust=True"
            if _STATSMODELS_OK
            else "Aproximación: Var(promedios mensuales) / Var(serie)"
        ),
        "diccionario_variables": _diccionario_variables(),
        "diagnostico": diag,
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación ---------------------------------------------------
    print("\nVerificación cruzada:")
    print(
        f"      Total features: {len(fieldnames) - 3}  "
        f"(Pilar A: {metadata['n_features_pilar_a']}, "
        f"Pilar B: {metadata['n_features_pilar_b']}, "
        f"Pilar C: {metadata['n_features_pilar_c']})"
    )
    print(
        f"      Mediana de pa_ratio_remesa_cuota: "
        f"{diag['pa_ratio_remesa_cuota']['mediana']:.3f}"
    )
    print(
        f"      P25 / P75: "
        f"{diag['pa_ratio_remesa_cuota']['p25']:.3f} / "
        f"{diag['pa_ratio_remesa_cuota']['p75']:.3f}"
    )
    print(
        f"      Hogares con ratio < 1.0 (cuota > remesa): "
        f"{diag['pa_ratio_remesa_cuota']['n_ratio_menor_1']:,} "
        f"({diag['pa_ratio_remesa_cuota']['pct_ratio_menor_1']:.4f})"
    )
    print(
        f"      Score regional (mediana): "
        f"{diag['pc_score_regional']['mediana']:.3f}"
    )
    print(
        f"      Antigüedad mediana flujo: "
        f"{diag['pa_antiguedad_meses']['mediana']:.1f} meses"
    )
    print(
        f"      Fuerza estacional (mediana): "
        f"{diag['pa_fuerza_estacional']['mediana']:.4f}"
    )
    print(
        f"      Pendiente normalizada (P5/P95): "
        f"[{diag['pa_pendiente_norm']['p5']:.4f}, "
        f"{diag['pa_pendiente_norm']['p95']:.4f}]"
    )

    print("\nFASE 6 completada.")


def _diagnostico(registros: list[dict[str, object]]) -> dict[str, object]:
    """Calcula percentiles y media para todas las columnas numéricas."""
    diag: dict[str, object] = {}
    if not registros:
        return diag

    cols_num = [
        c
        for c, v in registros[0].items()
        if isinstance(v, (int, float)) and c not in ("id_hogar",)
    ]

    for col in cols_num:
        vals = np.array([r[col] for r in registros], dtype=float)
        d: dict[str, float | int] = {
            "media": float(np.mean(vals)),
            "mediana": float(np.median(vals)),
            "desv": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
        }
        if col == "pa_ratio_remesa_cuota":
            d["n_ratio_menor_1"] = int(np.sum(vals < 1.0))
            d["pct_ratio_menor_1"] = float(np.mean(vals < 1.0))
        diag[col] = d

    return diag


def _diccionario_variables() -> dict[str, str]:
    """Diccionario human-readable de las variables del dataset."""
    return {
        "id_hogar": "Identificador único del hogar (Fase 4).",
        "cve_municipio": "Clave INEGI del municipio (5 dígitos).",
        "municipio": "Nombre del municipio.",
        # Pilar A
        "pa_mediana_remesa": (
            "Mediana del monto recibido en meses con envío (USD)."
        ),
        "pa_media_remesa": "Media del monto recibido en meses con envío (USD).",
        "pa_desv_remesa": "Desviación estándar del monto (USD).",
        "pa_cv_remesa": "Coeficiente de variación: desv/media.",
        "pa_pendiente_24m": (
            "Pendiente OLS de la serie de 24 meses (USD/mes). "
            "Positivo = flujo creciente."
        ),
        "pa_pendiente_norm": "Pendiente normalizada por mediana (% mensual).",
        "pa_n_meses_sin_envio": "Conteo de meses sin envío en la ventana.",
        "pa_max_racha_sin_envio": (
            "Mayor número de meses consecutivos sin envío."
        ),
        "pa_fuerza_estacional": (
            "Fuerza del componente estacional en [0,1]."
        ),
        "pa_antiguedad_meses": (
            "Antigüedad reportada del flujo de remesas (Fase 4)."
        ),
        "pa_cuota_propuesta_mxn": (
            "Cuota mensual del crédito hipotético (MXN)."
        ),
        "pa_ratio_remesa_cuota": (
            "Mediana_remesa_USD / cuota_USD. Capacidad de pago."
        ),
        # Pilar B
        "pb_edad": "Edad del receptor (años).",
        "pb_genero_F": "Indicador 1 si receptor mujer.",
        "pb_n_dependientes": "Número de dependientes en el hogar.",
        "pb_escolaridad_ord": "Escolaridad ordinal 0-4.",
        "pb_vivienda_propia": "Indicador 1 si vivienda propia.",
        "pb_vivienda_rentada": "Indicador 1 si vivienda rentada.",
        # Pilar C
        "pc_iim_dp2": "Índice de Intensidad Migratoria CONAPO DP2 (2020).",
        "pc_pct_viv_remesas": (
            "% de viviendas receptoras de remesas en el municipio (INEGI 2020)."
        ),
        "pc_pct_viv_emigrantes": (
            "% de viviendas con emigrantes (INEGI 2020)."
        ),
        "pc_imn_2020": "Índice de marginación normalizado (CONAPO 2020).",
        "pc_escolaridad_promedio_mun": (
            "Escolaridad promedio del municipio en años (INEGI 2020)."
        ),
        "pc_pct_pob_ocupada_2sm": (
            "% de población ocupada con ingreso ≤ 2 salarios mínimos."
        ),
        "pc_score_regional": (
            "Score regional municipal del Componente 1 [0,1]."
        ),
        "pc_sub_capacidad_pago": (
            "Sub-score regional de capacidad de pago."
        ),
        "pc_sub_intensidad_migratoria": (
            "Sub-score regional de intensidad migratoria."
        ),
    }


if __name__ == "__main__":
    main()
