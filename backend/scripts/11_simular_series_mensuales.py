"""
Simulación de las series mensuales individuales de remesas por hogar.

Para cada hogar del padrón sintético (Fase 4) se simula una serie mensual
de 24 meses (2023-01 a 2024-12) que constituye el "historial transaccional"
sobre el cual se construirán las features del Pilar A en Fase 6 y se
entrenará el modelo individual de PD en Fase 8.

Modelo de simulación
--------------------
Para el hogar i en el mes t:

    remesa_{i,t} = M_i × s(mes_t) × ε_{i,t} × ind_envio_{i,t}

donde:

- M_i es el monto base mensual del hogar (`remesa_mediana_esperada_usd`
  del padrón).
- s(mes_t) es el factor estacional empírico de Jalisco estimado en Fase 3
  (escala promedio anual = 1.0).
- ε_{i,t} ~ Lognormal con coeficiente de variación CV = 0.30 (calibrado
  con la dispersión mensual agregada de la serie nacional Banxico
  2013-2024). La media de ε_{i,t} es 1 por construcción para no introducir
  sesgo en el monto base.
- ind_envio_{i,t} ~ Bernoulli(p=0.95) es el indicador de "mes con envío".
  La probabilidad de interrupción mensual de 5% se calibra con BBVA
  Research (Anuario de Migración y Remesas México 2024) que documenta que
  cerca del 5% de los hogares receptores experimentan al menos un mes sin
  envío al año.

Ventana
-------
Todos los hogares se simulan sobre la misma ventana de 24 meses
(2023-01 a 2024-12). La heterogeneidad de antigüedad del flujo por hogar
NO se pierde: se conserva como feature tabular `antiguedad_recepcion_meses`
en el padrón (Fase 4), y servirá como input adicional al modelo individual.

Insumos
-------
data/processed/padron_hogares_sinteticos.csv
data/processed/jalisco_municipal_remesas_mensuales.csv
    (para extraer factores estacionales empíricos de Jalisco)

Salida
------
data/processed/series_mensuales_hogares.csv
    Wide format: una fila por hogar con 24 columnas mensuales más
    metadatos. Estructura compatible con el proyecto previo de credit
    scoring (un registro por sujeto).
data/processed/series_mensuales_hogares_metadata.json
    Parámetros de simulación, distribuciones, sanity checks.

Reproducibilidad
----------------
Seed = 42 con numpy.random.default_rng. La secuencia de muestreo es
determinista respecto al orden de hogares en el padrón.

Uso
---
    python scripts/11_simular_series_mensuales.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PADRON_CSV = (
    PROJECT_ROOT / "data" / "processed" / "padron_hogares_sinteticos.csv"
)
MENSUAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jalisco_municipal_remesas_mensuales.csv"
)
OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "series_mensuales_hogares.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "series_mensuales_hogares_metadata.json"
)

# ---------------------------------------------------------------------------
# Hiperparámetros de la simulación
# ---------------------------------------------------------------------------
SEED = 42

# Ventana de simulación: 24 meses, ene-2023 a dic-2024.
ANIO_INICIO = 2023
ANIO_FIN = 2024
N_MESES = (ANIO_FIN - ANIO_INICIO + 1) * 12

# Coeficiente de variación del shock idiosincrático (lognormal con
# media = 1).
CV_SHOCK = 0.30

# Probabilidad mensual de envío (1 - prob de interrupción).
P_ENVIO = 0.95


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def parse_float(s: str) -> float | None:
    """Convierte string a float manejando vacíos."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def cargar_padron() -> list[dict[str, object]]:
    """Carga el padrón sintético como lista de dicts."""
    hogares: list[dict[str, object]] = []
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
            hogares.append(row)
    return hogares


def calcular_factores_estacionales() -> dict[int, float]:
    """
    Recalcula los factores estacionales mensuales de Jalisco a partir de
    la serie desagregada de Fase 3.

    El factor s(m) es el cociente promedio (sobre los años 2013-2024) del
    valor del mes m respecto al promedio mensual del año correspondiente,
    de modo que el promedio anual de los 12 factores es exactamente 1.0.

    Returns
    -------
    dict[int, float]
        mes (1..12) -> factor estacional.
    """
    by_year: dict[int, list[float]] = {}
    by_year_month: dict[tuple[int, int], float] = {}

    with open(MENSUAL_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            fecha = row["fecha"]  # YYYY-MM-DD
            anio = int(fecha[:4])
            mes = int(fecha[5:7])
            v = parse_float(row["remesas_musd_mensual"])
            if v is None:
                continue
            key = (anio, mes)
            by_year_month[key] = by_year_month.get(key, 0.0) + v
            by_year.setdefault(anio, [])
            # Acumulamos por (anio, mes); luego convertimos.

    # Construir serie agregada Jalisco mensual.
    sums_by_anio_mes: dict[tuple[int, int], float] = by_year_month
    anios = sorted({a for (a, _) in sums_by_anio_mes.keys()})

    factor_acum: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for anio in anios:
        valores_mes = [sums_by_anio_mes.get((anio, m), 0.0) for m in range(1, 13)]
        if all(v == 0.0 for v in valores_mes):
            continue
        promedio = float(np.mean(valores_mes))
        if promedio <= 0:
            continue
        for m, v in enumerate(valores_mes, start=1):
            factor_acum[m].append(v / promedio)

    factores = {m: float(np.mean(factor_acum[m])) for m in range(1, 13)}

    # Renormalizar para garantizar promedio = 1.0 exacto.
    avg = float(np.mean(list(factores.values())))
    factores = {m: factores[m] / avg for m in range(1, 13)}
    return factores


def generar_calendario(n_meses: int) -> list[tuple[int, int]]:
    """Genera lista de (año, mes) para las N_MESES de la ventana."""
    calendario: list[tuple[int, int]] = []
    anio = ANIO_INICIO
    mes = 1
    for _ in range(n_meses):
        calendario.append((anio, mes))
        mes += 1
        if mes > 12:
            mes = 1
            anio += 1
    return calendario


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
def simular_series(
    hogares: list[dict[str, object]],
    factores_estacionales: dict[int, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula las series mensuales para todos los hogares.

    Parameters
    ----------
    hogares : list[dict]
        Padrón sintético.
    factores_estacionales : dict[int, float]
        Factor s(m) por mes calendario (1..12).
    rng : np.random.Generator
        Generador reproducible.

    Returns
    -------
    series : np.ndarray (n_hogares, N_MESES)
        Montos mensuales en USD.
    indicadores : np.ndarray (n_hogares, N_MESES)
        1 si hubo envío, 0 si fue mes sin envío.
    """
    n = len(hogares)
    montos_base = np.array(
        [h["remesa_mediana_esperada_usd"] for h in hogares]
    )

    calendario = generar_calendario(N_MESES)
    s_vec = np.array(
        [factores_estacionales[mes] for (_, mes) in calendario]
    )  # (N_MESES,)

    # Shock lognormal con media = 1 y CV = CV_SHOCK.
    # Si X ~ Lognormal(mu, sigma): E[X] = exp(mu + sigma^2/2), CV =
    # sqrt(exp(sigma^2) - 1). Resolvemos:
    sigma_ln = float(np.sqrt(np.log(1.0 + CV_SHOCK ** 2)))
    mu_ln = -0.5 * sigma_ln ** 2  # garantiza E[X] = 1

    eps = rng.lognormal(mu_ln, sigma_ln, size=(n, N_MESES))

    # Indicadores Bernoulli de envío.
    ind = rng.binomial(1, P_ENVIO, size=(n, N_MESES))

    # Construcción multiplicativa.
    series = (
        montos_base[:, None]  # (n, 1)
        * s_vec[None, :]  # (1, N_MESES)
        * eps  # (n, N_MESES)
        * ind  # (n, N_MESES)
    )

    return series, ind


# ---------------------------------------------------------------------------
# Persistencia y diagnóstico
# ---------------------------------------------------------------------------
def construir_dataset_wide(
    hogares: list[dict[str, object]],
    series: np.ndarray,
    calendario: list[tuple[int, int]],
) -> tuple[list[str], list[list[object]]]:
    """
    Construye el dataset en formato ancho.

    Cada fila es un hogar; cada mes es una columna `m_YYYY_MM` con el
    monto en USD. Se incluyen además `id_hogar`, atributos demográficos
    seleccionados (para que el archivo sea autocontenido) y un total
    acumulado en la ventana.

    Returns
    -------
    fieldnames, rows : tuple
    """
    mes_cols = [f"m_{anio:04d}_{mes:02d}" for anio, mes in calendario]
    fieldnames = [
        "id_hogar",
        "cve_municipio",
        "municipio",
        "edad_receptor",
        "genero_receptor",
        "escolaridad",
        "n_dependientes",
        "tipo_vivienda_actual",
        "antiguedad_recepcion_meses",
        "remesa_mediana_esperada_usd",
        "remesa_total_24m",
        *mes_cols,
    ]

    rows: list[list[object]] = []
    for i, h in enumerate(hogares):
        fila_montos = [round(float(x), 4) for x in series[i, :]]
        total_24m = round(float(series[i, :].sum()), 4)
        rows.append(
            [
                h["id_hogar"],
                h["cve_municipio"],
                h["municipio"],
                h["edad_receptor"],
                h["genero_receptor"],
                h["escolaridad"],
                h["n_dependientes"],
                h["tipo_vivienda_actual"],
                h["antiguedad_recepcion_meses"],
                h["remesa_mediana_esperada_usd"],
                total_24m,
                *fila_montos,
            ]
        )
    return fieldnames, rows


def calcular_diagnostico(
    series: np.ndarray, ind: np.ndarray, hogares: list[dict[str, object]]
) -> dict[str, object]:
    """Estadísticos agregados sobre la simulación."""
    n, T = series.shape

    medianas_hogar = np.median(series, axis=1)  # incluyendo ceros
    medianas_no_cero = np.array(
        [np.median(s[s > 0]) if np.any(s > 0) else 0.0 for s in series]
    )

    cv_hogar = np.array(
        [
            (np.std(s[s > 0]) / np.mean(s[s > 0])) if np.sum(s > 0) > 1 else np.nan
            for s in series
        ]
    )
    cv_hogar = cv_hogar[~np.isnan(cv_hogar)]

    n_envios_hogar = ind.sum(axis=1)
    pct_meses_envio = ind.sum() / (n * T)

    montos_base = np.array(
        [h["remesa_mediana_esperada_usd"] for h in hogares]
    )

    promedio_mensual_jal = series.sum(axis=0) / 1_000_000  # USD millones
    # Suma total Jalisco en USD millones (si fuese todo el universo, no lo
    # es: este es solo el padrón sintético de 10k hogares).

    return {
        "n_hogares": int(n),
        "n_meses": int(T),
        "n_observaciones": int(n * T),
        "pct_meses_con_envio_realizado": float(pct_meses_envio),
        "pct_meses_con_envio_target": P_ENVIO,
        "n_envios_por_hogar": {
            "media": float(np.mean(n_envios_hogar)),
            "mediana": float(np.median(n_envios_hogar)),
            "min": int(np.min(n_envios_hogar)),
            "max": int(np.max(n_envios_hogar)),
            "p25": float(np.percentile(n_envios_hogar, 25)),
            "p75": float(np.percentile(n_envios_hogar, 75)),
        },
        "remesa_mediana_por_hogar_usd": {
            "incluyendo_ceros": {
                "media": float(np.mean(medianas_hogar)),
                "mediana": float(np.median(medianas_hogar)),
                "p25": float(np.percentile(medianas_hogar, 25)),
                "p75": float(np.percentile(medianas_hogar, 75)),
            },
            "excluyendo_ceros": {
                "media": float(np.mean(medianas_no_cero)),
                "mediana": float(np.median(medianas_no_cero)),
                "p25": float(np.percentile(medianas_no_cero, 25)),
                "p75": float(np.percentile(medianas_no_cero, 75)),
            },
        },
        "cv_intra_hogar": {
            "media": float(np.mean(cv_hogar)),
            "mediana": float(np.median(cv_hogar)),
            "target_aprox": CV_SHOCK,
            "n_hogares_con_cv": int(len(cv_hogar)),
        },
        "monto_base_promedio_padron": float(np.mean(montos_base)),
        "monto_realizado_promedio": float(np.mean(series[ind == 1])),
        "agregado_jalisco_24m_usd_millones": float(series.sum() / 1_000_000),
        "promedio_mensual_jalisco_padron_musd": {
            "media": float(np.mean(promedio_mensual_jal)),
            "min": float(np.min(promedio_mensual_jal)),
            "max": float(np.max(promedio_mensual_jal)),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la simulación de series mensuales por hogar."""
    for path in [PADRON_CSV, MENSUAL_CSV]:
        if not path.exists():
            print(f"ERROR: no se encontró {path}", file=sys.stderr)
            sys.exit(1)

    print("[1/4] Cargando padrón sintético...")
    hogares = cargar_padron()
    print(f"      Hogares cargados: {len(hogares):,}")

    print("[2/4] Calculando factores estacionales empíricos de Jalisco...")
    factores = calcular_factores_estacionales()
    print(f"      Pico estacional: mes {max(factores, key=factores.get)} "
          f"({max(factores.values()):.4f})")
    print(f"      Valle estacional: mes {min(factores, key=factores.get)} "
          f"({min(factores.values()):.4f})")
    print(f"      Promedio (debe ser 1.0): {np.mean(list(factores.values())):.6f}")

    print(
        f"[3/4] Simulando series ({len(hogares):,} hogares × "
        f"{N_MESES} meses)..."
    )
    rng = np.random.default_rng(SEED)
    series, ind = simular_series(hogares, factores, rng)
    print(
        f"      Tensor generado: {series.shape} "
        f"({series.size:,} observaciones)"
    )

    print("[4/4] Persistiendo dataset wide y metadatos...")
    calendario = generar_calendario(N_MESES)
    fieldnames, rows = construir_dataset_wide(hogares, series, calendario)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"      Filas: {len(rows):,} × Columnas: {len(fieldnames)}")

    diag = calcular_diagnostico(series, ind, hogares)
    metadata = {
        "descripcion": (
            "Series mensuales individuales simuladas para los 10,000 "
            "hogares receptores del padrón sintético. Ventana fija de "
            f"{N_MESES} meses ({ANIO_INICIO}-01 a {ANIO_FIN}-12). "
            "Modelo multiplicativo con monto base por hogar, factor "
            "estacional empírico de Jalisco (Fase 3), shock lognormal "
            "y indicador de envío Bernoulli."
        ),
        "n_hogares": len(hogares),
        "n_meses": N_MESES,
        "ventana": f"{ANIO_INICIO}-01 a {ANIO_FIN}-12",
        "seed": SEED,
        "modelo_simulacion": (
            "remesa_{i,t} = M_i × s(mes_t) × eps_{i,t} × ind_envio_{i,t}"
        ),
        "componentes": {
            "M_i": "remesa_mediana_esperada_usd del padrón (Fase 4).",
            "s(mes_t)": (
                "Factor estacional empírico de Jalisco estimado en Fase 3 "
                "(promedio anual = 1.0)."
            ),
            "eps_{i,t}": (
                f"Lognormal con media=1, CV={CV_SHOCK}. mu_ln = -sigma_ln^2/2; "
                "sigma_ln = sqrt(log(1+CV^2))."
            ),
            "ind_envio_{i,t}": (
                f"Bernoulli({P_ENVIO}) i.i.d. mes a mes."
            ),
        },
        "calibracion": {
            "CV_shock": {
                "valor": CV_SHOCK,
                "fuente": (
                    "Calibrado con la dispersión mensual de la serie "
                    "nacional Banxico SE27803 2013-2024 (variabilidad "
                    "intra-anual residual del flujo)."
                ),
            },
            "P_envio": {
                "valor": P_ENVIO,
                "fuente": (
                    "BBVA Research, Anuario de Migración y Remesas "
                    "México 2024 — proporción de hogares receptores con "
                    "interrupciones mensuales del flujo."
                ),
            },
            "estacionalidad": {
                "fuente": (
                    "Fase 3 — desagregación Chow-Lin AR(1) de Banxico CE166 "
                    "trimestral con indicadora SE27803."
                ),
                "factores": factores,
            },
        },
        "formato_salida": (
            "Wide format: 1 fila por hogar, 11 columnas de metadatos "
            f"+ {N_MESES} columnas mensuales m_YYYY_MM. Compatible con "
            "modelos tabulares (LogReg, XGBoost) sin reshape; para LSTM "
            "se debe reshape a (n_hogares, n_meses, 1) en Fase 8."
        ),
        "diagnostico": diag,
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación cruzada -------------------------------------------
    print("\nVerificación cruzada:")
    print(
        f"      % meses con envío: "
        f"{diag['pct_meses_con_envio_realizado']:.4f} "
        f"(target: {P_ENVIO})"
    )
    print(
        f"      Envíos por hogar: media = "
        f"{diag['n_envios_por_hogar']['media']:.2f} de {N_MESES} meses "
        f"(esperado ≈ {N_MESES * P_ENVIO:.1f})"
    )
    print(
        f"      CV intra-hogar (excluyendo ceros): "
        f"media = {diag['cv_intra_hogar']['media']:.4f} "
        f"(target ≈ {CV_SHOCK})"
    )
    print(
        f"      Monto base promedio padrón: USD "
        f"{diag['monto_base_promedio_padron']:.2f}"
    )
    print(
        f"      Monto realizado promedio (en meses con envío): USD "
        f"{diag['monto_realizado_promedio']:.2f}"
    )
    print(
        f"      Agregado Jalisco (padrón) 24m: USD "
        f"{diag['agregado_jalisco_24m_usd_millones']:.2f} M"
    )
    print(
        f"      (este es el agregado del padrón de 10k hogares, NO del "
        f"universo total)"
    )

    print("\nFASE 5 completada.")


if __name__ == "__main__":
    main()
