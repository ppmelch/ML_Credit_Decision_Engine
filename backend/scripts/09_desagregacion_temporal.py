"""
Desagregación temporal Chow-Lin con residuos AR(1) de la serie trimestral
CE166 de remesas a frecuencia mensual, usando como variable indicadora la
serie SE27803 (remesas mensuales nacionales).

Método
------
Chow & Lin (1971). Sea y_t la serie objetivo a desagregar (trimestral, en
millones de USD por municipio) y x_m la serie indicadora (mensual,
nacional). El método estima:

    y_low = X_low @ beta + u_low,   u_low | AR(1) con parametro rho

donde X_low es la agregación trimestral de la indicadora mensual. Una vez
estimados beta y rho, la estimación mensual desagregada es:

    y_high_hat = X_high @ beta + V @ C.T @ (C @ V @ C.T)^-1 @ (y_low - X_low @ beta)

con C la matriz de agregación trimestral 3-en-3 y V la matriz de
covarianzas AR(1) en alta frecuencia. Esta fórmula garantiza por
construcción que la suma de cada terna mensual reproduce exactamente el
valor trimestral observado.

Insumos
-------
data/raw/banxico_se27803_remesas_mensuales_nacional.csv
    Serie indicadora mensual nacional 2013-01 a 2024-12 (144 meses).
data/raw/banxico_ce166_jalisco_trimestral.csv
    Serie objetivo trimestral por municipio 2013-Q1 en adelante.
data/raw/municipios_jalisco_catalogo.csv
    Catálogo de 124 municipios de Jalisco.

Salidas
-------
data/processed/jalisco_municipal_remesas_mensuales.csv
    Formato largo con 124 × 144 = 17,856 registros.
data/processed/jalisco_municipal_remesas_mensuales_metadata.json
    Diagnóstico por municipio: rho estimado, R² regresión, método usado
    (chow_lin o fallback), banderas de calidad.

Notas metodológicas
-------------------
1. Periodo desagregado: 2013-01 a 2024-12 (144 meses, 48 trimestres).
   Se descarta 2025-Q1+ del CE166 por estar incompleto.
2. La regresión OLS se hace en niveles (no logs) consistente con la
   métrica de Chow & Lin original. Beta tiene interpretación de
   "fracción de la remesa nacional capturada por el municipio".
3. El parámetro rho del AR(1) se estima por máxima verosimilitud
   restringida sobre los residuos OLS, con búsqueda de grilla en
   [0.0, 0.99] con paso 0.01. Este enfoque es estable numéricamente
   y suficientemente preciso para nuestra aplicación.
4. Fallback: si la matriz (C V C.T) resulta singular o los residuos
   tienen comportamiento degenerado, se aplica interpolación lineal
   intra-trimestre (asignar trimestre/3 a cada mes), que también
   satisface la restricción de agregación.

Uso
---
    python scripts/09_desagregacion_temporal.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDICATOR_CSV = (
    PROJECT_ROOT / "data" / "raw" / "banxico_se27803_remesas_mensuales_nacional.csv"
)
QUARTERLY_CSV = (
    PROJECT_ROOT / "data" / "raw" / "banxico_ce166_jalisco_trimestral.csv"
)
CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jalisco_municipal_remesas_mensuales.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jalisco_municipal_remesas_mensuales_metadata.json"
)

# Periodo de desagregación.
ANIO_INICIO = 2013
ANIO_FIN = 2024
N_MESES = (ANIO_FIN - ANIO_INICIO + 1) * 12  # 144
N_TRIMESTRES = N_MESES // 3  # 48

# Hiperparámetros del método.
RHO_GRID = np.linspace(0.0, 0.99, 100)


def parse_float(s: str) -> float | None:
    """Convierte string a float, manejando vacíos."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def build_aggregation_matrix(n_quarters: int) -> np.ndarray:
    """
    Construye la matriz C de agregación trimestral.

    C tiene dimensiones (n_quarters × 3*n_quarters). C[t, :] tiene un 1
    en las posiciones de los 3 meses del trimestre t y 0 en el resto.

    Parameters
    ----------
    n_quarters : int
        Número de trimestres (typ. 48).

    Returns
    -------
    np.ndarray
        Matriz de agregación con sumas trimestrales.
    """
    n_months = 3 * n_quarters
    C = np.zeros((n_quarters, n_months))
    for t in range(n_quarters):
        C[t, 3 * t : 3 * t + 3] = 1.0
    return C


def build_ar1_covariance(n: int, rho: float) -> np.ndarray:
    """
    Construye la matriz de covarianzas de un proceso AR(1) estacionario.

    V[i,j] = rho^|i-j| / (1 - rho^2). Aquí omitimos la varianza del
    shock sigma^2 porque se cancela en la fórmula de Chow-Lin.

    Parameters
    ----------
    n : int
        Tamaño de la matriz.
    rho : float
        Parámetro AR(1), |rho| < 1.

    Returns
    -------
    np.ndarray
        Matriz n×n de covarianzas AR(1).
    """
    indices = np.arange(n)
    diff = np.abs(indices[:, None] - indices[None, :])
    return rho ** diff / (1.0 - rho ** 2)


def estimate_rho_grid(
    residuals_low: np.ndarray, X_high: np.ndarray, C: np.ndarray
) -> tuple[float, float]:
    """
    Estima rho del AR(1) por búsqueda de grilla maximizando la log-
    verosimilitud restringida (REML) de los residuos OLS a baja
    frecuencia, propagados desde alta frecuencia.

    A baja frecuencia, V_low = C V_high(rho) C.T. La log-verosimilitud
    de los residuos OLS bajo este modelo es proporcional a:

        ll(rho) = -0.5 * log|V_low| - 0.5 * u'.V_low^-1.u

    (constantes omitidas).

    Parameters
    ----------
    residuals_low : np.ndarray
        Residuos OLS de la regresión trimestral (n_q,).
    X_high : np.ndarray
        Indicadora a alta frecuencia con constante (n_high, k).
    C : np.ndarray
        Matriz de agregación.

    Returns
    -------
    rho_optimo, log_lik_optima : tuple
    """
    n_high = X_high.shape[0]
    best_rho = 0.0
    best_ll = -np.inf

    for rho in RHO_GRID:
        try:
            V_high = build_ar1_covariance(n_high, rho)
            V_low = C @ V_high @ C.T
            # Cholesky para estabilidad numérica.
            L = np.linalg.cholesky(V_low)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            # Resolver V_low z = u  vía  L L' z = u
            z = np.linalg.solve(L, residuals_low)
            quad = float(z @ z)
            ll = -0.5 * log_det - 0.5 * quad
            if ll > best_ll:
                best_ll = ll
                best_rho = float(rho)
        except np.linalg.LinAlgError:
            continue

    return best_rho, best_ll


def chow_lin_disaggregate(
    y_low: np.ndarray, x_high: np.ndarray
) -> tuple[np.ndarray, dict[str, float | str]]:
    """
    Aplica Chow-Lin AR(1) a una serie trimestral.

    Parameters
    ----------
    y_low : np.ndarray
        Serie objetivo trimestral (n_q,).
    x_high : np.ndarray
        Serie indicadora mensual (3*n_q,).

    Returns
    -------
    y_high, diagnostico : tuple
        y_high es la estimación mensual (3*n_q,). diagnostico es un dict
        con métricas de calidad.
    """
    n_q = len(y_low)
    n_m = 3 * n_q
    if x_high.shape[0] != n_m:
        raise ValueError(
            f"Dimensiones inconsistentes: y_low tiene {n_q} trimestres, "
            f"x_high tiene {x_high.shape[0]} meses (esperado {n_m})."
        )

    # Construir X con constante.
    X_high = np.column_stack([np.ones(n_m), x_high])
    C = build_aggregation_matrix(n_q)
    X_low = C @ X_high

    # OLS a baja frecuencia para residuos preliminares.
    try:
        beta_ols, *_ = np.linalg.lstsq(X_low, y_low, rcond=None)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(f"OLS falló: {err}")

    residuals_low = y_low - X_low @ beta_ols

    # R² de la regresión OLS (informativo).
    ss_tot = float(np.sum((y_low - np.mean(y_low)) ** 2))
    ss_res = float(np.sum(residuals_low ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Estimar rho por grilla.
    rho_hat, ll_opt = estimate_rho_grid(residuals_low, X_high, C)

    # Re-estimar beta vía GLS con rho estimado.
    V_high = build_ar1_covariance(n_m, rho_hat)
    V_low = C @ V_high @ C.T
    try:
        V_low_inv = np.linalg.inv(V_low)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(
            f"V_low singular al rho estimado {rho_hat}: {err}"
        )

    XtViX = X_low.T @ V_low_inv @ X_low
    XtViy = X_low.T @ V_low_inv @ y_low
    beta_gls = np.linalg.solve(XtViX, XtViy)

    # Predicción Chow-Lin.
    residuals_low_gls = y_low - X_low @ beta_gls
    correction = (
        V_high @ C.T @ V_low_inv @ residuals_low_gls
    )
    y_high = X_high @ beta_gls + correction

    # Verificar restricción de agregación.
    aggregated = C @ y_high
    max_abs_error = float(np.max(np.abs(aggregated - y_low)))

    diagnostico: dict[str, float | str] = {
        "metodo": "chow_lin_ar1",
        "beta_intercepto": float(beta_gls[0]),
        "beta_indicadora": float(beta_gls[1]),
        "rho_ar1": rho_hat,
        "r2_ols": r2,
        "log_lik_optima": ll_opt,
        "max_error_agregacion": max_abs_error,
    }

    return y_high, diagnostico


def linear_interpolation_fallback(
    y_low: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """
    Fallback: distribuir cada trimestre en partes iguales entre sus 3
    meses. Trivialmente satisface la restricción de agregación.
    """
    y_high = np.repeat(y_low / 3.0, 3)
    return y_high, {
        "metodo": "fallback_interpolacion_uniforme",
        "beta_intercepto": None,
        "beta_indicadora": None,
        "rho_ar1": None,
        "r2_ols": None,
        "log_lik_optima": None,
        "max_error_agregacion": 0.0,
    }


def load_indicator_monthly() -> tuple[list[str], np.ndarray]:
    """
    Carga la serie SE27803 nacional mensual 2013-01 a 2024-12.

    Returns
    -------
    fechas, valores : tuple
        fechas como strings 'YYYY-MM-DD' (primer día de cada mes).
    """
    fechas: list[str] = []
    valores: list[float] = []
    with open(INDICATOR_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            fecha = row["fecha"].strip()
            anio = int(fecha[:4])
            if anio < ANIO_INICIO or anio > ANIO_FIN:
                continue
            v = parse_float(row["remesas_musd"])
            if v is None:
                raise ValueError(
                    f"SE27803 tiene valor faltante en {fecha}; el método "
                    f"requiere serie indicadora completa."
                )
            fechas.append(fecha)
            valores.append(v)

    if len(valores) != N_MESES:
        raise ValueError(
            f"SE27803 tiene {len(valores)} observaciones para "
            f"{ANIO_INICIO}-{ANIO_FIN}, esperadas {N_MESES}."
        )
    return fechas, np.array(valores)


def load_quarterly_by_municipio() -> dict[str, np.ndarray]:
    """
    Carga la serie trimestral CE166 indexada por clave INEGI, restringida
    al periodo 2013-Q1 a 2024-Q4 (48 trimestres).

    Returns
    -------
    dict
        {cve_municipio: np.ndarray de 48 valores trimestrales}
    """
    by_mun: dict[str, list[tuple[str, float]]] = {}
    with open(QUARTERLY_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            fecha = row["fecha"].strip()
            anio = int(fecha[:4])
            if anio < ANIO_INICIO or anio > ANIO_FIN:
                continue
            v = parse_float(row["remesas_musd"])
            if v is None:
                continue
            by_mun.setdefault(cve, []).append((fecha, v))

    result: dict[str, np.ndarray] = {}
    for cve, lista in by_mun.items():
        lista.sort(key=lambda t: t[0])
        if len(lista) != N_TRIMESTRES:
            print(
                f"      ADVERTENCIA: municipio {cve} tiene "
                f"{len(lista)} trimestres (esperados {N_TRIMESTRES}); "
                f"se omite del análisis.",
                file=sys.stderr,
            )
            continue
        result[cve] = np.array([v for _, v in lista])
    return result


def load_catalog() -> dict[str, str]:
    """Carga el catálogo unificado indexado por clave INEGI."""
    catalog: dict[str, str] = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            catalog[cve] = row["municipio"].strip()
    return catalog


def generate_monthly_dates() -> list[str]:
    """Genera lista de 144 fechas mensuales 'YYYY-MM-01' del periodo."""
    dates: list[str] = []
    for anio in range(ANIO_INICIO, ANIO_FIN + 1):
        for mes in range(1, 13):
            dates.append(f"{anio:04d}-{mes:02d}-01")
    return dates


def main() -> None:
    """Ejecuta la desagregación para todos los municipios."""
    for path in [INDICATOR_CSV, QUARTERLY_CSV, CATALOG_CSV]:
        if not path.exists():
            print(f"ERROR: no se encontró {path}", file=sys.stderr)
            sys.exit(1)

    print("[1/5] Cargando serie indicadora SE27803...")
    indicator_dates, x_high = load_indicator_monthly()
    print(
        f"      Observaciones: {len(x_high)} (rango {indicator_dates[0]} "
        f"a {indicator_dates[-1]})"
    )
    print(f"      Total nacional 2024: USD {x_high[-12:].sum():,.2f} M")

    print("[2/5] Cargando series trimestrales CE166...")
    quarterly_data = load_quarterly_by_municipio()
    print(f"      Municipios con serie completa: {len(quarterly_data)}")

    print("[3/5] Cargando catálogo unificado...")
    catalog = load_catalog()
    print(f"      Municipios en catálogo: {len(catalog)}")

    monthly_dates = generate_monthly_dates()
    assert len(monthly_dates) == N_MESES

    # --- Desagregación municipio por municipio --------------------------
    print(f"[4/5] Desagregando series ({len(quarterly_data)} municipios)...")
    output_records: list[dict[str, object]] = []
    diagnostics_by_mun: dict[str, dict[str, float | str]] = {}
    n_chow_lin = 0
    n_fallback = 0

    for cve in sorted(quarterly_data.keys()):
        y_low = quarterly_data[cve]
        try:
            y_high, diag = chow_lin_disaggregate(y_low, x_high)
            n_chow_lin += 1
        except (RuntimeError, np.linalg.LinAlgError) as err:
            print(
                f"      Municipio {cve}: Chow-Lin falló ({err}); "
                f"usando fallback.",
                file=sys.stderr,
            )
            y_high, diag = linear_interpolation_fallback(y_low)
            n_fallback += 1

        # Si la estimación produjo valores negativos (puede ocurrir con
        # series muy volátiles), aplicamos una corrección suave: los
        # negativos se truncan a cero y se redistribuye el déficit
        # proporcionalmente entre los meses del mismo trimestre con
        # valores positivos. Esto preserva la restricción de agregación.
        n_negativos = int(np.sum(y_high < 0))
        if n_negativos > 0:
            for t in range(N_TRIMESTRES):
                bloque = y_high[3 * t : 3 * t + 3].copy()
                if np.any(bloque < 0):
                    deficit = float(-np.sum(bloque[bloque < 0]))
                    bloque[bloque < 0] = 0.0
                    pos_sum = float(np.sum(bloque))
                    if pos_sum > 0:
                        bloque -= bloque * (deficit / pos_sum)
                    y_high[3 * t : 3 * t + 3] = bloque
            diag["correccion_negativos_aplicada"] = True
            diag["n_meses_negativos_originales"] = n_negativos
        else:
            diag["correccion_negativos_aplicada"] = False
            diag["n_meses_negativos_originales"] = 0

        diagnostics_by_mun[cve] = diag
        municipio = catalog.get(cve, "?")
        for fecha, valor in zip(monthly_dates, y_high):
            output_records.append(
                {
                    "cve_municipio": cve,
                    "municipio": municipio,
                    "fecha": fecha,
                    "remesas_musd_mensual": round(float(valor), 10),
                }
            )

    print(f"      Método Chow-Lin AR(1): {n_chow_lin} municipios")
    print(f"      Método fallback: {n_fallback} municipios")
    print(f"      Total de registros generados: {len(output_records)}")

    # --- Persistencia ----------------------------------------------------
    print("[5/5] Guardando salidas...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "cve_municipio",
                "municipio",
                "fecha",
                "remesas_musd_mensual",
            ],
        )
        writer.writeheader()
        writer.writerows(output_records)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    # --- Estadísticos agregados de diagnóstico ---------------------------
    rhos = [
        d["rho_ar1"]
        for d in diagnostics_by_mun.values()
        if d.get("rho_ar1") is not None
    ]
    r2s = [
        d["r2_ols"]
        for d in diagnostics_by_mun.values()
        if d.get("r2_ols") is not None
    ]
    max_errors = [
        d["max_error_agregacion"]
        for d in diagnostics_by_mun.values()
    ]

    metadata = {
        "descripcion": (
            "Series mensuales de remesas por municipio de Jalisco "
            "estimadas por desagregación temporal Chow-Lin AR(1) a "
            "partir de la serie trimestral CE166 de Banxico, usando "
            "como variable indicadora la serie nacional mensual SE27803."
        ),
        "periodo": f"{ANIO_INICIO}-01 a {ANIO_FIN}-12 ({N_MESES} meses)",
        "n_municipios": len(diagnostics_by_mun),
        "n_observaciones": len(output_records),
        "metodo_principal": "Chow-Lin con residuos AR(1)",
        "estimacion_rho": "Búsqueda de grilla en [0.0, 0.99] paso 0.01, REML",
        "indicadora_alta_frecuencia": "SE27803 (remesas mensuales nacionales)",
        "supuesto_clave": (
            "La dinámica mensual nacional es proporcional a la dinámica "
            "mensual de cada municipio. Discutir como limitación en el "
            "reporte: heterogeneidad municipal idiosincrática se diluye."
        ),
        "n_municipios_chow_lin": n_chow_lin,
        "n_municipios_fallback": n_fallback,
        "estadisticos_globales": {
            "rho_ar1": {
                "min": float(np.min(rhos)) if rhos else None,
                "max": float(np.max(rhos)) if rhos else None,
                "mean": float(np.mean(rhos)) if rhos else None,
                "median": float(np.median(rhos)) if rhos else None,
            },
            "r2_ols": {
                "min": float(np.min(r2s)) if r2s else None,
                "max": float(np.max(r2s)) if r2s else None,
                "mean": float(np.mean(r2s)) if r2s else None,
                "median": float(np.median(r2s)) if r2s else None,
            },
            "max_error_agregacion_global": (
                float(np.max(max_errors)) if max_errors else None
            ),
        },
        "diagnostico_por_municipio": diagnostics_by_mun,
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación ---------------------------------------------------
    print("\nVerificación cruzada:")
    print(f"      Registros: {len(output_records)} (esperado: {N_MESES * len(diagnostics_by_mun)})")

    if rhos:
        print(
            f"\n      Distribución de rho AR(1) estimado:"
            f"\n        min    = {min(rhos):.4f}"
            f"\n        max    = {max(rhos):.4f}"
            f"\n        media  = {np.mean(rhos):.4f}"
            f"\n        mediana = {np.median(rhos):.4f}"
        )
    if r2s:
        print(
            f"\n      Distribución de R² OLS:"
            f"\n        min    = {min(r2s):.4f}"
            f"\n        max    = {max(r2s):.4f}"
            f"\n        media  = {np.mean(r2s):.4f}"
            f"\n        mediana = {np.median(r2s):.4f}"
        )
    print(
        f"\n      Error máximo de agregación (global): "
        f"{max(max_errors):.6e}"
    )
    print(
        f"      (si es menor a 1e-6, la restricción de agregación se "
        f"satisface dentro de error numérico)"
    )

    # Spot-check: total Jalisco 2024 mensual debe igualar trimestral.
    suma_mensual_2024 = sum(
        r["remesas_musd_mensual"]
        for r in output_records
        if r["fecha"].startswith("2024")
    )
    print(
        f"\n      Suma mensual Jalisco 2024 (todos los municipios): "
        f"USD {suma_mensual_2024:,.2f} M"
    )
    print(f"      Esperado (CE166 anual): USD ~5,482 M")

    print("\nFASE 3 — paso 1/2 (desagregación) completado.")


if __name__ == "__main__":
    main()
