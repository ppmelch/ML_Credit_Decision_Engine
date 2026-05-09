"""
Construye el score regional municipal del Componente 1 del modelo de
crédito remesa. El score se interpreta como una calificación de
favorabilidad regional para el otorgamiento de crédito hipotecario a
hogares receptores de remesas.

Metodología
-----------
Se calculan tres sub-scores para cada municipio, cada uno normalizado
al rango [0, 1] con escalado min-max sobre los 124 municipios:

1. Capacidad de pago regional (peso 40%):
   Combina el nivel y estabilidad del flujo de remesas con el porcentaje
   de viviendas receptoras. Componentes:
       - Mediana de remesas trimestrales 2020-2024 (mayor = mejor).
       - Estabilidad temporal: 1 - normalize(coef_variacion).
       - % de viviendas receptoras de remesas (CONAPO IIM).
   Promedio simple de los tres componentes normalizados.

2. Calidad socioeconómica regional (peso 40%):
   Refleja la fortaleza estructural del entorno. Componentes:
       - Inversión del índice de marginación (1 - normalize(im_2020)),
         de modo que mayor sea mejor.
       - Escolaridad promedio (mayor = mejor).
       - Tasa de ocupación (mayor = mejor).
       - % viviendas con internet (mayor = mejor).
   Promedio simple de los cuatro componentes normalizados.

3. Intensidad migratoria estructural (peso 20%):
   Captura el grado de incorporación del fenómeno migratorio en el
   tejido social. Componentes:
       - Índice IIM continuo de CONAPO (mayor = más migratorio = más
         remesas potenciales).
   Un único componente normalizado.

Score final (en [0, 1]):
    score = 0.40 × cap_pago + 0.40 × calidad_socio + 0.20 × intensidad

Justificación de pesos
----------------------
La estructura 40/40/20 sigue la lógica estándar en credit scoring para
modelos de crédito hipotecario: capacidad de pago (40%) y solvencia
estructural del entorno (40%) son los dos pilares principales según el
Consumer Financial Protection Bureau (CFPB), mientras que la
intensidad migratoria opera como ajuste contextual (20%) específico al
caso de hogares receptores de remesas. Estos pesos son justificables
desde la literatura citada en el reporte final y pueden recalibrarse
en una segunda iteración tras validación empírica.

Validación
----------
El score se valida cruzándolo con dos categorizaciones oficiales:
    - gim_dp2 (CONAPO IIM): se espera correlación positiva con
      cap_pago y intensidad.
    - gm_2020 (CONAPO Marginación): se espera correlación negativa
      con calidad_socio.

Insumos
-------
data/processed/jalisco_municipal_features.csv

Salidas
-------
data/processed/jalisco_score_regional.csv
data/processed/jalisco_score_regional_metadata.json

Uso
---
    python scripts/08_construir_score_regional.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "jalisco_municipal_features.csv"
)
OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "jalisco_score_regional.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jalisco_score_regional_metadata.json"
)

# Pesos del score compuesto.
W_CAPACIDAD_PAGO = 0.40
W_CALIDAD_SOCIO = 0.40
W_INTENSIDAD = 0.20


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


def min_max_normalize(
    values: list[float | None], invert: bool = False
) -> list[float | None]:
    """
    Normaliza una lista al rango [0, 1] usando min-max scaling.

    Parameters
    ----------
    values : list of float or None
        Valores originales. None se preserva como None.
    invert : bool
        Si True, devuelve 1 - normalize(x), de modo que el valor más
        alto en input se mapee a 0 (útil para variables donde alto = malo).

    Returns
    -------
    list of float or None
        Valores normalizados.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return [None] * len(values)
    vmin = min(valid)
    vmax = max(valid)
    if vmax == vmin:
        # Sin variabilidad: devolver 0.5 a todos los válidos.
        return [None if v is None else 0.5 for v in values]

    out: list[float | None] = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        norm = (v - vmin) / (vmax - vmin)
        if invert:
            norm = 1.0 - norm
        out.append(norm)
    return out


def safe_mean(values: list[float | None]) -> float | None:
    """Promedio simple ignorando None. Devuelve None si todos son None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def main() -> None:
    """Calcula el score regional y persiste el resultado."""
    if not INPUT_CSV.exists():
        print(
            f"ERROR: no se encontró {INPUT_CSV}. "
            f"Corre primero scripts/07_unificar_features_municipales.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[1/4] Cargando tabla de features municipales...")
    rows: list[dict[str, object]] = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    print(f"      Filas cargadas: {len(rows)}")

    # --- Extraer columnas que entran al score como listas paralelas ----
    print("[2/4] Normalizando variables (min-max)...")

    # Capacidad de pago.
    mediana_remesas = [parse_float(r["mediana_remesas_2020_2024"]) for r in rows]
    cv_remesas = [parse_float(r["cv_remesas_2020_2024"]) for r in rows]
    pct_viv_remesas = [parse_float(r["pct_viv_remesas"]) for r in rows]

    # Calidad socioeconómica.
    im_2020 = [parse_float(r["im_2020"]) for r in rows]
    escolaridad = [parse_float(r["escolaridad_promedio"]) for r in rows]
    tasa_ocupacion = [parse_float(r["tasa_ocupacion"]) for r in rows]
    pct_internet = [parse_float(r["pct_viv_internet"]) for r in rows]

    # Intensidad migratoria.
    iim_dp2 = [parse_float(r["iim_dp2"]) for r in rows]

    # Normalizar.
    n_mediana = min_max_normalize(mediana_remesas)
    n_estabilidad = min_max_normalize(cv_remesas, invert=True)
    n_pct_remesas = min_max_normalize(pct_viv_remesas)
    n_marg_inv = min_max_normalize(im_2020, invert=True)
    n_escolaridad = min_max_normalize(escolaridad)
    n_ocupacion = min_max_normalize(tasa_ocupacion)
    n_internet = min_max_normalize(pct_internet)
    n_iim = min_max_normalize(iim_dp2)

    # --- Construir sub-scores --------------------------------------------
    print("[3/4] Computando sub-scores y score final...")
    output_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        cap_pago = safe_mean(
            [n_mediana[i], n_estabilidad[i], n_pct_remesas[i]]
        )
        calidad = safe_mean(
            [n_marg_inv[i], n_escolaridad[i], n_ocupacion[i], n_internet[i]]
        )
        intensidad = n_iim[i]

        # Score final: promedio ponderado, omitiendo subscores faltantes
        # con re-normalización de pesos disponibles.
        contributions = []
        weights = []
        if cap_pago is not None:
            contributions.append(cap_pago * W_CAPACIDAD_PAGO)
            weights.append(W_CAPACIDAD_PAGO)
        if calidad is not None:
            contributions.append(calidad * W_CALIDAD_SOCIO)
            weights.append(W_CALIDAD_SOCIO)
        if intensidad is not None:
            contributions.append(intensidad * W_INTENSIDAD)
            weights.append(W_INTENSIDAD)

        if weights:
            score = sum(contributions) / sum(weights)
        else:
            score = None

        output_rows.append(
            {
                "cve_municipio": row["cve_municipio"],
                "municipio": row["municipio"],
                "sub_score_capacidad_pago": (
                    round(cap_pago, 6) if cap_pago is not None else ""
                ),
                "sub_score_calidad_socioeconomica": (
                    round(calidad, 6) if calidad is not None else ""
                ),
                "sub_score_intensidad_migratoria": (
                    round(intensidad, 6) if intensidad is not None else ""
                ),
                "score_regional": (
                    round(score, 6) if score is not None else ""
                ),
                "gim_dp2": row.get("gim_dp2", ""),
                "gm_2020": row.get("gm_2020", ""),
            }
        )

    # Ordenar por score descendente para que la cima sea visible.
    output_rows.sort(
        key=lambda r: (
            r["score_regional"] if r["score_regional"] != "" else -1
        ),
        reverse=True,
    )

    # --- Persistencia ---------------------------------------------------
    print("[4/4] Guardando salidas...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cve_municipio",
        "municipio",
        "sub_score_capacidad_pago",
        "sub_score_calidad_socioeconomica",
        "sub_score_intensidad_migratoria",
        "score_regional",
        "gim_dp2",
        "gm_2020",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    metadata = {
        "descripcion": (
            "Score regional municipal del Componente 1 del modelo de "
            "crédito remesa, en escala [0, 1] donde mayor = más favorable "
            "para otorgamiento de crédito hipotecario."
        ),
        "n_municipios": len(output_rows),
        "metodologia": {
            "pesos": {
                "capacidad_pago": W_CAPACIDAD_PAGO,
                "calidad_socioeconomica": W_CALIDAD_SOCIO,
                "intensidad_migratoria": W_INTENSIDAD,
            },
            "componentes": {
                "capacidad_pago": [
                    "mediana_remesas_2020_2024 (normalizada)",
                    "estabilidad = 1 - cv_remesas_2020_2024 (normalizada)",
                    "pct_viv_remesas (normalizada)",
                ],
                "calidad_socioeconomica": [
                    "1 - im_2020 (normalizada, invertida)",
                    "escolaridad_promedio (normalizada)",
                    "tasa_ocupacion (normalizada)",
                    "pct_viv_internet (normalizada)",
                ],
                "intensidad_migratoria": ["iim_dp2 (normalizada)"],
            },
            "normalizacion": "Min-max scaling sobre los 124 municipios",
            "ponderacion": (
                "Promedio ponderado de los tres sub-scores. Si algún "
                "sub-score es nulo, los pesos se re-normalizan."
            ),
        },
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación ----------------------------------------------------
    print("\nVerificación cruzada:")

    scores = [
        r["score_regional"]
        for r in output_rows
        if r["score_regional"] != ""
    ]
    print(f"      Scores válidos: {len(scores)}")
    print(
        f"      min   = {min(scores):.4f}"
        f"\n      max   = {max(scores):.4f}"
        f"\n      media = {sum(scores)/len(scores):.4f}"
        f"\n      mediana = {statistics.median(scores):.4f}"
    )

    print(f"\n      Top 10 municipios (mejor score regional):")
    for r in output_rows[:10]:
        print(
            f"        {r['cve_municipio']} {r['municipio']:<28}  "
            f"score={r['score_regional']:.4f}  "
            f"(IIM:{r['gim_dp2']:>9} | Marg:{r['gm_2020']:>9})"
        )

    print(f"\n      Bottom 10 municipios (peor score regional):")
    for r in output_rows[-10:]:
        print(
            f"        {r['cve_municipio']} {r['municipio']:<28}  "
            f"score={r['score_regional']:.4f}  "
            f"(IIM:{r['gim_dp2']:>9} | Marg:{r['gm_2020']:>9})"
        )

    # Validación cruzada sencilla: media del score por categoría de
    # marginación. Esperamos: Muy bajo > Bajo > Medio > Alto > Muy alto.
    print(f"\n      Score promedio por grado de marginación:")
    by_marg: dict[str, list[float]] = {}
    for r in output_rows:
        gm = r["gm_2020"] or "(sin)"
        if r["score_regional"] != "":
            by_marg.setdefault(gm, []).append(r["score_regional"])
    for gm in ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]:
        if gm in by_marg:
            vals = by_marg[gm]
            print(
                f"        {gm:>10}: {sum(vals)/len(vals):.4f} "
                f"(n={len(vals)})"
            )

    print(f"\n      Score promedio por grado de intensidad migratoria:")
    by_iim: dict[str, list[float]] = {}
    for r in output_rows:
        gi = r["gim_dp2"] or "(sin)"
        if r["score_regional"] != "":
            by_iim.setdefault(gi, []).append(r["score_regional"])
    for gi in ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]:
        if gi in by_iim:
            vals = by_iim[gi]
            print(
                f"        {gi:>10}: {sum(vals)/len(vals):.4f} "
                f"(n={len(vals)})"
            )

    print("\nFASE 2 — paso 2/3 (score regional) completado.")


if __name__ == "__main__":
    main()
