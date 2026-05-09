"""
Unifica los cuatro datasets municipales generados en la Fase 1 en una
única tabla cross-section con 124 filas (una por municipio de Jalisco)
y aproximadamente 25 columnas de features listas para el cálculo del
score regional municipal.

Insumos
-------
data/raw/municipios_jalisco_catalogo.csv
    Catálogo unificado del proyecto (clave INEGI + nombre + geometría).
data/raw/banxico_ce166_jalisco_trimestral.csv
    Serie trimestral de remesas por municipio 2013-2026.
data/raw/conapo_iim_jalisco_2020.csv
    Indicadores de intensidad migratoria 2020 por municipio.
data/raw/conapo_marginacion_jalisco_2020.csv
    Indicadores de marginación 2020 por municipio.
data/raw/inegi_iter_jalisco_2020.csv
    Indicadores socioeconómicos del Censo 2020 por municipio.

Salidas
-------
data/processed/jalisco_municipal_features.csv
    Tabla cross-section 124 × ~25 columnas. Variables agrupadas en
    cuatro categorías:
        Identificación: cve_municipio, municipio, area_km2.
        Remesas (CE166): mediana_remesas_2020_2024, cv_remesas_2020_2024,
                         pendiente_remesas_2020_2024, remesas_2024_total,
                         remesas_per_capita_2024.
        Intensidad migratoria (IIM): pct_viv_remesas, pct_viv_emigrantes,
                                     pct_viv_circular, pct_viv_retorno,
                                     iim_dp2, gim_dp2, pos_nacional.
        Marginación (CONAPO): pct_pob_analfabeta_15ymas,
                              pct_pob_sin_basica_15ymas,
                              pct_viv_sin_drenaje, pct_viv_sin_energia,
                              pct_viv_sin_agua, pct_viv_piso_tierra,
                              pct_viv_hacinamiento,
                              pct_pob_localidades_menores_5000,
                              pct_pob_ocupada_hasta_2sm,
                              im_2020, gm_2020, imn_2020.
        Socioeconómicas (Iter): pob_total, viv_part_hab,
                                escolaridad_promedio, tasa_ocupacion,
                                pct_viv_internet, pct_viv_celular,
                                pct_viv_pc, pct_viv_auto.
data/processed/jalisco_municipal_features_metadata.json

Notas metodológicas
-------------------
1. Todas las features temporales del CE166 se computan sobre el periodo
   2020-2024 (5 años, 20 trimestres) por estabilidad post-pandemia y
   alineación con el Censo 2020. La tendencia se calcula como pendiente
   OLS simple del flujo trimestral en MUSD vs trimestre numérico.

2. El coeficiente de variación (CV) usa media simple como denominador.
   Si la media es cero o el municipio tiene menos de 12 observaciones
   en el periodo, se reporta NaN.

3. Remesas per cápita 2024 = (suma anual 2024 en USD millones × 1e6) /
   pob_total. Resultado en USD per cápita.

Uso
---
    python scripts/07_unificar_features_municipales.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"
CE166_CSV = PROJECT_ROOT / "data" / "raw" / "banxico_ce166_jalisco_trimestral.csv"
IIM_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_iim_jalisco_2020.csv"
MARG_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_marginacion_jalisco_2020.csv"
ITER_CSV = PROJECT_ROOT / "data" / "raw" / "inegi_iter_jalisco_2020.csv"

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "jalisco_municipal_features.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jalisco_municipal_features_metadata.json"
)

PERIODO_INICIO = "2020-01-01"
PERIODO_FIN = "2024-12-31"
ANIO_PER_CAPITA = 2024


def parse_float(s: str) -> float | None:
    """Convierte un valor string a float, manejando vacíos."""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(s: str) -> int | None:
    """Convierte un valor string a int, manejando vacíos."""
    v = parse_float(s)
    return int(v) if v is not None else None


def ols_slope(x: list[float], y: list[float]) -> float | None:
    """
    Calcula la pendiente OLS simple (sin librería estadística) sobre dos
    listas paralelas. Devuelve None si hay menos de 3 observaciones o
    si la varianza de x es cero.
    """
    n = len(x)
    if n < 3 or len(y) != n:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    if den == 0:
        return None
    return num / den


def load_catalog() -> dict[str, dict[str, str]]:
    """Carga el catálogo unificado indexado por clave INEGI."""
    catalog: dict[str, dict[str, str]] = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cve = row["cve_municipio"].strip().zfill(5)
            catalog[cve] = {
                "municipio": row["municipio"].strip(),
                "area_km2": parse_float(row["area_km2"]),
                "lat_centro": parse_float(row["lat_centro"]),
                "lon_centro": parse_float(row["lon_centro"]),
            }
    return catalog


def compute_ce166_features(
    cve: str, ce166_by_municipio: dict[str, list[dict[str, object]]]
) -> dict[str, float | None]:
    """
    Calcula features temporales del CE166 para un municipio, restringido
    al periodo 2020-2024 (20 trimestres esperados).
    """
    series = ce166_by_municipio.get(cve, [])
    # Filtrar al periodo de interés.
    series_periodo = [
        r
        for r in series
        if PERIODO_INICIO <= str(r["fecha"]) <= PERIODO_FIN
        and r["remesas_musd"] is not None
    ]
    valores = [float(r["remesas_musd"]) for r in series_periodo]

    features: dict[str, float | None] = {
        "mediana_remesas_2020_2024": None,
        "cv_remesas_2020_2024": None,
        "pendiente_remesas_2020_2024": None,
        "remesas_2024_total": None,
        "remesas_per_capita_2024": None,  # se completa después con pob_total
    }

    if len(valores) >= 12:
        features["mediana_remesas_2020_2024"] = round(
            statistics.median(valores), 6
        )
        media = sum(valores) / len(valores)
        if media > 0:
            sd = statistics.pstdev(valores)
            features["cv_remesas_2020_2024"] = round(sd / media, 6)
        # Pendiente OLS: x = índice trimestral 0..n-1, y = valor.
        x = list(range(len(valores)))
        slope = ols_slope([float(i) for i in x], valores)
        if slope is not None:
            features["pendiente_remesas_2020_2024"] = round(slope, 6)

    # Total 2024 y per cápita.
    valores_2024 = [
        float(r["remesas_musd"])
        for r in series
        if str(r["fecha"]).startswith("2024")
        and r["remesas_musd"] is not None
    ]
    if valores_2024:
        total_2024 = sum(valores_2024)
        features["remesas_2024_total"] = round(total_2024, 4)

    return features


def main() -> None:
    """Carga, une, computa features y persiste el output."""
    # Validar que existan todos los insumos.
    for path in [CATALOG_CSV, CE166_CSV, IIM_CSV, MARG_CSV, ITER_CSV]:
        if not path.exists():
            print(f"ERROR: no se encontró {path}", file=sys.stderr)
            sys.exit(1)

    print("[1/6] Cargando catálogo unificado...")
    catalog = load_catalog()
    print(f"      Municipios en catálogo: {len(catalog)}")

    print("[2/6] Cargando serie trimestral CE166...")
    ce166_by_mun: dict[str, list[dict[str, object]]] = {}
    with open(CE166_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cve = row["cve_municipio"].strip().zfill(5)
            ce166_by_mun.setdefault(cve, []).append(
                {
                    "fecha": row["fecha"].strip(),
                    "remesas_musd": parse_float(row["remesas_musd"]),
                }
            )
    print(f"      Municipios con serie CE166: {len(ce166_by_mun)}")

    print("[3/6] Cargando datasets municipales...")
    iim_by_mun: dict[str, dict[str, object]] = {}
    with open(IIM_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            iim_by_mun[cve] = {
                "pct_viv_remesas": parse_float(row.get("pct_viv_remesas")),
                "pct_viv_emigrantes": parse_float(
                    row.get("pct_viv_emigrantes")
                ),
                "pct_viv_circular": parse_float(row.get("pct_viv_circular")),
                "pct_viv_retorno": parse_float(row.get("pct_viv_retorno")),
                "iim_dp2": parse_float(row.get("iim_dp2")),
                "gim_dp2": (row.get("gim_dp2") or "").strip(),
                "pos_nacional": parse_int(row.get("pos_nacional")),
            }
    print(f"      Municipios IIM: {len(iim_by_mun)}")

    marg_by_mun: dict[str, dict[str, object]] = {}
    with open(MARG_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            marg_by_mun[cve] = {
                "pct_pob_analfabeta_15ymas": parse_float(
                    row.get("pct_pob_analfabeta_15ymas")
                ),
                "pct_pob_sin_basica_15ymas": parse_float(
                    row.get("pct_pob_sin_basica_15ymas")
                ),
                "pct_viv_sin_drenaje": parse_float(
                    row.get("pct_viv_sin_drenaje")
                ),
                "pct_viv_sin_energia": parse_float(
                    row.get("pct_viv_sin_energia")
                ),
                "pct_viv_sin_agua": parse_float(row.get("pct_viv_sin_agua")),
                "pct_viv_piso_tierra": parse_float(
                    row.get("pct_viv_piso_tierra")
                ),
                "pct_viv_hacinamiento": parse_float(
                    row.get("pct_viv_hacinamiento")
                ),
                "pct_pob_localidades_menores_5000": parse_float(
                    row.get("pct_pob_localidades_menores_5000")
                ),
                "pct_pob_ocupada_hasta_2sm": parse_float(
                    row.get("pct_pob_ocupada_hasta_2sm")
                ),
                "im_2020": parse_float(row.get("im_2020")),
                "gm_2020": (row.get("gm_2020") or "").strip(),
                "imn_2020": parse_float(row.get("imn_2020")),
            }
    print(f"      Municipios Marginación: {len(marg_by_mun)}")

    iter_by_mun: dict[str, dict[str, object]] = {}
    with open(ITER_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            iter_by_mun[cve] = {
                "pob_total": parse_int(row.get("pob_total")),
                "viv_part_hab": parse_int(row.get("viv_part_hab")),
                "escolaridad_promedio": parse_float(
                    row.get("escolaridad_promedio")
                ),
                "tasa_ocupacion": parse_float(row.get("tasa_ocupacion")),
                "pct_viv_internet": parse_float(row.get("pct_viv_internet")),
                "pct_viv_celular": parse_float(row.get("pct_viv_celular")),
                "pct_viv_pc": parse_float(row.get("pct_viv_pc")),
                "pct_viv_auto": parse_float(row.get("pct_viv_auto")),
            }
    print(f"      Municipios Iter: {len(iter_by_mun)}")

    # --- Construir tabla unificada ---------------------------------------
    print("[4/6] Construyendo tabla cross-section...")
    output_rows: list[dict[str, object]] = []
    advertencias: list[str] = []

    for cve in sorted(catalog.keys()):
        cat = catalog[cve]
        rec: dict[str, object] = {
            "cve_municipio": cve,
            "municipio": cat["municipio"],
            "area_km2": cat["area_km2"],
            "lat_centro": cat["lat_centro"],
            "lon_centro": cat["lon_centro"],
        }

        # Verificar disponibilidad en cada fuente.
        if cve not in ce166_by_mun:
            advertencias.append(f"{cve} sin datos CE166")
        if cve not in iim_by_mun:
            advertencias.append(f"{cve} sin datos IIM")
        if cve not in marg_by_mun:
            advertencias.append(f"{cve} sin datos Marginación")
        if cve not in iter_by_mun:
            advertencias.append(f"{cve} sin datos Iter")

        # Features CE166.
        rec.update(compute_ce166_features(cve, ce166_by_mun))

        # Features IIM.
        rec.update(iim_by_mun.get(cve, {}))

        # Features Marginación.
        rec.update(marg_by_mun.get(cve, {}))

        # Features Iter.
        rec.update(iter_by_mun.get(cve, {}))

        # Calcular remesas per cápita 2024 ahora que tenemos pob_total.
        total_2024 = rec.get("remesas_2024_total")
        pob_total = rec.get("pob_total")
        if total_2024 and pob_total and pob_total > 0:
            # remesas_2024_total está en MUSD; convertir a USD y dividir.
            rec["remesas_per_capita_2024"] = round(
                (total_2024 * 1_000_000) / pob_total, 4
            )

        output_rows.append(rec)

    if advertencias:
        print(
            f"\nADVERTENCIAS ({len(advertencias)}):", file=sys.stderr
        )
        for a in advertencias[:20]:
            print(f"      - {a}", file=sys.stderr)

    # --- Persistencia ----------------------------------------------------
    print("[5/6] Guardando salida...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # Identificación.
        "cve_municipio", "municipio", "area_km2", "lat_centro", "lon_centro",
        # Remesas (CE166).
        "mediana_remesas_2020_2024", "cv_remesas_2020_2024",
        "pendiente_remesas_2020_2024", "remesas_2024_total",
        "remesas_per_capita_2024",
        # Intensidad migratoria (IIM).
        "pct_viv_remesas", "pct_viv_emigrantes", "pct_viv_circular",
        "pct_viv_retorno", "iim_dp2", "gim_dp2", "pos_nacional",
        # Marginación.
        "pct_pob_analfabeta_15ymas", "pct_pob_sin_basica_15ymas",
        "pct_viv_sin_drenaje", "pct_viv_sin_energia", "pct_viv_sin_agua",
        "pct_viv_piso_tierra", "pct_viv_hacinamiento",
        "pct_pob_localidades_menores_5000", "pct_pob_ocupada_hasta_2sm",
        "im_2020", "gm_2020", "imn_2020",
        # Socioeconómicas (Iter).
        "pob_total", "viv_part_hab", "escolaridad_promedio",
        "tasa_ocupacion", "pct_viv_internet", "pct_viv_celular",
        "pct_viv_pc", "pct_viv_auto",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in output_rows:
            row_out = {
                k: ("" if rec.get(k) is None else rec.get(k))
                for k in fieldnames
            }
            writer.writerow(row_out)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    # --- Metadatos -------------------------------------------------------
    metadata = {
        "descripcion": (
            "Tabla cross-section unificada de features municipales para "
            "el cálculo del score regional municipal del Componente 1 del "
            "modelo de crédito remesa."
        ),
        "n_municipios": len(output_rows),
        "n_features": len(fieldnames) - 5,  # excluir identificación
        "periodo_temporal_features_remesas": (
            f"{PERIODO_INICIO} a {PERIODO_FIN}"
        ),
        "fuentes": {
            "catalogo": str(CATALOG_CSV.name),
            "remesas_trimestrales": str(CE166_CSV.name),
            "intensidad_migratoria": str(IIM_CSV.name),
            "marginacion": str(MARG_CSV.name),
            "indicadores_iter": str(ITER_CSV.name),
        },
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "notas": [
            "Las features temporales del CE166 (mediana, CV, pendiente) "
            "se computan sobre 20 trimestres del periodo 2020-2024 por "
            "estabilidad post-pandemia y alineación con el Censo 2020.",
            "remesas_per_capita_2024 está en USD per cápita (= "
            "remesas_2024_total × 1e6 / pob_total).",
            "Si un municipio tiene menos de 12 observaciones trimestrales "
            "en el periodo, las features temporales son NaN.",
        ],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación ----------------------------------------------------
    print("\nVerificación cruzada:")
    print(f"      Filas: {len(output_rows)} (esperado: 124)")

    # Top 5 por remesas per cápita.
    rpc = [
        (r["municipio"], r["remesas_per_capita_2024"], r["pob_total"])
        for r in output_rows
        if r.get("remesas_per_capita_2024") is not None
    ]
    rpc.sort(key=lambda x: x[1], reverse=True)
    print(f"\n      Top 5 municipios por remesas per cápita 2024:")
    for nombre, rpcv, pob in rpc[:5]:
        print(
            f"        {nombre:<30} USD {rpcv:>8,.2f}/hab "
            f"(pob: {pob:>8,})"
        )

    # Bottom 5.
    print(f"\n      Bottom 5 municipios por remesas per cápita 2024:")
    for nombre, rpcv, pob in rpc[-5:]:
        print(
            f"        {nombre:<30} USD {rpcv:>8,.2f}/hab "
            f"(pob: {pob:>8,})"
        )

    # Estadísticos generales.
    rpc_vals = [v for _, v, _ in rpc]
    print(
        f"\n      Estadísticos remesas per cápita 2024:"
        f"\n        min   = USD {min(rpc_vals):,.2f}"
        f"\n        max   = USD {max(rpc_vals):,.2f}"
        f"\n        media = USD {sum(rpc_vals)/len(rpc_vals):,.2f}"
        f"\n        mediana = USD {statistics.median(rpc_vals):,.2f}"
    )

    print("\nFASE 2 — paso 1/3 (unificación) completado.")


if __name__ == "__main__":
    main()
