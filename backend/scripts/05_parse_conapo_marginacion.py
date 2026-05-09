"""
Parser del Índice de Marginación 2020 a nivel municipal publicado por la
Secretaría General del Consejo Nacional de Población (SG-CONAPO).

A diferencia del IIM (Índice de Intensidad Migratoria), que mide la
relación con el flujo migratorio México-Estados Unidos, el Índice de
Marginación mide carencias estructurales en cuatro dimensiones:
educación, vivienda, ingresos y distribución de la población. Su
metodología combina nueve indicadores socioeconómicos en un índice
compuesto que estratifica los municipios en cinco grados.

Insumos
-------
data/raw/conapo_marginacion_municipal_2020_raw.csv
    Archivo "imm_2020-3.csv" descargado desde
    https://www.datos.gob.mx/dataset/indice-de-marginacion-2020
    Codificación: latin-1 (alineada con el archivo IIM de CONAPO).
data/raw/municipios_jalisco_catalogo.csv
    Catálogo unificado del proyecto.

Salidas
-------
data/raw/conapo_marginacion_jalisco_2020.csv
    Tabla municipio × indicadores con 124 filas.
data/raw/conapo_marginacion_metadata.json

Uso
---
    python scripts/05_parse_conapo_marginacion.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_marginacion_municipal_2020_raw.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_marginacion_jalisco_2020.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "conapo_marginacion_metadata.json"

JALISCO_CVE_ENT = "14"

# Mapeo de columnas: nombre original CONAPO → nombre semántico de salida.
COLUMN_RENAME: dict[str, str] = {
    "POB_TOT": "pob_total",
    "ANALF": "pct_pob_analfabeta_15ymas",
    "SBASC": "pct_pob_sin_basica_15ymas",
    "OVSDE": "pct_viv_sin_drenaje",
    "OVSEE": "pct_viv_sin_energia",
    "OVSAE": "pct_viv_sin_agua",
    "OVPT": "pct_viv_piso_tierra",
    "VHAC": "pct_viv_hacinamiento",
    "PL.5000": "pct_pob_localidades_menores_5000",
    "PO2SM": "pct_pob_ocupada_hasta_2sm",
    "IM_2020": "im_2020",
    "GM_2020": "gm_2020",
    "IMN_2020": "imn_2020",
}


def fix_mojibake(text: str) -> str:
    """
    Repara el doble encoding de los archivos CONAPO (UTF-8 escrito como
    latin-1).
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def parse_float(raw: str) -> float | None:
    """Convierte un valor crudo a float, tratando vacío como missing."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() in {"N/E", "NA", "NAN", "ND"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(raw: str) -> int | None:
    """Convierte un valor crudo a int, tratando vacío como missing."""
    v = parse_float(raw)
    return int(v) if v is not None else None


def load_catalog() -> dict[str, str]:
    """Carga el catálogo unificado indexado por clave INEGI."""
    catalog: dict[str, str] = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cve = row["cve_municipio"].strip().zfill(5)
            catalog[cve] = row["municipio"].strip()
    return catalog


def main() -> None:
    """Ejecuta el parseo, validaciones y persistencia."""
    if not RAW_CSV.exists():
        print(f"ERROR: no se encontró {RAW_CSV}", file=sys.stderr)
        sys.exit(1)
    if not CATALOG_CSV.exists():
        print(
            f"ERROR: no se encontró {CATALOG_CSV}. "
            f"Corre primero scripts/00_build_municipios_catalog.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[1/4] Cargando catálogo unificado...")
    catalog = load_catalog()
    print(f"      Municipios en catálogo: {len(catalog)}")

    print(f"[2/4] Leyendo CSV crudo de CONAPO ({RAW_CSV.name})...")
    with open(RAW_CSV, "r", encoding="latin-1", newline="") as fh:
        reader = csv.DictReader(fh)
        jalisco_rows = [
            row for row in reader if row.get("CVE_ENT", "").strip() == JALISCO_CVE_ENT
        ]
    print(f"      Filas con CVE_ENT={JALISCO_CVE_ENT}: {len(jalisco_rows)}")

    # --- Cruce con catálogo ----------------------------------------------
    print("[3/4] Cruzando contra catálogo del proyecto...")
    output_rows: list[dict[str, object]] = []
    cve_in_conapo: set[str] = set()
    excluded_log: list[str] = []

    for row in jalisco_rows:
        cve = row.get("CVE_MUN", "").strip().zfill(5)
        cve_in_conapo.add(cve)

        if cve not in catalog:
            nom_raw = row.get("NOM_MUN", "?")
            excluded_log.append(f"cve {cve} ({fix_mojibake(nom_raw)})")
            continue

        rec: dict[str, object] = {
            "cve_municipio": cve,
            "municipio": catalog[cve],
        }
        # Columnas numéricas continuas.
        for raw_col, out_col in COLUMN_RENAME.items():
            if raw_col == "POB_TOT":
                rec[out_col] = parse_int(row.get(raw_col, ""))
            elif raw_col == "GM_2020":
                # Categórico: aplicar fix_mojibake por si trae acentos.
                gm_raw = (row.get(raw_col) or "").strip()
                rec[out_col] = fix_mojibake(gm_raw)
            else:
                rec[out_col] = parse_float(row.get(raw_col, ""))
        output_rows.append(rec)

    missing_in_conapo = set(catalog.keys()) - cve_in_conapo
    print(f"      Filas conservadas: {len(output_rows)}")
    print(f"      Filas excluidas (no en catálogo): {len(excluded_log)}")
    for item in excluded_log:
        print(f"        - {item}")
    if missing_in_conapo:
        print(
            f"\nADVERTENCIA: {len(missing_in_conapo)} municipios del catálogo "
            f"no aparecen en CONAPO Marginación:",
            file=sys.stderr,
        )
        for cve in sorted(missing_in_conapo):
            print(f"        - cve {cve} ({catalog[cve]})", file=sys.stderr)

    if len(output_rows) != len(catalog):
        print(
            f"\nERROR: filas de salida ({len(output_rows)}) ≠ catálogo "
            f"({len(catalog)})",
            file=sys.stderr,
        )
        sys.exit(1)

    output_rows.sort(key=lambda r: r["cve_municipio"])

    # --- Persistencia ----------------------------------------------------
    print("[4/4] Guardando salidas...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cve_municipio", "municipio"] + list(COLUMN_RENAME.values())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in output_rows:
            rec_out = {k: ("" if v is None else v) for k, v in rec.items()}
            writer.writerow(rec_out)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    metadata = {
        "dataset": "Índice de Marginación Municipal 2020",
        "institucion": "Secretaría General del Consejo Nacional de Población (SG-CONAPO)",
        "n_municipios_jalisco_publicados_conapo": len(jalisco_rows),
        "n_municipios_incluidos_en_dataset": len(output_rows),
        "n_municipios_excluidos": len(excluded_log),
        "municipios_excluidos": excluded_log,
        "metodologia": (
            "Índice compuesto que combina nueve indicadores socioeconómicos "
            "en cuatro dimensiones: educación (% población analfabeta de 15 "
            "años o más, % población sin primaria completa de 15 años o más), "
            "vivienda (% viviendas sin drenaje ni excusado, sin energía "
            "eléctrica, sin agua entubada, con piso de tierra, con algún "
            "nivel de hacinamiento), distribución poblacional (% población en "
            "localidades menores a 5,000 habitantes) e ingresos (% población "
            "ocupada con ingreso de hasta 2 salarios mínimos). El índice "
            "continuo (im_2020) se estratifica en cinco grados (gm_2020): "
            "Muy bajo, Bajo, Medio, Alto, Muy alto. La versión normalizada "
            "(imn_2020) reescala el índice al rango [0, 1]."
        ),
        "columnas_origen_a_destino": COLUMN_RENAME,
        "unidades": {
            "pob_total": "Habitantes",
            "pct_*": "Porcentaje (0-100)",
            "im_2020": "Índice continuo (mayor = más marginación)",
            "gm_2020": "Categoría ordinal",
            "imn_2020": "Índice normalizado en rango [0, 1]",
        },
        "source_url": "https://www.datos.gob.mx/dataset/indice-de-marginacion-2020",
        "raw_file": RAW_CSV.name,
        "raw_file_origen": "imm_2020-3.csv",
        "raw_file_encoding": "latin-1",
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "apa7_reference": (
            "Consejo Nacional de Población. (2021). Índices de marginación "
            "2020 [Base de datos a nivel municipal]. Secretaría General del "
            "Consejo Nacional de Población. https://www.gob.mx/conapo/"
            "documentos/indices-de-marginacion-2020-284372"
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación cruzada --------------------------------------------
    print("\nVerificación cruzada — distribución por grado de marginación:")
    gm_counts: dict[str, int] = {}
    for rec in output_rows:
        gm = rec["gm_2020"] or "(sin dato)"
        gm_counts[gm] = gm_counts.get(gm, 0) + 1
    for gm in ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto", "(sin dato)"]:
        if gm in gm_counts:
            print(f"      {gm:>10}: {gm_counts[gm]:>3} municipios")

    im_values = [r["im_2020"] for r in output_rows if r["im_2020"] is not None]
    if im_values:
        print(
            f"\nResumen del índice de marginación continuo (im_2020):"
            f"\n      min   = {min(im_values):.4f}"
            f"\n      max   = {max(im_values):.4f}"
            f"\n      media = {sum(im_values)/len(im_values):.4f}"
        )

    print("\nFASE 1 — paso 4b/5 completado.")


if __name__ == "__main__":
    main()
