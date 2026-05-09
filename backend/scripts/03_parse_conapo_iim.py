"""
Parser del Índice de Intensidad Migratoria México-Estados Unidos 2020,
nivel municipal, publicado por la Secretaría General del Consejo Nacional
de Población (SG-CONAPO).

Versión refactorizada (post-construcción del catálogo unificado): cruza
por clave INEGI directamente contra el catálogo del proyecto.

Insumos
-------
data/raw/conapo_iim_municipal_2020_raw.csv
    Archivo "06_iim_mex_eeuu_2020_municipio.csv" descargado desde
    https://www.datos.gob.mx/dataset/indice_intensidad_migratoria
    Codificación: latin-1.
data/raw/municipios_jalisco_catalogo.csv
    Catálogo unificado del proyecto (generado por 00_build_municipios_catalog.py).

Salidas
-------
data/raw/conapo_iim_jalisco_2020.csv
    Tabla municipio × indicadores con 124 filas.
data/raw/conapo_iim_metadata.json

Uso
---
    python scripts/03_parse_conapo_iim.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_iim_municipal_2020_raw.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_iim_jalisco_2020.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "conapo_iim_metadata.json"

JALISCO_CVE_ENT = "14"


def fix_mojibake(text: str) -> str:
    """
    Repara mojibake del archivo CONAPO (doble encoding UTF-8/latin-1).
    Si la cadena no necesita reparación, la devuelve sin cambios.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def parse_float(raw: str) -> float | None:
    """Convierte un valor crudo a float, devolviendo None si está vacío."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() in {"N/E", "NA", "NAN"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_int(raw: str) -> int | None:
    """Convierte un valor crudo a int, devolviendo None si está vacío."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_catalog() -> dict[str, str]:
    """
    Carga el catálogo unificado indexado por clave INEGI.

    Returns
    -------
    dict
        {cve_municipio (5 dígitos): nombre oficial del municipio}
    """
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
            row for row in reader if row.get("cve_ent", "").strip() == JALISCO_CVE_ENT
        ]
    print(f"      Filas con cve_ent={JALISCO_CVE_ENT}: {len(jalisco_rows)}")

    print("[3/4] Cruzando contra catálogo del proyecto...")
    output_rows: list[dict[str, object]] = []
    cve_in_conapo: set[str] = set()
    excluded_log: list[str] = []

    for row in jalisco_rows:
        cve = row.get("cve_mun", "").strip().zfill(5)
        cve_in_conapo.add(cve)

        if cve not in catalog:
            nom_raw = row.get("nom_mun", "?")
            excluded_log.append(f"cve {cve} ({fix_mojibake(nom_raw)})")
            continue

        output_rows.append(
            {
                "cve_municipio": cve,
                "municipio": catalog[cve],
                "viv_tot": parse_int(row.get("viv_tot", "")),
                "pct_viv_remesas": parse_float(row.get("viv_rem", "")),
                "pct_viv_emigrantes": parse_float(row.get("viv_emig", "")),
                "pct_viv_circular": parse_float(row.get("viv_circ", "")),
                "pct_viv_retorno": parse_float(row.get("viv_ret", "")),
                "iim_dp2": parse_float(row.get("iim_dp2", "")),
                "gim_dp2": (row.get("gim_dp2") or "").strip(),
                "pos_nacional": parse_int(row.get("pos_nal", "")),
            }
        )

    missing_in_conapo = set(catalog.keys()) - cve_in_conapo
    print(f"      Filas conservadas: {len(output_rows)}")
    print(f"      Filas excluidas (no en catálogo): {len(excluded_log)}")
    for item in excluded_log:
        print(f"        - {item}")
    if missing_in_conapo:
        print(
            f"\nADVERTENCIA: {len(missing_in_conapo)} municipios del catálogo "
            f"no aparecen en CONAPO:",
            file=sys.stderr,
        )
        for cve in sorted(missing_in_conapo):
            print(f"        - cve {cve} ({catalog[cve]})", file=sys.stderr)

    if len(output_rows) != len(catalog):
        print(
            f"ADVERTENCIA: filas de salida ({len(output_rows)}) ≠ catálogo "
            f"({len(catalog)})",
            file=sys.stderr,
        )

    output_rows.sort(key=lambda r: r["cve_municipio"])

    print(f"[4/4] Guardando salidas...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cve_municipio",
        "municipio",
        "viv_tot",
        "pct_viv_remesas",
        "pct_viv_emigrantes",
        "pct_viv_circular",
        "pct_viv_retorno",
        "iim_dp2",
        "gim_dp2",
        "pos_nacional",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in output_rows:
            rec_out = {k: ("" if v is None else v) for k, v in rec.items()}
            writer.writerow(rec_out)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    metadata = {
        "dataset": "Índice de Intensidad Migratoria México-Estados Unidos 2020 — Municipal",
        "institucion": "Secretaría General del Consejo Nacional de Población (SG-CONAPO)",
        "n_municipios_jalisco_publicados_conapo": len(jalisco_rows),
        "n_municipios_incluidos_en_dataset": len(output_rows),
        "n_municipios_excluidos": len(excluded_log),
        "municipios_excluidos": excluded_log,
        "metodologia": (
            "Índice DP2 (distancia multivariante de Pena-Trapero). Sintetiza "
            "cuatro indicadores: % viviendas receptoras de remesas, % viviendas "
            "con emigrantes a EE.UU., % viviendas con migrantes circulares y "
            "% viviendas con migrantes de retorno. El índice continuo (iim_dp2) "
            "se estratifica en cinco grados (gim_dp2): Muy bajo, Bajo, Medio, "
            "Alto, Muy alto."
        ),
        "unidades": {
            "viv_tot": "Viviendas (conteo absoluto)",
            "pct_viv_remesas": "Porcentaje (0-100)",
            "pct_viv_emigrantes": "Porcentaje (0-100)",
            "pct_viv_circular": "Porcentaje (0-100)",
            "pct_viv_retorno": "Porcentaje (0-100)",
            "iim_dp2": "Índice continuo (mayor = más intensidad migratoria)",
            "gim_dp2": "Categoría ordinal",
            "pos_nacional": "Posición en el ranking nacional (1 = mayor intensidad)",
        },
        "source_url": "https://www.datos.gob.mx/dataset/indice_intensidad_migratoria",
        "raw_file": RAW_CSV.name,
        "raw_file_origen": "06_iim_mex_eeuu_2020_municipio.csv",
        "encoding_origen": "latin-1",
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "apa7_reference": (
            "Consejo Nacional de Población. (2021). Índices de intensidad "
            "migratoria México-Estados Unidos 2020 [Base de datos a nivel "
            "municipal]. Secretaría General del Consejo Nacional de Población. "
            "https://www.gob.mx/conapo/documentos/indice-de-intensidad-"
            "migratoria-mexico-estados-unidos-2020"
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    print("\nVerificación cruzada — distribución por grado de intensidad migratoria:")
    gim_counts: dict[str, int] = {}
    for rec in output_rows:
        gim = rec["gim_dp2"] or "(sin dato)"
        gim_counts[gim] = gim_counts.get(gim, 0) + 1
    for gim in ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto", "(sin dato)"]:
        if gim in gim_counts:
            print(f"      {gim:>10}: {gim_counts[gim]:>3} municipios")

    iim_values = [r["iim_dp2"] for r in output_rows if r["iim_dp2"] is not None]
    if iim_values:
        print(
            f"\nResumen del índice IIM continuo (iim_dp2):"
            f"\n      min   = {min(iim_values):.4f}"
            f"\n      max   = {max(iim_values):.4f}"
            f"\n      media = {sum(iim_values)/len(iim_values):.4f}"
        )

    print("\nFASE 1 — paso 3/5 completado.")


if __name__ == "__main__":
    main()
