"""
Parser del Iter 2020 (Indicadores por entidad y municipio) del Censo de
Población y Vivienda 2020 publicado por el INEGI.

Lee el archivo crudo `ITER_14CSV20.csv` correspondiente a Jalisco, filtra
los totales municipales (filas con MUN distinto de 0 y LOC igual a 0) y
extrae un subconjunto de indicadores socioeconómicos relevantes para el
Componente 1 (score regional municipal) del modelo. Los indicadores de
viviendas se convierten de conteos absolutos a porcentajes dividiendo
entre `TVIVPARHAB` (total de viviendas particulares habitadas).

Insumos
-------
data/raw/inegi_iter_jalisco_2020_raw.csv
    Archivo "ITER_14CSV20.csv" extraído del ZIP descargado en
    https://www.inegi.org.mx/programas/ccpv/2020/#descargas
    Codificación: UTF-8 con BOM. Importante leerlo como 'utf-8-sig' para
    descartar el BOM que el INEGI antepone al primer encabezado.
data/raw/municipios_jalisco_catalogo.csv
    Catálogo unificado del proyecto.

Salidas
-------
data/raw/inegi_iter_jalisco_2020.csv
    Tabla municipio × indicadores con 124 filas.
data/raw/inegi_iter_metadata.json

Notas metodológicas
-------------------
1. El Iter publica datos a tres niveles de agregación geográfica:
   - Total estatal: ENTIDAD = 14, MUN = 0, LOC = 0
   - Total municipal: ENTIDAD = 14, MUN ≠ 0, LOC = 0
   - Localidad individual: ENTIDAD = 14, MUN ≠ 0, LOC ≠ 0
   Filtramos sólo el segundo nivel.

2. La clave INEGI a nivel municipio se construye concatenando
   ENTIDAD (2 dígitos) + MUN (3 dígitos), p.ej. 14001 = Jalisco/Acatic.

3. Los valores marcados con asterisco (*) por el INEGI representan
   datos confidenciales por privacidad estadística (muy pocos casos en
   la celda). Se tratan como missing (None).

4. Los porcentajes calculados (`pct_viv_*`) usan `TVIVPARHAB` como
   denominador, no `VIVTOT`. Esto es la convención estándar en análisis
   socioeconómico mexicano: el porcentaje de viviendas con cierto
   atributo se computa sobre las viviendas habitadas, no sobre el total
   incluyendo desocupadas.

Uso
---
    python scripts/04_parse_inegi_iter.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "inegi_iter_jalisco_2020_raw.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "inegi_iter_jalisco_2020.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "inegi_iter_metadata.json"

JALISCO_CVE_ENT = "14"

# Variables a extraer directamente como conteos absolutos.
ABSOLUTE_VARS: dict[str, str] = {
    "POBTOT": "pob_total",
    "TVIVPARHAB": "viv_part_hab",
    "PEA": "pea",
    "POCUPADA": "pob_ocupada",
}

# Variables que se persisten tal cual (escalares no-conteo).
SCALAR_VARS: dict[str, str] = {
    "GRAPROES": "escolaridad_promedio",
}

# Variables de viviendas que se convierten a porcentaje sobre TVIVPARHAB.
PCT_OVER_TVIVPARHAB: dict[str, str] = {
    "VPH_INTER": "pct_viv_internet",
    "VPH_CEL": "pct_viv_celular",
    "VPH_PC": "pct_viv_pc",
    "VPH_AUTOM": "pct_viv_auto",
}


def parse_inegi_value(raw: str) -> float | None:
    """
    Convierte un valor crudo del Iter a float.

    Trata como missing: cadena vacía, '*' (confidencialidad estadística),
    'N/D' y 'NA'.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s in {"*", "N/D", "NA", "ND"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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

    print(f"[2/4] Leyendo CSV crudo del Iter ({RAW_CSV.name})...")
    # El archivo Iter del INEGI viene en UTF-8 con BOM. Leerlo como
    # latin-1 produce mojibake en nombres con tildes; leerlo como UTF-8
    # puro deja el BOM como parte del primer encabezado de columna.
    # 'utf-8-sig' resuelve ambos problemas.
    with open(RAW_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"      Filas totales en el archivo: {len(rows)}")

    # --- Filtrar a totales municipales -----------------------------------
    # Los códigos del Iter vienen con relleno de ceros: MUN es de 3 dígitos
    # ('000', '001', ..., '125') y LOC es de 4 dígitos ('0000', '0001', ...).
    # Las filas de "Total del Municipio" tienen LOC = '0000'. La fila del
    # total estatal tiene MUN = '000' y LOC = '0000'. Filtramos ambas
    # condiciones usando comparación entera para tolerar variaciones.
    print("[3/4] Filtrando totales municipales (MUN ≠ 0, LOC = 0)...")
    municipal_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("ENTIDAD", "").strip() != JALISCO_CVE_ENT:
            continue
        try:
            mun_int = int(row.get("MUN", "").strip())
            loc_int = int(row.get("LOC", "").strip())
        except ValueError:
            continue
        if mun_int == 0:
            continue
        if loc_int != 0:
            continue
        municipal_rows.append(row)
    print(f"      Totales municipales encontrados: {len(municipal_rows)}")

    # --- Cruce con catálogo y construcción del output --------------------
    output_rows: list[dict[str, object]] = []
    cve_in_iter: set[str] = set()
    excluded_log: list[str] = []

    for row in municipal_rows:
        ent = str(int(row.get("ENTIDAD", "").strip())).zfill(2)
        mun = str(int(row.get("MUN", "").strip())).zfill(3)
        cve = ent + mun
        cve_in_iter.add(cve)

        if cve not in catalog:
            excluded_log.append(f"cve {cve} ({row.get('NOM_MUN', '?')})")
            continue

        # Conteos absolutos.
        rec: dict[str, object] = {
            "cve_municipio": cve,
            "municipio": catalog[cve],
        }
        for raw_col, out_col in ABSOLUTE_VARS.items():
            v = parse_inegi_value(row.get(raw_col, ""))
            rec[out_col] = int(v) if v is not None else None

        # Escalares (escolaridad).
        for raw_col, out_col in SCALAR_VARS.items():
            rec[out_col] = parse_inegi_value(row.get(raw_col, ""))

        # Porcentajes sobre TVIVPARHAB.
        denom = parse_inegi_value(row.get("TVIVPARHAB", ""))
        for raw_col, out_col in PCT_OVER_TVIVPARHAB.items():
            num = parse_inegi_value(row.get(raw_col, ""))
            if num is None or denom is None or denom == 0:
                rec[out_col] = None
            else:
                rec[out_col] = round(num / denom * 100, 4)

        # Tasa de ocupación: pob ocupada / PEA.
        pea = rec.get("pea")
        ocup = rec.get("pob_ocupada")
        if pea and pea > 0 and ocup is not None:
            rec["tasa_ocupacion"] = round(ocup / pea * 100, 4)
        else:
            rec["tasa_ocupacion"] = None

        output_rows.append(rec)

    missing_in_iter = set(catalog.keys()) - cve_in_iter
    print(f"      Filas conservadas: {len(output_rows)}")
    print(f"      Filas excluidas (no en catálogo): {len(excluded_log)}")
    for item in excluded_log:
        print(f"        - {item}")
    if missing_in_iter:
        print(
            f"\nADVERTENCIA: {len(missing_in_iter)} municipios del catálogo "
            f"no aparecen en el Iter:",
            file=sys.stderr,
        )
        for cve in sorted(missing_in_iter):
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
    fieldnames = [
        "cve_municipio",
        "municipio",
        "pob_total",
        "viv_part_hab",
        "escolaridad_promedio",
        "pea",
        "pob_ocupada",
        "tasa_ocupacion",
        "pct_viv_internet",
        "pct_viv_celular",
        "pct_viv_pc",
        "pct_viv_auto",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in output_rows:
            rec_out = {k: ("" if v is None else v) for k, v in rec.items()}
            writer.writerow(rec_out)
    print(f"      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    metadata = {
        "dataset": "Iter 2020 — Censo de Población y Vivienda 2020",
        "institucion": "Instituto Nacional de Estadística y Geografía (INEGI)",
        "n_municipios_publicados_jalisco": len(municipal_rows),
        "n_municipios_incluidos_en_dataset": len(output_rows),
        "n_municipios_excluidos": len(excluded_log),
        "municipios_excluidos": excluded_log,
        "indicadores_extraidos": {
            "absolutos": ABSOLUTE_VARS,
            "escalares": SCALAR_VARS,
            "porcentajes_sobre_TVIVPARHAB": PCT_OVER_TVIVPARHAB,
            "derivados": {
                "tasa_ocupacion": "POCUPADA / PEA × 100",
            },
        },
        "tratamiento_missing": (
            "Los valores '*' del Iter representan dato confidencial por "
            "privacidad estadística y se tratan como missing."
        ),
        "source_url": "https://www.inegi.org.mx/programas/ccpv/2020/#descargas",
        "raw_file": RAW_CSV.name,
        "raw_file_origen": "ITER_14CSV20.csv",
        "raw_file_encoding": "utf-8 con BOM (utf-8-sig)",
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "apa7_reference": (
            "Instituto Nacional de Estadística y Geografía. (2021). Censo de "
            "Población y Vivienda 2020: Indicadores por entidad y municipio "
            "(Iter), Jalisco. INEGI. "
            "https://www.inegi.org.mx/programas/ccpv/2020/"
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación ----------------------------------------------------
    print("\nVerificación cruzada:")
    pob_total = sum(r["pob_total"] for r in output_rows if r["pob_total"])
    print(f"      Población total Jalisco (suma 124 municipios): {pob_total:,}")
    print(f"      Esperado oficial INEGI 2020: ~8,348,151")

    # Top 5 municipios por población.
    top5 = sorted(output_rows, key=lambda r: r["pob_total"] or 0, reverse=True)[:5]
    print(f"\n      Top 5 municipios por población:")
    for r in top5:
        print(f"        {r['cve_municipio']} {r['municipio']:<25}  {r['pob_total']:>10,}")

    print(f"\n      Estadísticas de pct_viv_internet:")
    pcts = [r["pct_viv_internet"] for r in output_rows if r["pct_viv_internet"] is not None]
    if pcts:
        print(f"        min   = {min(pcts):.2f}%")
        print(f"        max   = {max(pcts):.2f}%")
        print(f"        media = {sum(pcts)/len(pcts):.2f}%")

    print("\nFASE 1 — paso 4a/5 completado.")


if __name__ == "__main__":
    main()
