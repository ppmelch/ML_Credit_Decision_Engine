"""
Parser del cuadro CE166 de Banxico: Ingresos por remesas, distribución
por municipio, frecuencia trimestral, en millones de dólares estadounidenses.

Versión refactorizada (post-construcción del catálogo unificado): cruza
las columnas de Banxico contra el catálogo del proyecto usando el nombre
oficial INEGI/CONAPO del municipio (con tildes correctas), que coincide
exactamente con cómo Banxico publica los nombres en el cuadro CE166.
La clave INEGI (`cve_municipio`) se hereda del catálogo, no se reconstruye.

Insumos
-------
data/raw/banxico_ce166_remesas_municipales_raw.csv
    Cuadro CE166 descargado manualmente del portal SIE.
    URL: https://www.banxico.org.mx/SieInternet/consultarDirectorioInternetAction.do
         ?accion=consultarCuadro&idCuadro=CE166&locale=es
    Tipo de información: Niveles
    Codificación: Windows-1252 (cp1252). El portal SIE de Banxico exporta
    los CSV con codificación Windows estándar; leerlo como UTF-8 produce
    UnicodeDecodeError en cualquier carácter acentuado.
data/raw/municipios_jalisco_catalogo.csv
    Catálogo unificado del proyecto (generado por 00_build_municipios_catalog.py).

Salidas
-------
data/raw/banxico_ce166_jalisco_trimestral.csv
    Serie trimestral por municipio en formato largo, con clave INEGI.
data/raw/banxico_ce166_metadata.json
    Metadatos: fuente, fecha de consulta, referencia APA7.

Uso
---
    python scripts/02_parse_banxico_ce166.py
"""

from __future__ import annotations

import csv
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "banxico_ce166_remesas_municipales_raw.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "banxico_ce166_jalisco_trimestral.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "banxico_ce166_metadata.json"

# Estructura del CSV crudo (1-indexed):
#   Fila 11 → títulos descriptivos
#   Fila 19 → claves SIE de la serie (SE41187, SE41188, ...)
#   Fila 20+ → datos (col A = fecha trimestral, col B+ = valores)
TITLE_ROW = 10   # 0-indexed
SERIES_ROW = 18  # 0-indexed
FIRST_DATA_ROW = 19  # 0-indexed

# Municipios deliberadamente excluidos del análisis.
EXCLUDED_FROM_BANXICO: list[str] = [
    "San Ignacio Cerro Gordo",  # No está en el catálogo geográfico base.
    "No identificado",          # Remesas no asignables a un municipio específico.
]

# Equivalencias específicas Banxico → nombre oficial del catálogo del proyecto.
# Banxico usa el nombre histórico de algunos municipios mientras que
# CONAPO/INEGI publica el nombre oficial vigente. Estas son discrepancias
# que NO se resuelven en `00_build_municipios_catalog.py` porque allí el
# cruce es FAO ↔ CONAPO; las que aparecen aquí son específicas del cuadro
# CE166 de Banxico.
BANXICO_NAME_EQUIVALENCES: dict[str, str] = {
    "Tlaquepaque": "San Pedro Tlaquepaque",
}


def normalize(name: str) -> str:
    """Normaliza nombres para cruce robusto: sin tildes, minúsculas, recortado."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accent.strip().lower()


def parse_banxico_value(raw: str) -> float | None:
    """Convierte un valor crudo de Banxico (con coma de miles) a float."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() == "N/E":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_quarter_date(raw: str) -> tuple[datetime, str]:
    """Convierte 'dd/mm/yyyy' al primer día del trimestre y etiqueta 'YYYY-Q#'."""
    dt = datetime.strptime(raw.strip(), "%d/%m/%Y")
    quarter = (dt.month - 1) // 3 + 1
    return dt, f"{dt.year}-Q{quarter}"


def load_catalog() -> dict[str, dict[str, str]]:
    """
    Carga el catálogo unificado indexado por nombre oficial normalizado.

    Returns
    -------
    dict
        {nombre_normalizado: {'cve_municipio', 'municipio'}}
    """
    catalog: dict[str, dict[str, str]] = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cve = row["cve_municipio"].strip().zfill(5)
            nombre_oficial = row["municipio"].strip()
            catalog[normalize(nombre_oficial)] = {
                "cve_municipio": cve,
                "municipio": nombre_oficial,
            }
    return catalog


def main() -> None:
    """Ejecuta el parseo completo y persiste el CSV largo + metadatos."""
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

    print(f"[2/4] Leyendo CSV crudo de Banxico ({RAW_CSV.name})...")
    # Banxico exporta el cuadro CE166 desde el portal SIE con codificación
    # Windows-1252 (cp1252), no UTF-8. Intentar leerlo como UTF-8 produce
    # un UnicodeDecodeError en el primer byte de cualquier carácter
    # acentuado (ej. la 'é' de "Banco de México" en la fila 2).
    with open(RAW_CSV, "r", encoding="cp1252", newline="") as fh:
        rows = list(csv.reader(fh))

    titles = rows[TITLE_ROW]
    series_codes = rows[SERIES_ROW]
    print(f"      Filas totales: {len(rows)}, columnas: {len(titles)}")

    # --- Identificar columnas de Jalisco que están en el catálogo --------
    print("[3/4] Filtrando columnas de Jalisco contra el catálogo...")
    target_columns: list[dict[str, str | int]] = []
    excluded_log: list[str] = []

    for col_idx, title in enumerate(titles):
        if not title:
            continue
        parts = [p.strip() for p in title.split(",")]
        if len(parts) < 4 or parts[2].lower() != "jalisco":
            continue
        municipio_banxico = parts[3]
        if municipio_banxico in EXCLUDED_FROM_BANXICO:
            excluded_log.append(municipio_banxico)
            continue
        # Aplicar equivalencias Banxico → nombre oficial del catálogo,
        # antes de buscar la coincidencia.
        municipio_para_cruce = BANXICO_NAME_EQUIVALENCES.get(
            municipio_banxico, municipio_banxico
        )
        key = normalize(municipio_para_cruce)
        if key not in catalog:
            excluded_log.append(
                f"{municipio_banxico} (sin contraparte en catálogo unificado)"
            )
            continue
        record = catalog[key]
        target_columns.append(
            {
                "col_idx": col_idx,
                "municipio_banxico": municipio_banxico,
                "municipio": record["municipio"],
                "cve_municipio": record["cve_municipio"],
                "serie_sie": series_codes[col_idx].strip(),
            }
        )

    print(f"      Columnas seleccionadas: {len(target_columns)}")
    print(f"      Columnas excluidas: {len(excluded_log)}")
    for item in excluded_log:
        print(f"        - {item}")

    if len(target_columns) != len(catalog):
        print(
            f"\nERROR: se esperaban {len(catalog)} columnas, "
            f"se obtuvieron {len(target_columns)}.\n"
            f"Causa más probable: el archivo CSV crudo se está leyendo con "
            f"un encoding incorrecto, generando mojibake que rompe el cruce "
            f"de nombres. El archivo de Banxico viene en cp1252 (Windows-1252), "
            f"no en UTF-8 ni latin-1.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Iterar filas y armar formato largo ------------------------------
    print("[4/4] Transformando a formato largo...")
    long_records: list[dict[str, object]] = []
    n_rows_data = 0
    n_missing = 0

    for row in rows[FIRST_DATA_ROW:]:
        if not row or not row[0].strip():
            continue
        try:
            fecha, trimestre = parse_quarter_date(row[0])
        except (ValueError, IndexError):
            continue
        n_rows_data += 1

        for col in target_columns:
            col_idx = col["col_idx"]
            raw_value = row[col_idx] if col_idx < len(row) else ""
            value = parse_banxico_value(raw_value)
            if value is None:
                n_missing += 1
            long_records.append(
                {
                    "cve_municipio": col["cve_municipio"],
                    "municipio": col["municipio"],
                    "serie_sie": col["serie_sie"],
                    "fecha": fecha.strftime("%Y-%m-%d"),
                    "trimestre": trimestre,
                    "remesas_musd": value,
                }
            )

    print(f"      Filas de datos procesadas: {n_rows_data}")
    print(f"      Registros largos generados: {len(long_records)}")
    print(f"      Valores faltantes: {n_missing}")

    # --- Persistencia ----------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cve_municipio",
        "municipio",
        "serie_sie",
        "fecha",
        "trimestre",
        "remesas_musd",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in long_records:
            if rec["remesas_musd"] is None:
                rec = {**rec, "remesas_musd": ""}
            writer.writerow(rec)
    print(f"\n      CSV largo: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    metadata = {
        "cuadro_id": "CE166",
        "cuadro_nombre": "Ingresos por remesas, distribución por municipio",
        "unit": "Millones de dólares estadounidenses",
        "frequency": "Trimestral",
        "tipo_informacion": "Niveles",
        "n_municipios_jalisco_publicados_banxico": 125,
        "n_municipios_incluidos_en_dataset": len(target_columns),
        "n_observaciones_trimestrales": n_rows_data,
        "n_registros_largos": len(long_records),
        "n_valores_faltantes": n_missing,
        "municipios_excluidos": EXCLUDED_FROM_BANXICO,
        "equivalencias_nombre_banxico": BANXICO_NAME_EQUIVALENCES,
        "metodo_cruce": (
            "Cruce por nombre oficial normalizado (sin tildes, minúsculas) "
            "contra el catálogo unificado del proyecto. La clave INEGI "
            "se hereda del catálogo, que a su vez la deriva del archivo "
            "CONAPO IIM 2020."
        ),
        "source_url": (
            "https://www.banxico.org.mx/SieInternet/consultarDirectorioInternetAction.do"
            "?accion=consultarCuadro&idCuadro=CE166&locale=es"
        ),
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_file": RAW_CSV.name,
        "raw_file_encoding": "cp1252 (Windows-1252)",
        "apa7_reference": (
            "Banco de México. (2025). Ingresos por remesas, distribución por "
            "municipio [Cuadro CE166]. Sistema de Información Económica. "
            "https://www.banxico.org.mx/SieInternet/consultarDirectorioInternet"
            "Action.do?accion=consultarCuadro&idCuadro=CE166"
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación cruzada --------------------------------------------
    print("\nVerificación cruzada — total Jalisco por año:")
    annual: dict[int, float] = {}
    for rec in long_records:
        val = rec["remesas_musd"]
        if val == "" or val is None:
            continue
        year = int(rec["fecha"][:4])
        annual[year] = annual.get(year, 0.0) + float(val)
    for year in sorted(annual):
        print(f"      {year}: USD {annual[year]:>10,.2f} M")

    print("\nFASE 1 — paso 2/5 completado.")


if __name__ == "__main__":
    main()
