"""
Construye el catálogo unificado de municipios de Jalisco como fuente de
verdad del proyecto.

Cruza dos insumos:

1. El catálogo geográfico de FAO GAUL / Earth Engine
   (`municipios_jalisco_completo.csv`), que aporta las geometrías
   (centroide y área) pero usa una codificación interna FAO sin
   correspondencia con la convención oficial mexicana.

2. El archivo de CONAPO IIM 2020 a nivel municipal
   (`conapo_iim_municipal_2020_raw.csv`), que aporta la clave INEGI
   oficial de 5 dígitos (`cve_mun`) y el nombre oficial del municipio
   con codificación latin-1.

El cruce se realiza por nombre normalizado (sin tildes, minúsculas,
recortado), aplicando la tabla de equivalencias documentada en el
paso 2 del proyecto para los 11 municipios cuyos nombres difieren
entre el catálogo FAO y los registros oficiales mexicanos.

Salida
------
data/raw/municipios_jalisco_catalogo.csv
    Catálogo unificado con clave INEGI como identificador principal.
    Columnas:
        cve_municipio       — clave INEGI 5 dígitos (string, fuente de verdad)
        municipio           — nombre oficial INEGI/CONAPO (con tildes)
        municipio_fao       — nombre original del catálogo FAO (auditoría)
        adm2_code_fao       — clave FAO original (auditoría)
        area_km2            — superficie del municipio (FAO GAUL)
        lat_centro          — latitud del centroide (FAO GAUL)
        lon_centro          — longitud del centroide (FAO GAUL)
data/raw/municipios_jalisco_catalogo_metadata.json
    Documentación del cruce: equivalencias aplicadas, exclusiones, fuentes.

Notas
-----
- San Ignacio Cerro Gordo (cve 14125) se excluye porque no está en el
  catálogo FAO base (creado en 2005, posterior a la versión administrativa
  usada por FAO GAUL).
- Los 11 municipios con discrepancias ortográficas se mapean usando la
  tabla NAME_EQUIVALENCES, validada contra registros oficiales INEGI.

Uso
---
    python scripts/00_build_municipios_catalog.py
"""

from __future__ import annotations

import csv
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAO_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_completo.csv"
CONAPO_CSV = PROJECT_ROOT / "data" / "raw" / "conapo_iim_municipal_2020_raw.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo_metadata.json"

JALISCO_CVE_ENT = "14"

# Equivalencias: nombre en catálogo FAO (clave) → nombre oficial INEGI/CONAPO (valor).
# Validadas contra el registro oficial de claves INEGI:
#   - 'Antonio Escobedo' es el nombre histórico de San Juanito de Escobedo (14007).
#   - 'Ciudad Guzmán' es la cabecera de Zapotlán el Grande (14023).
#   - 'Manuel M. Diéguez' fue el nombre de San Gabriel (14099) hasta 1992.
#   - 'Ciudad Venustiano Carranza' fue el nombre de Santa María del Oro (14056).
#   - El resto son truncamientos o normalizaciones ortográficas del catálogo FAO.
NAME_EQUIVALENCES: dict[str, str] = {
    "Antonio Escobedo": "San Juanito de Escobedo",
    "Atemajac De Brisuela": "Atemajac de Brizuela",
    "Ciudad Guzman": "Zapotlán el Grande",
    "Ciudad Venustiano C.": "Santa María del Oro",
    "Concepcion De Buenos A.": "Concepción de Buenos Aires",
    "Cuatitlan": "Cuautitlán de García Barragán",
    "Ixtlahuacan De Los Membri": "Ixtlahuacán de los Membrillos",
    "Manuel M. Dieguez": "San Gabriel",
    "San Cristobal De La B.": "San Cristóbal de la Barranca",
    "Sta. Maria De Los Angeles": "Santa María de los Ángeles",
    "Tlaquepaque": "San Pedro Tlaquepaque",
    "Yahualica De Gonzalez G.": "Yahualica de González Gallo",
}


def fix_mojibake(text: str) -> str:
    """
    Repara mojibake típico del archivo CONAPO (UTF-8 leído erróneamente
    como latin-1, p.ej. 'AcatlÃ¡n' → 'Acatlán').

    Si la cadena no tiene mojibake, la devuelve sin cambios. Si la
    reparación falla por alguna razón, también devuelve el original.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize(name: str) -> str:
    """Quita tildes, pasa a minúsculas y recorta espacios."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accent = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_accent.strip().lower()


def load_fao_catalog() -> list[dict[str, str]]:
    """Lee el catálogo FAO GAUL como lista de registros."""
    rows: list[dict[str, str]] = []
    with open(FAO_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def load_conapo_jalisco() -> tuple[dict[str, str], dict[str, str]]:
    """
    Lee el archivo CONAPO IIM y devuelve dos diccionarios:
        by_name        — {nombre_normalizado: cve_municipio_str_5d}
        official_name  — {nombre_normalizado: nombre_oficial_corregido}

    El archivo CONAPO está publicado con doble encoding (UTF-8 escrito
    como latin-1), por lo que los nombres requieren reparación de
    mojibake antes de poder cruzarse.
    """
    by_name: dict[str, str] = {}
    name_lookup: dict[str, str] = {}
    with open(CONAPO_CSV, "r", encoding="latin-1", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("cve_ent", "").strip() != JALISCO_CVE_ENT:
                continue
            cve = row.get("cve_mun", "").strip().zfill(5)
            nom_raw = row.get("nom_mun", "").strip()
            nom = fix_mojibake(nom_raw)
            key = normalize(nom)
            by_name[key] = cve
            name_lookup[key] = nom
    return by_name, name_lookup


def main() -> None:
    """Construye el catálogo unificado y lo persiste."""
    if not FAO_CSV.exists():
        print(f"ERROR: no se encontró {FAO_CSV}", file=sys.stderr)
        sys.exit(1)
    if not CONAPO_CSV.exists():
        print(f"ERROR: no se encontró {CONAPO_CSV}", file=sys.stderr)
        sys.exit(1)

    print("[1/4] Cargando catálogo FAO GAUL...")
    fao_rows = load_fao_catalog()
    print(f"      Filas FAO: {len(fao_rows)}")

    print("[2/4] Cargando claves INEGI desde CONAPO...")
    conapo_by_name, conapo_official_name = load_conapo_jalisco()
    print(f"      Municipios CONAPO Jalisco: {len(conapo_by_name)}")

    # --- Cruce -----------------------------------------------------------
    print("[3/4] Cruzando FAO ↔ CONAPO/INEGI...")
    output_rows: list[dict[str, object]] = []
    unmapped_fao: list[str] = []

    for fao in fao_rows:
        fao_name = fao["municipio"].strip()
        # Si el nombre FAO está en la tabla de equivalencias, usar el
        # nombre oficial; si no, usar el FAO directo.
        target_name = NAME_EQUIVALENCES.get(fao_name, fao_name)
        target_key = normalize(target_name)

        if target_key not in conapo_by_name:
            unmapped_fao.append(f"{fao_name!r} (buscado como {target_name!r})")
            continue

        cve = conapo_by_name[target_key]
        official_name = conapo_official_name[target_key]

        output_rows.append(
            {
                "cve_municipio": cve,
                "municipio": official_name,
                "municipio_fao": fao_name,
                "adm2_code_fao": fao["adm2_code"].strip(),
                "area_km2": fao["area_km2"].strip(),
                "lat_centro": fao["lat_centro"].strip(),
                "lon_centro": fao["lon_centro"].strip(),
            }
        )

    if unmapped_fao:
        print(
            f"\nADVERTENCIA: {len(unmapped_fao)} filas FAO sin contraparte:",
            file=sys.stderr,
        )
        for item in unmapped_fao:
            print(f"        - {item}", file=sys.stderr)

    # Validar que las claves INEGI sean únicas.
    cves = [r["cve_municipio"] for r in output_rows]
    if len(cves) != len(set(cves)):
        print("ADVERTENCIA: claves INEGI duplicadas detectadas.", file=sys.stderr)

    # Identificar los municipios CONAPO que quedaron fuera (esperamos 14125).
    cves_in_output = set(cves)
    cves_in_conapo = set(conapo_by_name.values())
    excluded_from_conapo = sorted(cves_in_conapo - cves_in_output)
    print(f"      Filas en catálogo unificado: {len(output_rows)}")
    print(f"      Municipios CONAPO no incluidos: {len(excluded_from_conapo)}")
    for cve in excluded_from_conapo:
        # Buscar el nombre oficial.
        official = next(
            (n for n, c in conapo_by_name.items() if c == cve), "?"
        )
        print(f"        - cve {cve} ({conapo_official_name.get(official, '?')})")

    # Ordenar por clave INEGI para reproducibilidad.
    output_rows.sort(key=lambda r: r["cve_municipio"])

    # --- Persistencia ----------------------------------------------------
    print(f"[4/4] Guardando salidas...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cve_municipio",
        "municipio",
        "municipio_fao",
        "adm2_code_fao",
        "area_km2",
        "lat_centro",
        "lon_centro",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"      Catálogo: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    metadata = {
        "descripcion": (
            "Catálogo unificado de municipios de Jalisco. Combina geometrías "
            "del catálogo FAO GAUL / Earth Engine con claves INEGI oficiales "
            "tomadas del archivo de CONAPO IIM 2020. Funciona como fuente de "
            "verdad para todos los cruces del proyecto."
        ),
        "n_municipios": len(output_rows),
        "n_municipios_jalisco_oficiales": 125,
        "n_municipios_excluidos": len(excluded_from_conapo),
        "municipios_excluidos_cves": excluded_from_conapo,
        "razon_exclusion": (
            "San Ignacio Cerro Gordo (14125) no está en el catálogo geográfico "
            "FAO base, posiblemente porque la versión administrativa de FAO GAUL "
            "es anterior a la creación del municipio en 2005."
        ),
        "fuentes": {
            "geometrias": "FAO Global Administrative Unit Layers (GAUL) vía Earth Engine",
            "claves_inegi_y_nombres": "CONAPO, IIM Mex-EUA 2020, archivo municipal",
        },
        "equivalencias_ortograficas": NAME_EQUIVALENCES,
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación final ----------------------------------------------
    print(f"\nVerificación cruzada — primeras y últimas 3 filas del catálogo:")
    for rec in output_rows[:3]:
        print(
            f"      {rec['cve_municipio']} | {rec['municipio']:<30} "
            f"| FAO: {rec['municipio_fao']}"
        )
    print("      ...")
    for rec in output_rows[-3:]:
        print(
            f"      {rec['cve_municipio']} | {rec['municipio']:<30} "
            f"| FAO: {rec['municipio_fao']}"
        )

    print("\nCatálogo unificado construido. Listo para re-ejecutar 02_ y 03_.")


if __name__ == "__main__":
    main()
