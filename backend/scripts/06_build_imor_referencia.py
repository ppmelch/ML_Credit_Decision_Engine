"""
Construye el dataset de referencia del Índice de Morosidad (IMOR) de la
cartera de vivienda de la banca comercial mexicana, que sirve como
parámetro de calibración para la generación del target sintético en la
Fase 7 (probabilidad base de incumplimiento).

Combina dos fuentes oficiales de Banxico:

1. Serie mensual 2016-2022 extraída del Informe Trimestral
   Enero-Marzo 2022 (publicada en formato tabla web por Banxico).
2. Cifras anuales 2022-2024 con metodología IFRS9 reportadas en los
   Reportes de Estabilidad Financiera de Banxico.

Cambio metodológico
-------------------
A partir de enero de 2022, el IMOR se calcula bajo el estándar
internacional IFRS9: cartera clasificada como etapa 3 entre cartera
total. Antes de esa fecha, el IMOR era saldo de cartera vencida entre
saldo de cartera total. El cambio no implica deterioro real del crédito
sino una reclasificación contable.

Para fines de calibración del modelo de crédito remesa, se usa el
promedio post-IFRS9 (2022-2024) por ser la metodología vigente.

Insumos
-------
data/raw/banxico_imor_tabla_web_2016_2022_raw.csv
    Tabla web "Índices de Morosidad del Crédito al Sector Privado No
    Financiero" del Informe Trimestral Ene-Mar 2022 de Banxico.
    URL: https://www.banxico.org.mx/TablasWeb/informes-trimestrales/
         enero-marzo-2022/FC0831F1-8E84-4EE1-B8D0-85BF235754E2.html
    Codificación: ISO-8859-1 (latin-1).

Salidas
-------
data/raw/banxico_imor_vivienda_mensual.csv
    Serie mensual 2016-01 a 2022-03 con columnas: fecha, imor_vivienda_bc.
data/raw/banxico_imor_vivienda_referencia.json
    Resumen consolidado con:
    - Cifras anuales pre-IFRS9 (cierre de cada año 2016-2021).
    - Cifras anuales post-IFRS9 (2022-2024) reportadas en RSF.
    - Promedio post-IFRS9 propuesto como pd_base para Fase 7.
    - Citas APA7 de cada fuente.

Uso
---
    python scripts/06_build_imor_referencia.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = (
    PROJECT_ROOT / "data" / "raw" / "banxico_imor_tabla_web_2016_2022_raw.csv"
)
OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "raw" / "banxico_imor_vivienda_mensual.csv"
)
REFERENCE_JSON = (
    PROJECT_ROOT / "data" / "raw" / "banxico_imor_vivienda_referencia.json"
)

# Posición de la columna IMOR Vivienda Banca Comercial en el CSV crudo.
# Tras inspección: col 0 = fecha, col 5 = Vivienda - Banca Comercial.
COL_FECHA = 0
COL_VIVIENDA_BC = 5

# Mapeo de los meses abreviados en español usados por Banxico.
MES_MAP: dict[str, int] = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}

# IMOR vivienda banca comercial cifras anuales post-IFRS9. Estas cifras
# se reportan a cierre de diciembre de cada año en los Reportes de
# Estabilidad Financiera de Banxico:
#   2022: Reporte Primer Semestre 2023, sección Riesgo de Crédito.
#   2023: Reporte Primer Semestre 2024, sección Riesgo de Crédito.
#   2024: Reporte Primer Semestre 2025, sección Riesgo de Crédito.
IMOR_VIVIENDA_POST_IFRS9: dict[int, dict[str, float | str]] = {
    2022: {
        "imor": 2.6,
        "imor_ajustado": None,
        "fuente": "Reporte de Estabilidad Financiera, Primer Semestre 2023",
    },
    2023: {
        "imor": 2.6,
        "imor_ajustado": 3.2,
        "fuente": "Reporte de Estabilidad Financiera, Primer Semestre 2024",
    },
    2024: {
        "imor": 2.8,
        "imor_ajustado": 3.3,
        "fuente": "Reporte de Estabilidad Financiera, Primer Semestre 2025",
    },
}


def parse_fecha_es(raw: str) -> datetime | None:
    """
    Convierte etiquetas mensuales de Banxico ('ene-16', 'feb-22') a datetime.

    Parameters
    ----------
    raw : str
        Etiqueta cruda en formato 'mmm-aa'.

    Returns
    -------
    datetime | None
        Primer día del mes correspondiente, o None si no parsea.
    """
    raw = raw.strip().lower()
    if "-" not in raw:
        return None
    parts = raw.split("-")
    if len(parts) != 2:
        return None
    mes_str, anio_str = parts
    if mes_str not in MES_MAP:
        return None
    try:
        anio = 2000 + int(anio_str)
        return datetime(anio, MES_MAP[mes_str], 1)
    except ValueError:
        return None


def parse_float(raw: str) -> float | None:
    """Convierte string a float, manejando vacíos."""
    if raw is None:
        return None
    s = str(raw).strip().strip('"')
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    """Lee el CSV crudo, filtra la serie mensual y construye el JSON."""
    if not RAW_CSV.exists():
        print(f"ERROR: no se encontró {RAW_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Leyendo tabla web de Banxico ({RAW_CSV.name})...")
    # El archivo viene en ISO-8859-1 (latin-1).
    with open(RAW_CSV, "r", encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    print(f"      Filas totales en archivo: {len(rows)}")

    # --- Extraer la serie mensual ----------------------------------------
    print("[2/3] Extrayendo serie mensual de IMOR vivienda banca comercial...")
    monthly: list[dict[str, object]] = []
    for row in rows:
        if not row or len(row) <= COL_VIVIENDA_BC:
            continue
        fecha = parse_fecha_es(row[COL_FECHA])
        if fecha is None:
            continue
        valor = parse_float(row[COL_VIVIENDA_BC])
        if valor is None:
            continue
        monthly.append(
            {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "imor_vivienda_bc": valor,
            }
        )
    print(f"      Observaciones mensuales extraídas: {len(monthly)}")

    if not monthly:
        print("ERROR: no se extrajo ninguna fila de datos.", file=sys.stderr)
        sys.exit(1)

    # --- Persistir CSV mensual ------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fecha", "imor_vivienda_bc"])
        writer.writeheader()
        writer.writerows(monthly)
    print(f"      CSV mensual: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    # --- Calcular cierres anuales pre-IFRS9 (datos mensuales 2016-2021) -
    # Sólo se consideran años con metodología pre-IFRS9 (hasta 2021),
    # tomando el dato de diciembre como cifra de cierre anual.
    yearly_pre_ifrs9: dict[int, float] = {}
    for rec in monthly:
        year = int(rec["fecha"][:4])
        month = int(rec["fecha"][5:7])
        if year >= 2022:
            continue
        if month == 12:
            yearly_pre_ifrs9[year] = rec["imor_vivienda_bc"]

    # Promedio de los 12 meses por año (alternativa más estable).
    yearly_pre_ifrs9_promedio: dict[int, float] = {}
    grouped: dict[int, list[float]] = {}
    for rec in monthly:
        year = int(rec["fecha"][:4])
        if year >= 2022:
            continue
        grouped.setdefault(year, []).append(rec["imor_vivienda_bc"])
    for year, values in grouped.items():
        if len(values) >= 6:  # al menos medio año
            yearly_pre_ifrs9_promedio[year] = round(
                sum(values) / len(values), 4
            )

    # --- Construir parámetro de calibración pd_base ---------------------
    valores_post_ifrs9 = [
        IMOR_VIVIENDA_POST_IFRS9[y]["imor"]
        for y in IMOR_VIVIENDA_POST_IFRS9
    ]
    pd_base = round(sum(valores_post_ifrs9) / len(valores_post_ifrs9), 4)

    # --- Construir el JSON de referencia --------------------------------
    print("[3/3] Construyendo JSON de referencia...")

    reference = {
        "descripcion": (
            "Índice de Morosidad (IMOR) de la cartera de vivienda de la "
            "banca comercial mexicana. Sirve como parámetro de calibración "
            "para la generación del target sintético en la Fase 7 del "
            "modelo de crédito remesa."
        ),
        "cambio_metodologico": (
            "A partir de enero 2022, el IMOR se calcula bajo el estándar "
            "internacional IFRS9 (cartera etapa 3 / cartera total). Antes "
            "de esa fecha era cartera vencida / cartera total. El cambio "
            "no refleja deterioro real, sino reclasificación contable."
        ),
        "serie_mensual_pre_ifrs9": {
            "rango_temporal": "2016-01 a 2021-12",
            "n_observaciones": sum(
                1 for r in monthly if int(r["fecha"][:4]) <= 2021
            ),
            "csv_path": OUTPUT_CSV.name,
            "cierre_anual": yearly_pre_ifrs9,
            "promedio_anual": yearly_pre_ifrs9_promedio,
            "fuente": (
                "Banco de México, Informe Trimestral Enero-Marzo 2022, "
                "tabla 'Índices de Morosidad del Crédito al Sector Privado "
                "No Financiero'."
            ),
        },
        "serie_anual_post_ifrs9": {
            "rango_temporal": "2022 a 2024",
            "valores": IMOR_VIVIENDA_POST_IFRS9,
            "fuente": (
                "Banco de México, Reporte de Estabilidad Financiera, "
                "Primer Semestre de 2023, 2024 y 2025."
            ),
        },
        "transicion_2021_2022": {
            "imor_dic_2021": yearly_pre_ifrs9.get(2021),
            "imor_ene_2022_post_ifrs9": next(
                (
                    r["imor_vivienda_bc"]
                    for r in monthly
                    if r["fecha"] == "2022-01-01"
                ),
                None,
            ),
            "imor_dic_2022": IMOR_VIVIENDA_POST_IFRS9[2022]["imor"],
            "nota": (
                "El salto observado entre dic-2021 (3.15%) y los valores "
                "post-IFRS9 refleja el cambio metodológico, no deterioro "
                "del crédito hipotecario."
            ),
        },
        "calibracion_modelo": {
            "pd_base_post_ifrs9_promedio": pd_base,
            "interpretacion": (
                f"Tasa base de incumplimiento esperada a 12 meses sobre "
                f"cartera hipotecaria mexicana bajo metodología vigente "
                f"(IFRS9). Promedio simple de cierres anuales 2022-2024 = "
                f"{pd_base}%. Recomendado como pd_base para la Fase 7 del "
                f"modelo de crédito remesa."
            ),
            "valores_individuales_pct": {
                str(y): IMOR_VIVIENDA_POST_IFRS9[y]["imor"]
                for y in IMOR_VIVIENDA_POST_IFRS9
            },
        },
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "apa7_references": [
            (
                "Banco de México. (2022). Índices de Morosidad del Crédito "
                "al Sector Privado No Financiero. Informe Trimestral "
                "Enero-Marzo 2022. https://www.banxico.org.mx/TablasWeb/"
                "informes-trimestrales/enero-marzo-2022/"
                "FC0831F1-8E84-4EE1-B8D0-85BF235754E2.html"
            ),
            (
                "Banco de México. (2023). Reporte de Estabilidad "
                "Financiera, Primer Semestre 2023. "
                "https://www.banxico.org.mx/publicaciones-y-prensa/"
                "reportes-sobre-el-sistema-financiero/"
            ),
            (
                "Banco de México. (2024). Reporte de Estabilidad "
                "Financiera, Primer Semestre 2024. "
                "https://www.banxico.org.mx/publicaciones-y-prensa/"
                "reportes-sobre-el-sistema-financiero/"
            ),
            (
                "Banco de México. (2025). Reporte de Estabilidad "
                "Financiera, Primer Semestre 2025. "
                "https://www.banxico.org.mx/publicaciones-y-prensa/"
                "reportes-sobre-el-sistema-financiero/"
            ),
        ],
    }

    with open(REFERENCE_JSON, "w", encoding="utf-8") as fh:
        json.dump(reference, fh, indent=2, ensure_ascii=False)
    print(f"      JSON de referencia: {REFERENCE_JSON.relative_to(PROJECT_ROOT)}")

    # --- Verificación ----------------------------------------------------
    print("\nResumen de la serie mensual pre-IFRS9:")
    print(f"      Observaciones: {len(monthly)} meses")
    print(f"      Rango: {monthly[0]['fecha']} a {monthly[-1]['fecha']}")
    print(f"      Mín: {min(r['imor_vivienda_bc'] for r in monthly):.2f}%")
    print(f"      Máx: {max(r['imor_vivienda_bc'] for r in monthly):.2f}%")

    print("\nCierres anuales pre-IFRS9 (cierre dic):")
    for year in sorted(yearly_pre_ifrs9):
        print(f"      {year}: {yearly_pre_ifrs9[year]:.2f}%")

    print("\nCifras anuales post-IFRS9:")
    for year in sorted(IMOR_VIVIENDA_POST_IFRS9):
        v = IMOR_VIVIENDA_POST_IFRS9[year]
        ajustado = (
            f" (ajustado: {v['imor_ajustado']}%)"
            if v["imor_ajustado"] is not None
            else ""
        )
        print(f"      {year}: {v['imor']}%{ajustado}")

    print(f"\nParámetro de calibración Fase 7:")
    print(f"      pd_base = {pd_base}% (promedio 2022-2024 post-IFRS9)")

    print("\nFASE 1 — paso 5/5 completado.")


if __name__ == "__main__":
    main()
