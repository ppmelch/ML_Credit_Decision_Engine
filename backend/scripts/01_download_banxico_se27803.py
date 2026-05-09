"""
Descarga la serie SE27803 de Banxico: Ingresos por remesas familiares,
total nacional, frecuencia mensual, en millones de dólares estadounidenses.

Rango descargado: 2013-01-01 a 2024-12-31 (144 observaciones esperadas).

Salidas
-------
data/raw/banxico_se27803_remesas_mensuales_nacional.csv
    Serie mensual con columnas: fecha, remesas_musd
data/raw/banxico_se27803_metadata.json
    Metadatos de la consulta: URL fuente, fecha de consulta, referencia APA7.

Verificación
------------
El script imprime al final la suma anual 2024 como prueba de consistencia.
El total esperado para 2024 ronda los USD 64,746 millones (récord histórico
reportado por Banxico).

Uso
---
    python scripts/01_download_banxico_se27803.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permitir importar desde src/ cuando se ejecuta el script directamente.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.banxico_client import BanxicoSIEClient  # noqa: E402


# --- Configuración de la descarga ----------------------------------------
SERIES_ID = "SE27803"
START_DATE = "2013-01-01"
END_DATE = "2024-12-31"
EXPECTED_OBSERVATIONS = 144  # 12 años × 12 meses

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = OUTPUT_DIR / "banxico_se27803_remesas_mensuales_nacional.csv"
METADATA_PATH = OUTPUT_DIR / "banxico_se27803_metadata.json"


def main() -> None:
    """Ejecuta la descarga, valida el resultado y persiste los archivos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = BanxicoSIEClient()
    print(f"[1/3] Descargando serie {SERIES_ID} ({START_DATE} a {END_DATE})...")
    df = client.fetch_series(SERIES_ID, START_DATE, END_DATE)

    # Renombrar 'valor' a un nombre semántico para el CSV de salida.
    df = df.rename(columns={"valor": "remesas_musd"})

    # --- Validaciones -----------------------------------------------------
    n_obs = len(df)
    n_missing = df["remesas_musd"].isna().sum()
    print(f"      Observaciones recibidas: {n_obs} (esperadas: {EXPECTED_OBSERVATIONS})")
    print(f"      Valores faltantes: {n_missing}")

    if n_obs != EXPECTED_OBSERVATIONS:
        print(
            f"      ADVERTENCIA: número de observaciones distinto al esperado.",
            file=sys.stderr,
        )

    # --- Persistencia del CSV --------------------------------------------
    print(f"[2/3] Guardando CSV en {CSV_PATH.relative_to(PROJECT_ROOT)}...")
    df.to_csv(CSV_PATH, index=False, date_format="%Y-%m-%d")

    # --- Persistencia de metadatos ---------------------------------------
    metadata = {
        "series_id": SERIES_ID,
        "series_name": (
            "Ingresos por remesas familiares, total nacional, frecuencia mensual"
        ),
        "unit": "Millones de dólares estadounidenses",
        "frequency": "Mensual",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "n_observations": int(n_obs),
        "n_missing": int(n_missing),
        "source_url": (
            f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
            f"{SERIES_ID}/datos/{START_DATE}/{END_DATE}"
        ),
        "portal_url": (
            "https://www.banxico.org.mx/SieAPIRest/service/v1/doc/"
            "consultaSeriesDeTiempo"
        ),
        "consultation_timestamp": client.get_consultation_timestamp(),
        "apa7_reference": (
            "Banco de México. (2025). Ingresos por remesas familiares, "
            "total nacional [Serie SE27803]. Sistema de Información "
            "Económica. https://www.banxico.org.mx/SieAPIRest/service/v1/"
            f"series/{SERIES_ID}"
        ),
    }
    print(f"      Guardando metadatos en {METADATA_PATH.relative_to(PROJECT_ROOT)}...")
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # --- Verificación cruzada: suma anual 2024 ---------------------------
    print("[3/3] Verificación cruzada — totales anuales:")
    df_yearly = (
        df.assign(year=df["fecha"].dt.year)
        .groupby("year")["remesas_musd"]
        .sum()
        .round(2)
    )
    for year, total in df_yearly.items():
        marker = "  <-- récord 2024" if year == 2024 else ""
        print(f"      {year}: USD {total:>12,.2f} M{marker}")

    total_2024 = df_yearly.get(2024, None)
    if total_2024 is not None:
        # Banxico reportó USD 64,745 M para 2024 (récord histórico).
        if abs(total_2024 - 64_746) < 500:
            print(
                f"\n      ✓ Suma 2024 = {total_2024:,.2f} M coincide con el "
                f"récord histórico reportado (~USD 64,746 M)."
            )
        else:
            print(
                f"\n      ADVERTENCIA: suma 2024 = {total_2024:,.2f} M se aleja "
                f"del valor esperado (~USD 64,746 M).",
                file=sys.stderr,
            )

    print("\nFASE 1 — paso 1/5 completado.")


if __name__ == "__main__":
    main()
