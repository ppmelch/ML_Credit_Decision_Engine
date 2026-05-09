"""
Generación del padrón sintético de hogares receptores de remesas en Jalisco.

Este script construye un padrón de 10,000 hogares sintéticos con atributos
demográficos del receptor y municipio de residencia. La asignación municipal
es proporcional al volumen real de remesas 2024 por municipio (Banxico CE166
desagregado), de modo que la concentración geográfica de los hogares
sintéticos refleja la concentración real del flujo de remesas en Jalisco.

Las distribuciones demográficas se calibran con valores agregados publicados
por CEMLA (Centro de Estudios Monetarios Latinoamericanos), BBVA Research
sobre el Anuario de Migración y Remesas México, e INEGI (Censo 2020) para
indicadores de hogares receptores.

Insumos
-------
data/processed/jalisco_municipal_remesas_mensuales.csv
    Series mensuales por municipio (Fase 3) — se usa el total 2024.
data/raw/municipios_jalisco_catalogo.csv
    Catálogo de 124 municipios con clave INEGI y nombre.

Salidas
-------
data/processed/padron_hogares_sinteticos.csv
    Padrón con 10,000 hogares (una fila por hogar). Columnas:
    id_hogar, cve_municipio, municipio, edad_receptor, genero_receptor,
    escolaridad, n_dependientes, tipo_vivienda_actual,
    antiguedad_recepcion_meses, remesa_mediana_esperada_usd.
data/processed/padron_hogares_sinteticos_metadata.json
    Parámetros de calibración, fuentes, sanity checks, totales realizados.

Reproducibilidad
----------------
Todas las muestras aleatorias usan numpy.random.default_rng con SEED=42.
Cualquier cambio en el orden de los municipios o en los parámetros altera
el padrón resultante.

Uso
---
    python scripts/10_generar_padron_hogares.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MENSUAL_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jalisco_municipal_remesas_mensuales.csv"
)
CATALOG_CSV = PROJECT_ROOT / "data" / "raw" / "municipios_jalisco_catalogo.csv"

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "padron_hogares_sinteticos.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "padron_hogares_sinteticos_metadata.json"
)

# ---------------------------------------------------------------------------
# Hiperparámetros del padrón
# ---------------------------------------------------------------------------
N_HOGARES = 10_000
SEED = 42
ANIO_PESOS = 2024  # Año cuyas remesas se usan para asignar hogares.

# ---------------------------------------------------------------------------
# Calibración de distribuciones demográficas
# ---------------------------------------------------------------------------
# Edad del receptor: normal truncada en [18, 80].
# Fuente: CEMLA (2024) reporta edad media del receptor entre 43-47 años.
EDAD_MEDIA = 45.0
EDAD_DESV = 12.0
EDAD_MIN = 18
EDAD_MAX = 80

# Género del receptor: P(femenino) ≈ 0.65.
# Fuente: BBVA Research, Anuario de Migración y Remesas México 2024.
P_FEMENINO = 0.65

# Escolaridad: distribución multinomial.
# Fuente: INEGI Censo 2020, características de hogares receptores en
# entidades de alta migración del centro-occidente.
ESCOLARIDAD_NIVELES = [
    "sin_estudios",
    "primaria",
    "secundaria",
    "media_superior",
    "superior",
]
ESCOLARIDAD_PROBS = [0.08, 0.32, 0.28, 0.20, 0.12]

# Dependientes: Poisson truncada en [0, 8].
# Fuente: CEMLA — tamaño promedio del hogar receptor ≈ 3.3 personas
# (≈ 2.3 dependientes además del receptor).
DEPENDIENTES_LAMBDA = 2.3
DEPENDIENTES_MAX = 8

# Tipo de vivienda actual.
# Fuente: INEGI Censo 2020 — tenencia en hogares receptores de remesas en
# Jalisco. La tasa de vivienda propia es notablemente alta.
VIVIENDA_TIPOS = ["propia", "rentada", "prestada"]
VIVIENDA_PROBS = [0.62, 0.22, 0.16]

# Antigüedad del flujo de remesas en meses: lognormal truncada [24, 240].
# Fuente: BBVA Research — duración media del flujo ~5 años (60 meses).
ANTIGUEDAD_MU_LN = float(np.log(60.0))
ANTIGUEDAD_SIGMA_LN = 0.70
ANTIGUEDAD_MIN = 24
ANTIGUEDAD_MAX = 240

# Remesa mediana esperada (USD/mes): lognormal.
# Fuente: Banxico 2024 — monto promedio por envío ≈ 390 USD; usamos esto
# como mediana esperada del flujo mensual del hogar (asumiendo ~1 envío al
# mes en promedio).
REMESA_MU_LN = float(np.log(380.0))
REMESA_SIGMA_LN = 0.45


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
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


def cargar_remesas_anuales(anio: int) -> dict[str, float]:
    """
    Carga el total de remesas en USD millones por municipio para un año.

    Parameters
    ----------
    anio : int
        Año de referencia (típicamente 2024).

    Returns
    -------
    dict[str, float]
        cve_municipio -> remesas_anuales_musd
    """
    totales: dict[str, float] = {}
    with open(MENSUAL_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            fecha = row["fecha"]
            if not fecha.startswith(str(anio)):
                continue
            cve = row["cve_municipio"].strip().zfill(5)
            v = parse_float(row["remesas_musd_mensual"])
            if v is None:
                continue
            totales[cve] = totales.get(cve, 0.0) + v
    return totales


def cargar_catalogo() -> dict[str, str]:
    """Carga catálogo cve_municipio -> nombre."""
    catalogo: dict[str, str] = {}
    with open(CATALOG_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cve = row["cve_municipio"].strip().zfill(5)
            catalogo[cve] = row["municipio"].strip()
    return catalogo


def asignar_hogares_por_municipio(
    totales: dict[str, float], n_hogares: int
) -> dict[str, int]:
    """
    Asigna hogares a municipios proporcionalmente a las remesas anuales.

    Aplica el método de los residuos mayores (Hamilton) para garantizar
    que la suma sea exactamente n_hogares y que cada municipio reciba al
    menos 1 hogar (piso operativo: ningún municipio queda sin
    representación). Para evitar sesgos, primero se asigna el piso
    obligatorio y luego se distribuyen los hogares restantes.

    Parameters
    ----------
    totales : dict[str, float]
        Remesas anuales por municipio.
    n_hogares : int
        Total de hogares a asignar.

    Returns
    -------
    dict[str, int]
        cve_municipio -> número de hogares.
    """
    n_municipios = len(totales)
    if n_hogares < n_municipios:
        raise ValueError(
            f"n_hogares={n_hogares} < n_municipios={n_municipios}; "
            "no es posible asignar al menos 1 hogar por municipio."
        )

    suma_total = sum(totales.values())
    cves = sorted(totales.keys())

    # Asignación inicial: 1 hogar por municipio (piso operativo).
    asignacion: dict[str, int] = {cve: 1 for cve in cves}
    restantes = n_hogares - n_municipios

    # Distribución proporcional de los restantes por método de
    # residuos mayores.
    cuotas = {cve: restantes * totales[cve] / suma_total for cve in cves}
    enteros = {cve: int(np.floor(cuotas[cve])) for cve in cves}
    asignados = sum(enteros.values())
    faltan = restantes - asignados

    # Repartir los hogares restantes entre municipios con mayor parte
    # fraccionaria.
    fracs = sorted(
        ((cuotas[cve] - enteros[cve], cve) for cve in cves),
        key=lambda x: x[0],
        reverse=True,
    )
    for i in range(faltan):
        _, cve = fracs[i]
        enteros[cve] += 1

    for cve in cves:
        asignacion[cve] += enteros[cve]

    assert sum(asignacion.values()) == n_hogares
    return asignacion


# ---------------------------------------------------------------------------
# Muestreo demográfico
# ---------------------------------------------------------------------------
def sample_normal_truncada(
    rng: np.random.Generator, mu: float, sigma: float, lo: float, hi: float, n: int
) -> np.ndarray:
    """Muestrea normal truncada por rechazo (eficiente cuando lo<<mu<<hi)."""
    out = np.empty(n)
    llenos = 0
    while llenos < n:
        cand = rng.normal(mu, sigma, size=n - llenos)
        validos = cand[(cand >= lo) & (cand <= hi)]
        k = min(len(validos), n - llenos)
        out[llenos : llenos + k] = validos[:k]
        llenos += k
    return out


def sample_poisson_truncada(
    rng: np.random.Generator, lam: float, hi: int, n: int
) -> np.ndarray:
    """Muestrea Poisson(lam) truncada en [0, hi]."""
    out = np.empty(n, dtype=int)
    llenos = 0
    while llenos < n:
        cand = rng.poisson(lam, size=n - llenos)
        validos = cand[cand <= hi]
        k = min(len(validos), n - llenos)
        out[llenos : llenos + k] = validos[:k]
        llenos += k
    return out


def sample_lognormal_truncada(
    rng: np.random.Generator,
    mu_ln: float,
    sigma_ln: float,
    lo: float,
    hi: float,
    n: int,
) -> np.ndarray:
    """Muestrea lognormal truncada por rechazo."""
    out = np.empty(n)
    llenos = 0
    while llenos < n:
        cand = rng.lognormal(mu_ln, sigma_ln, size=n - llenos)
        validos = cand[(cand >= lo) & (cand <= hi)]
        k = min(len(validos), n - llenos)
        out[llenos : llenos + k] = validos[:k]
        llenos += k
    return out


# ---------------------------------------------------------------------------
# Construcción del padrón
# ---------------------------------------------------------------------------
def generar_padron(
    asignacion: dict[str, int], catalogo: dict[str, str], rng: np.random.Generator
) -> list[dict[str, object]]:
    """
    Genera la lista de hogares con sus atributos demográficos.

    Parameters
    ----------
    asignacion : dict[str, int]
        Hogares por municipio.
    catalogo : dict[str, str]
        Nombres de municipios.
    rng : np.random.Generator
        Generador de números aleatorios reproducible.

    Returns
    -------
    list[dict]
        Padrón con un dict por hogar.
    """
    n = sum(asignacion.values())

    # Vector de cve_municipio expandido según la asignación.
    cves_expandido: list[str] = []
    for cve in sorted(asignacion.keys()):
        cves_expandido.extend([cve] * asignacion[cve])
    assert len(cves_expandido) == n

    # Muestreo de atributos (vectorizado).
    edades = sample_normal_truncada(
        rng, EDAD_MEDIA, EDAD_DESV, EDAD_MIN, EDAD_MAX, n
    ).round().astype(int)

    generos_idx = rng.binomial(1, P_FEMENINO, size=n)
    generos = np.where(generos_idx == 1, "F", "M")

    escolaridades = rng.choice(
        ESCOLARIDAD_NIVELES, size=n, p=ESCOLARIDAD_PROBS
    )

    dependientes = sample_poisson_truncada(
        rng, DEPENDIENTES_LAMBDA, DEPENDIENTES_MAX, n
    )

    viviendas = rng.choice(VIVIENDA_TIPOS, size=n, p=VIVIENDA_PROBS)

    antiguedad = (
        sample_lognormal_truncada(
            rng,
            ANTIGUEDAD_MU_LN,
            ANTIGUEDAD_SIGMA_LN,
            ANTIGUEDAD_MIN,
            ANTIGUEDAD_MAX,
            n,
        )
        .round()
        .astype(int)
    )

    remesa_mediana = rng.lognormal(REMESA_MU_LN, REMESA_SIGMA_LN, size=n)

    padron: list[dict[str, object]] = []
    for i in range(n):
        cve = cves_expandido[i]
        padron.append(
            {
                "id_hogar": f"H{i+1:06d}",
                "cve_municipio": cve,
                "municipio": catalogo.get(cve, "?"),
                "edad_receptor": int(edades[i]),
                "genero_receptor": str(generos[i]),
                "escolaridad": str(escolaridades[i]),
                "n_dependientes": int(dependientes[i]),
                "tipo_vivienda_actual": str(viviendas[i]),
                "antiguedad_recepcion_meses": int(antiguedad[i]),
                "remesa_mediana_esperada_usd": round(
                    float(remesa_mediana[i]), 2
                ),
            }
        )
    return padron


# ---------------------------------------------------------------------------
# Diagnóstico y validación
# ---------------------------------------------------------------------------
def calcular_diagnostico(
    padron: list[dict[str, object]],
    asignacion: dict[str, int],
    totales_remesas: dict[str, float],
) -> dict[str, object]:
    """Construye un diccionario de validaciones y resúmenes."""
    n = len(padron)

    edades = np.array([h["edad_receptor"] for h in padron])
    deps = np.array([h["n_dependientes"] for h in padron])
    antig = np.array([h["antiguedad_recepcion_meses"] for h in padron])
    remesa = np.array([h["remesa_mediana_esperada_usd"] for h in padron])

    # Conteos categóricos.
    def conteos(items: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in items:
            out[v] = out.get(v, 0) + 1
        return out

    generos = conteos([h["genero_receptor"] for h in padron])
    escolaridades = conteos([h["escolaridad"] for h in padron])
    viviendas = conteos([h["tipo_vivienda_actual"] for h in padron])

    # Sanity check: correlación entre asignación realizada y peso de
    # remesas.
    cves = sorted(asignacion.keys())
    asig_arr = np.array([asignacion[c] for c in cves], dtype=float)
    rem_arr = np.array([totales_remesas[c] for c in cves], dtype=float)
    corr_asignacion = float(np.corrcoef(asig_arr, rem_arr)[0, 1])

    # Top 10 municipios.
    top_10 = sorted(asignacion.items(), key=lambda kv: kv[1], reverse=True)[
        :10
    ]
    pct_gdl_zap = (
        asignacion.get("14039", 0) + asignacion.get("14120", 0)
    ) / n

    return {
        "n_hogares": n,
        "n_municipios_con_hogares": sum(
            1 for v in asignacion.values() if v > 0
        ),
        "asignacion_municipio_correlacion_remesas": round(corr_asignacion, 6),
        "edad_receptor": {
            "media": float(np.mean(edades)),
            "mediana": float(np.median(edades)),
            "p25": float(np.percentile(edades, 25)),
            "p75": float(np.percentile(edades, 75)),
            "min": int(np.min(edades)),
            "max": int(np.max(edades)),
            "target_media": EDAD_MEDIA,
        },
        "genero_receptor": {
            "F": generos.get("F", 0),
            "M": generos.get("M", 0),
            "pct_F_realizado": generos.get("F", 0) / n,
            "pct_F_target": P_FEMENINO,
        },
        "escolaridad": {
            **{
                k: {
                    "n": escolaridades.get(k, 0),
                    "pct_realizado": escolaridades.get(k, 0) / n,
                    "pct_target": p,
                }
                for k, p in zip(ESCOLARIDAD_NIVELES, ESCOLARIDAD_PROBS)
            }
        },
        "n_dependientes": {
            "media": float(np.mean(deps)),
            "target_media": DEPENDIENTES_LAMBDA,
            "min": int(np.min(deps)),
            "max": int(np.max(deps)),
        },
        "tipo_vivienda_actual": {
            **{
                k: {
                    "n": viviendas.get(k, 0),
                    "pct_realizado": viviendas.get(k, 0) / n,
                    "pct_target": p,
                }
                for k, p in zip(VIVIENDA_TIPOS, VIVIENDA_PROBS)
            }
        },
        "antiguedad_recepcion_meses": {
            "mediana": float(np.median(antig)),
            "p25": float(np.percentile(antig, 25)),
            "p75": float(np.percentile(antig, 75)),
            "min": int(np.min(antig)),
            "max": int(np.max(antig)),
            "target_mediana": 60,
        },
        "remesa_mediana_esperada_usd": {
            "mediana": float(np.median(remesa)),
            "media": float(np.mean(remesa)),
            "p25": float(np.percentile(remesa, 25)),
            "p75": float(np.percentile(remesa, 75)),
            "target_mediana": 380.0,
        },
        "top_10_municipios": [
            {
                "cve_municipio": cve,
                "n_hogares": k,
                "pct_padron": k / n,
            }
            for cve, k in top_10
        ],
        "pct_hogares_gdl_zapopan": round(pct_gdl_zap, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la generación del padrón sintético."""
    for path in [MENSUAL_CSV, CATALOG_CSV]:
        if not path.exists():
            print(f"ERROR: no se encontró {path}", file=sys.stderr)
            sys.exit(1)

    print(f"[1/4] Cargando remesas {ANIO_PESOS} por municipio...")
    totales = cargar_remesas_anuales(ANIO_PESOS)
    print(
        f"      Municipios con remesas {ANIO_PESOS}: {len(totales)}"
    )
    print(
        f"      Total Jalisco {ANIO_PESOS}: USD "
        f"{sum(totales.values()):,.2f} M"
    )

    print("[2/4] Cargando catálogo de municipios...")
    catalogo = cargar_catalogo()
    print(f"      Municipios en catálogo: {len(catalogo)}")

    print(f"[3/4] Asignando {N_HOGARES:,} hogares a municipios...")
    asignacion = asignar_hogares_por_municipio(totales, N_HOGARES)
    print(
        f"      Total asignado: {sum(asignacion.values()):,} "
        f"(esperado: {N_HOGARES:,})"
    )

    print("[4/4] Generando atributos demográficos...")
    rng = np.random.default_rng(SEED)
    padron = generar_padron(asignacion, catalogo, rng)
    print(f"      Hogares generados: {len(padron):,}")

    # --- Persistencia ---------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id_hogar",
        "cve_municipio",
        "municipio",
        "edad_receptor",
        "genero_receptor",
        "escolaridad",
        "n_dependientes",
        "tipo_vivienda_actual",
        "antiguedad_recepcion_meses",
        "remesa_mediana_esperada_usd",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(padron)
    print(f"\n      CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    diag = calcular_diagnostico(padron, asignacion, totales)

    metadata = {
        "descripcion": (
            "Padrón sintético de 10,000 hogares receptores de remesas en "
            "Jalisco. Asignación municipal proporcional al volumen real "
            f"de remesas {ANIO_PESOS} (Banxico CE166 desagregado vía "
            "Chow-Lin, Fase 3). Atributos demográficos calibrados con "
            "distribuciones agregadas de CEMLA, BBVA Research e INEGI "
            "Censo 2020."
        ),
        "n_hogares": N_HOGARES,
        "seed": SEED,
        "anio_pesos_municipales": ANIO_PESOS,
        "fuentes_calibracion": {
            "edad_receptor": "CEMLA (2024) — edad media receptor 43-47.",
            "genero_receptor": (
                "BBVA Research, Anuario de Migración y Remesas México 2024 — "
                "60-70% receptores son mujeres."
            ),
            "escolaridad": (
                "INEGI Censo 2020, hogares receptores en entidades de alta "
                "migración del centro-occidente."
            ),
            "n_dependientes": (
                "CEMLA — tamaño promedio del hogar receptor ≈ 3.3 personas."
            ),
            "tipo_vivienda_actual": (
                "INEGI Censo 2020 — tenencia de vivienda en hogares "
                "receptores de remesas en Jalisco."
            ),
            "antiguedad_recepcion_meses": (
                "BBVA Research — duración media del flujo ~5 años."
            ),
            "remesa_mediana_esperada_usd": (
                "Banxico (2024) — monto promedio por envío ≈ 380 USD."
            ),
        },
        "parametros_distribuciones": {
            "edad_normal_truncada": {
                "mu": EDAD_MEDIA,
                "sigma": EDAD_DESV,
                "lo": EDAD_MIN,
                "hi": EDAD_MAX,
            },
            "genero_bernoulli_p_F": P_FEMENINO,
            "escolaridad_categorica": dict(
                zip(ESCOLARIDAD_NIVELES, ESCOLARIDAD_PROBS)
            ),
            "dependientes_poisson_truncada": {
                "lambda": DEPENDIENTES_LAMBDA,
                "max": DEPENDIENTES_MAX,
            },
            "vivienda_categorica": dict(zip(VIVIENDA_TIPOS, VIVIENDA_PROBS)),
            "antiguedad_lognormal_truncada": {
                "mu_ln": ANTIGUEDAD_MU_LN,
                "sigma_ln": ANTIGUEDAD_SIGMA_LN,
                "lo": ANTIGUEDAD_MIN,
                "hi": ANTIGUEDAD_MAX,
            },
            "remesa_mediana_lognormal": {
                "mu_ln": REMESA_MU_LN,
                "sigma_ln": REMESA_SIGMA_LN,
            },
        },
        "diagnostico": diag,
        "asignacion_por_municipio": asignacion,
        "consultation_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"      Metadatos: {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    # --- Verificación ---------------------------------------------------
    print("\nVerificación cruzada:")
    print(
        f"      Hogares totales: {diag['n_hogares']:,} "
        f"(esperado: {N_HOGARES:,})"
    )
    print(
        f"      Municipios con ≥1 hogar: "
        f"{diag['n_municipios_con_hogares']} de {len(catalogo)}"
    )
    print(
        f"      Correlación asignación-remesas: "
        f"{diag['asignacion_municipio_correlacion_remesas']:.4f} "
        f"(esperado > 0.99)"
    )

    print(
        f"\n      Edad receptor: media = "
        f"{diag['edad_receptor']['media']:.2f} "
        f"(target {diag['edad_receptor']['target_media']})"
    )
    print(
        f"      Género F: {diag['genero_receptor']['pct_F_realizado']:.4f} "
        f"(target {diag['genero_receptor']['pct_F_target']})"
    )
    print(
        f"      Dependientes media: "
        f"{diag['n_dependientes']['media']:.3f} "
        f"(target λ={diag['n_dependientes']['target_media']})"
    )
    print(
        f"      Antigüedad mediana (meses): "
        f"{diag['antiguedad_recepcion_meses']['mediana']:.1f} "
        f"(target {diag['antiguedad_recepcion_meses']['target_mediana']})"
    )
    print(
        f"      Remesa mediana USD/mes: "
        f"{diag['remesa_mediana_esperada_usd']['mediana']:.2f} "
        f"(target {diag['remesa_mediana_esperada_usd']['target_mediana']})"
    )

    print("\n      Escolaridad (realizado vs target):")
    for k in ESCOLARIDAD_NIVELES:
        r = diag["escolaridad"][k]
        print(
            f"        {k:<16} {r['pct_realizado']:.4f}  "
            f"(target {r['pct_target']:.4f})"
        )

    print("\n      Tipo de vivienda (realizado vs target):")
    for k in VIVIENDA_TIPOS:
        r = diag["tipo_vivienda_actual"][k]
        print(
            f"        {k:<10} {r['pct_realizado']:.4f}  "
            f"(target {r['pct_target']:.4f})"
        )

    print(
        f"\n      Hogares en GDL+Zapopan: "
        f"{diag['pct_hogares_gdl_zapopan']:.4f} del padrón"
    )
    print("\n      Top 10 municipios por número de hogares:")
    for entry in diag["top_10_municipios"]:
        nombre = catalogo.get(entry["cve_municipio"], "?")
        print(
            f"        {entry['cve_municipio']} {nombre:<30} "
            f"{entry['n_hogares']:>5} ({entry['pct_padron']:.4f})"
        )

    print("\nFASE 4 completada.")


if __name__ == "__main__":
    main()
