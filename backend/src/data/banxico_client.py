"""
Cliente para la API REST del Sistema de Información Económica (SIE) de Banxico.

Encapsula la autenticación por token, la construcción de URLs, las llamadas
HTTP y el parseo de la respuesta JSON a un formato tabular consumible por
pandas. Diseñado para ser reutilizado por todos los scripts del proyecto
que necesiten descargar series del SIE.

Documentación oficial de la API:
    https://www.banxico.org.mx/SieAPIRest/service/v1/doc/
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


class BanxicoSIEClient:
    """
    Cliente ligero para consultar series de tiempo del SIE de Banxico.

    Parameters
    ----------
    token : str, optional
        Token de acceso a la API SIE. Si no se proporciona, se lee de la
        variable de entorno BANXICO_TOKEN (cargada vía python-dotenv).
    timeout : int, optional
        Tiempo máximo de espera por petición HTTP, en segundos.

    Raises
    ------
    ValueError
        Si no se encuentra el token ni como argumento ni en el entorno.
    """

    BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"

    def __init__(self, token: Optional[str] = None, timeout: int = 30) -> None:
        if token is None:
            load_dotenv()
            token = os.getenv("BANXICO_TOKEN")
        if not token:
            raise ValueError(
                "No se encontró el token de Banxico. Defínelo en .env como "
                "BANXICO_TOKEN=... o pásalo explícitamente al constructor."
            )
        self.token = token
        self.timeout = timeout

    def _build_url(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> str:
        """
        Construye la URL del endpoint de datos de una serie en un rango.

        Parameters
        ----------
        series_id : str
            Identificador de la serie en SIE (ej. 'SE27803').
        start_date : str
            Fecha inicial en formato 'YYYY-MM-DD'.
        end_date : str
            Fecha final en formato 'YYYY-MM-DD'.

        Returns
        -------
        str
            URL completa para la consulta.
        """
        return (
            f"{self.BASE_URL}/{series_id}/datos/{start_date}/{end_date}"
        )

    def fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Descarga una serie de tiempo y la devuelve como DataFrame ordenado.

        El parseo convierte el campo `fecha` (formato 'dd/mm/yyyy' devuelto
        por el SIE) a `datetime64[ns]` y el campo `dato` a `float64`. Los
        valores marcados como 'N/E' (no existe) por Banxico se convierten
        a NaN.

        Parameters
        ----------
        series_id : str
            Identificador de la serie en SIE (ej. 'SE27803').
        start_date : str
            Fecha inicial en formato 'YYYY-MM-DD'.
        end_date : str
            Fecha final en formato 'YYYY-MM-DD'.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas: fecha (datetime), valor (float),
            ordenado cronológicamente ascendente.

        Raises
        ------
        requests.HTTPError
            Si la respuesta HTTP no es 200.
        ValueError
            Si la respuesta no contiene datos en la estructura esperada.
        """
        url = self._build_url(series_id, start_date, end_date)
        headers = {"Bmx-Token": self.token}

        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        payload = response.json()

        # Estructura esperada: {"bmx": {"series": [{"idSerie": ..., "datos": [...]}]}}
        try:
            datos = payload["bmx"]["series"][0]["datos"]
        except (KeyError, IndexError) as err:
            raise ValueError(
                f"Respuesta inesperada de Banxico para la serie {series_id}: "
                f"{payload}"
            ) from err

        df = pd.DataFrame(datos)
        # Banxico devuelve fechas como 'dd/mm/yyyy' y valores como string
        # con coma como separador de miles (ej. "4,573.2226"). Eliminamos
        # la coma antes de convertir a float; los valores no disponibles
        # vienen marcados como 'N/E' y se convierten a NaN.
        df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
        df["dato"] = (
            df["dato"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("N/E", pd.NA)
        )
        df["dato"] = pd.to_numeric(df["dato"], errors="coerce")
        df = df.rename(columns={"dato": "valor"})
        df = df.sort_values("fecha").reset_index(drop=True)
        return df[["fecha", "valor"]]

    def get_consultation_timestamp(self) -> str:
        """
        Devuelve la marca temporal actual en formato ISO para metadatos.

        Returns
        -------
        str
            Fecha y hora de consulta en formato ISO 8601.
        """
        return datetime.now().isoformat(timespec="seconds")
