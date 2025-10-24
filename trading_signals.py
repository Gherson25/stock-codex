"""Herramientas para descargar datos, detectar niveles y generar señales de trading.

El módulo implementa un flujo de trabajo completo:
1. Descargar datos de diferentes fuentes.
2. Limpiar y combinar la información.
3. Detectar soportes y resistencias.
4. Generar señales y evaluar su desempeño con un backtesting simple.
5. Graficar los resultados con anotaciones.

Requisitos de terceros:
- pandas
- numpy
- matplotlib
- yfinance
- ta
- requests

Instale los paquetes con `pip install pandas numpy matplotlib yfinance ta requests` antes de ejecutar
este script.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Optional

import logging

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib import pyplot as plt
from ta.volatility import AverageTrueRange


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TradingSignal:
    """Representa una señal de trading detectada sobre una serie temporal."""

    timestamp: pd.Timestamp
    signal_type: str
    price: float
    level: float
    strength: str


def cargar_datos(
    ticker: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
    alpha_vantage_key: Optional[str] = None,
    finnhub_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Descarga datos del ticker solicitado desde múltiples fuentes.

    Parameters
    ----------
    ticker:
        Símbolo del activo.
    start, end:
        Rango de fechas. Si no se proporcionan se usa un año hacia atrás hasta hoy.
    interval:
        Intervalo aceptado por yfinance ("1d", "1h", "15m", etc.).
    alpha_vantage_key, finnhub_key:
        Claves opcionales para proveedores externos. Si se incluyen, se intentará
        descargar datos de ejemplo mediante solicitudes HTTP simuladas.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Diccionario con los dataframes por proveedor.
    """

    if start is None:
        start = datetime.utcnow() - timedelta(days=365)
    if end is None:
        end = datetime.utcnow()

    logging.info("Descargando datos de yfinance para %s", ticker)
    yf_df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if yf_df.empty:
        raise ValueError("No se pudieron descargar datos desde yfinance. Verifique el ticker y el intervalo.")

    data_sources: Dict[str, pd.DataFrame] = {"yfinance": yf_df}

    # Simulación simple de llamadas a proveedores alternativos.
    if alpha_vantage_key:
        logging.info("Simulando llamada a Alpha Vantage para %s", ticker)
        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "apikey": alpha_vantage_key,
                    "datatype": "csv",
                },
                timeout=10,
            )
            response.raise_for_status()
            csv_data = pd.read_csv(StringIO(response.text))
            csv_data.rename(
                columns={
                    "timestamp": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            csv_data["Date"] = pd.to_datetime(csv_data["Date"])
            csv_data.set_index("Date", inplace=True)
            data_sources["alpha_vantage"] = csv_data.sort_index()
        except Exception as exc:  # pragma: no cover - el propósito es resiliencia en producción
            logging.warning("No fue posible simular Alpha Vantage: %s", exc)

    if finnhub_key:
        logging.info("Simulando llamada a Finnhub para %s", ticker)
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": ticker, "token": finnhub_key},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            finnhub_df = pd.DataFrame(
                {
                    "Open": [payload.get("o", np.nan)],
                    "High": [payload.get("h", np.nan)],
                    "Low": [payload.get("l", np.nan)],
                    "Close": [payload.get("c", np.nan)],
                    "Volume": [payload.get("v", np.nan)],
                },
                index=[pd.Timestamp.utcnow()],
            )
            data_sources["finnhub"] = finnhub_df
        except Exception as exc:  # pragma: no cover
            logging.warning("No fue posible simular Finnhub: %s", exc)

    # Simulación adicional de una fuente hipotética "Argus Momentum".
    try:
        response = requests.get("https://httpbin.org/get", params={"symbol": ticker}, timeout=10)
        response.raise_for_status()
        argus_df = yf_df.copy().tail(5)
        argus_df["Momentum"] = argus_df["Close"].pct_change().rolling(3).mean()
        data_sources["argus_momentum"] = argus_df
    except Exception as exc:  # pragma: no cover
        logging.warning("No fue posible simular Argus Momentum: %s", exc)

    return data_sources


def combinar_fuentes(dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Funde múltiples dataframes de OHLCV en un único dataframe limpio."""

    frames = [df.copy() for df in dataframes if not df.empty]
    if not frames:
        raise ValueError("No se proporcionaron dataframes con datos válidos para combinar.")

    # Normalizar columnas esperadas.
    for df in frames:
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

    combined = pd.concat(frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined = combined.ffill().bfill()
    return combined


def _pivot_points(series: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Calcula pivotes (fractales) simples."""

    pivots = pd.Series(False, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left : i + right + 1]
        center = series.iloc[i]
        if center == window.max() and (window == center).sum() == 1:
            pivots.iloc[i] = True
    return pivots


def detectar_soportes_resistencias(
    data: pd.DataFrame, max_levels: int = 3, left: int = 2, right: int = 2
) -> Dict[str, List[float]]:
    """Obtiene niveles clave utilizando pivotes y filtra con ATR."""

    if data.empty:
        raise ValueError("El dataframe de datos está vacío.")

    highs = data["High"].copy()
    lows = data["Low"].copy()
    atr_indicator = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    atr = atr_indicator.average_true_range().fillna(method="bfill")
    atr_mean = float(atr.mean()) if not atr.isna().all() else 0.0

    pivot_highs = _pivot_points(highs, left=left, right=right)
    pivot_lows = _pivot_points(-lows, left=left, right=right)

    resistance_levels: List[float] = []
    support_levels: List[float] = []

    for idx in highs[pivot_highs].sort_values(ascending=False).index:
        level = highs.loc[idx]
        if all(abs(level - existing) > atr_mean for existing in resistance_levels):
            resistance_levels.append(level)
        if len(resistance_levels) >= max_levels:
            break

    for idx in lows[pivot_lows].sort_values().index:
        level = lows.loc[idx]
        if all(abs(level - existing) > atr_mean for existing in support_levels):
            support_levels.append(level)
        if len(support_levels) >= max_levels:
            break

    return {"supports": support_levels, "resistances": resistance_levels}


def generar_senales(data: pd.DataFrame, niveles: Dict[str, List[float]]) -> List[TradingSignal]:
    """Genera señales en función de rupturas o rebotes de los niveles detectados."""

    if data.empty:
        raise ValueError("El dataframe de datos está vacío.")

    signals: List[TradingSignal] = []
    supports = niveles.get("supports", [])
    resistances = niveles.get("resistances", [])

    for idx in range(1, len(data)):
        row = data.iloc[idx]
        prev_row = data.iloc[idx - 1]
        timestamp = data.index[idx]

        for level in supports:
            if prev_row["Close"] >= level and row["Close"] < level:
                strength = "venta fuerte" if row["Volume"] > prev_row["Volume"] else "venta débil"
                signals.append(TradingSignal(timestamp, "ruptura_soporte", row["Close"], level, strength))
            elif prev_row["Low"] > level and row["Low"] <= level:
                strength = "compra fuerte" if row["Volume"] > prev_row["Volume"] else "compra débil"
                signals.append(TradingSignal(timestamp, "rebote_soporte", row["Close"], level, strength))

        for level in resistances:
            if prev_row["Close"] <= level and row["Close"] > level:
                strength = "compra fuerte" if row["Volume"] > prev_row["Volume"] else "compra débil"
                signals.append(TradingSignal(timestamp, "ruptura_resistencia", row["Close"], level, strength))
            elif prev_row["High"] < level and row["High"] >= level:
                strength = "venta fuerte" if row["Volume"] > prev_row["Volume"] else "venta débil"
                signals.append(TradingSignal(timestamp, "rebote_resistencia", row["Close"], level, strength))

    return signals


def backtesting(data: pd.DataFrame, signals: List[TradingSignal], horizon: int = 10) -> float:
    """Evalúa el porcentaje de aciertos de las señales en un horizonte de velas."""

    if not signals:
        return 0.0

    closes = data["Close"].copy()
    success = 0

    for signal in signals:
        try:
            idx = closes.index.get_loc(signal.timestamp)
        except KeyError:
            continue
        future_window = closes.iloc[idx + 1 : idx + 1 + horizon]
        if future_window.empty:
            continue

        entry_price = signal.price
        if "compra" in signal.strength:
            if future_window.max() > entry_price:
                success += 1
        else:
            if future_window.min() < entry_price:
                success += 1

    return success / len(signals)


def graficar_resultados(
    data: pd.DataFrame,
    niveles: Dict[str, List[float]],
    signals: List[TradingSignal],
    ticker: str,
    mostrar: bool = True,
    ruta_guardado: Optional[str] = None,
) -> None:
    """Grafica el precio, niveles y señales utilizando matplotlib."""

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data["Close"], label="Precio cierre", color="black")

    for level in niveles.get("supports", []):
        plt.axhline(level, color="gray", linestyle="--", alpha=0.5, label="Soporte")
    for level in niveles.get("resistances", []):
        plt.axhline(level, color="silver", linestyle=":", alpha=0.7, label="Resistencia")

    for signal in signals:
        color = "green" if "compra" in signal.strength else "red"
        marker = "^" if "compra" in signal.strength else "v"
        plt.scatter(signal.timestamp, signal.price, color=color, marker=marker, s=100, label=signal.signal_type)
        plt.text(
            signal.timestamp,
            signal.price,
            f"{signal.signal_type}\n{signal.strength}",
            fontsize=8,
            color=color,
            ha="left",
            va="bottom" if "compra" in signal.strength else "top",
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.title(f"Señales de trading para {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True, linestyle=":", alpha=0.3)

    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    if mostrar:
        plt.show()
    else:
        plt.close()


def main(ticker: str = "AAPL", interval: str = "1d") -> None:
    """Ejecuta un flujo completo de ejemplo con las funciones del módulo."""

    logging.info("Iniciando flujo para %s", ticker)
    fuentes = cargar_datos(ticker, interval=interval)
    combinado = combinar_fuentes(fuentes.values())
    niveles = detectar_soportes_resistencias(combinado)
    senales = generar_senales(combinado, niveles)
    precision = backtesting(combinado, senales)

    logging.info("Se generaron %d señales con una precisión estimada de %.2f%%", len(senales), precision * 100)
    graficar_resultados(combinado, niveles, senales, ticker, mostrar=False)

    # Bloque comentado para alertas futuras.
    # from alerts import enviar_alerta_correo, enviar_alerta_telegram
    # for signal in senales:
    #     enviar_alerta_correo(signal)
    #     enviar_alerta_telegram(signal)


if __name__ == "__main__":
    main("TSLA", interval="1h")
