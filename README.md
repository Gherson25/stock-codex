# stock-codex

Este repositorio incluye utilidades para trabajar con señales de trading basadas en soportes y resistencias.

## Módulo `trading_signals.py`

El script `trading_signals.py` implementa un flujo completo:

1. Descarga datos de diferentes fuentes (yfinance por defecto, con simulaciones para Alpha Vantage, Finnhub y Argus Momentum).
2. Combina y limpia los dataframes descargados.
3. Detecta niveles de soporte y resistencia usando pivotes y ATR.
4. Genera señales basadas en rupturas y rebotes.
5. Realiza un backtesting simple para estimar precisión.
6. Grafica los resultados con matplotlib.

Ejecute el módulo directamente para ver un ejemplo:

```bash
python trading_signals.py
```

De manera predeterminada se analizan datos horarios de TSLA, pero puede modificar el ticker o intervalo editando la llamada a `main()`.

## Dependencias

Asegúrese de contar con las siguientes librerías de Python:

- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`
- `ta`
- `requests`

Instale todo con:

```bash
pip install pandas numpy matplotlib yfinance ta requests
```

> Nota: El módulo contiene un bloque comentado para futuras integraciones de alertas por correo o Telegram.

## Licencia

Este proyecto se distribuye bajo la misma licencia especificada inicialmente para el repositorio.
