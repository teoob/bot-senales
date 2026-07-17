"""
============================================================
 TEST DE ENVIO DE IMAGENES - NO ES LA ESTRATEGIA REAL
 Cruce simple de EMA 9/21 en 1m, sin ningun otro filtro.
 Objetivo unico: confirmar que el bot manda bien las capturas
 de los graficos por Telegram. Como el 1m cruza seguido, esto
 dispara señales en minutos en vez de esperar horas.

 Uso:
   python bot_test_visual.py --once   -> un chequeo y termina
   python bot_test_visual.py --loop   -> revisa cada 60s (Ctrl+C para cortar)

 Este script es descartable: una vez confirmado que las
 imagenes llegan bien, se borra y se sigue usando solo
 bot_senales.py (el sistema real).
============================================================
"""

import os
import sys
import io
import time
import requests
import pandas as pd
from datetime import datetime, timezone

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PARES   = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT"]
BINANCE_DATA = "https://data-api.binance.vision/api/v3/klines"


def traer_velas(par: str, intervalo: str = "1m", limite: int = 150) -> pd.DataFrame:
    r = requests.get(BINANCE_DATA, params={"symbol": par, "interval": intervalo, "limit": limite}, timeout=15)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "trades", "tbv", "tqv", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    ahora = datetime.now(timezone.utc)
    return df[df["close_time"] <= ahora][["open", "high", "low", "close", "volume", "close_time"]]


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def buscar_cruce(par: str):
    df = traer_velas(par)
    if len(df) < 30:
        return None
    df["ema9"], df["ema21"] = ema(df["close"], 9), ema(df["close"], 21)
    u, p = df.iloc[-1], df.iloc[-2]
    alcista = u.ema9 > u.ema21 and p.ema9 <= p.ema21
    bajista = u.ema9 < u.ema21 and p.ema9 >= p.ema21
    if not (alcista or bajista):
        return None
    return {"par": par, "lado": "LONG" if alcista else "SHORT", "precio": u.close, "df": df}


def generar_chart(s: dict) -> bytes:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    df = s["df"].iloc[-60:].copy()
    df.columns = [c.capitalize() if c != "close_time" else c for c in df.columns]
    ap = [
        mpf.make_addplot(df["Ema9"], color="#e74c3c", width=1),
        mpf.make_addplot(df["Ema21"], color="#f39c12", width=1),
    ]
    estilo = mpf.make_mpf_style(base_mpf_style="nightclouds", gridstyle=":")
    buf = io.BytesIO()
    fig, _ = mpf.plot(df, type="candle", style=estilo, addplot=ap, volume=True,
                       title=f"\n[TEST] {s['par']} 1m - {s['lado']}",
                       returnfig=True, figsize=(12, 7), tight_layout=True)
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def enviar_foto(caption: str, png: bytes):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                       data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                       files={"photo": ("chart.png", png, "image/png")}, timeout=30)
    r.raise_for_status()


def escanear():
    for par in PARES:
        try:
            s = buscar_cruce(par)
        except Exception as ex:
            print(f"[{par}] error: {ex}")
            continue
        if s is None:
            print(f"[{par}] sin cruce en 1m")
            continue
        e = "\U0001F7E2" if s["lado"] == "LONG" else "\U0001F534"
        cap = (f"\U0001F9EA <b>[TEST] {e} {s['lado']} {s['par']}</b> 1m\n"
               f"Precio: {s['precio']:.4f}\n"
               f"Esto es solo una prueba de env\u00edo de imagen, "
               f"no es una se\u00f1al real del sistema.")
        print(f"[{par}] CRUCE {s['lado']} -> enviando imagen de prueba")
        enviar_foto(cap, generar_chart(s))
        return True  # con una imagen enviada ya confirmamos que funciona
    return False


if __name__ == "__main__":
    if not TOKEN or not CHAT_ID:
        sys.exit("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en las variables de entorno.")
    print("=== MODO TEST: cruce simple EMA 9/21 en 1m, sin filtros ===")
    print("Este NO es el sistema real. Es solo para probar el envio de imagenes.")
    # Railway ejecuta esto sin argumentos -> loop continuo, igual que el bot real
    while True:
        try:
            enviado = escanear()
            if enviado:
                print("Imagen de prueba enviada correctamente.")
        except Exception as ex:
            print(f"error en el ciclo de test: {ex}")
        time.sleep(60)
