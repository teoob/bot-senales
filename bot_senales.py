"""
============================================================
 BOT DE SEÑALES - Sistema TEO
 Analiza SOL, ETH, BTC, BNB, XRP (futuros perpetuos Binance)
 cada 15 min y manda señales por Telegram con captura del chart.

 Sistema (1h, velas cerradas):
   - Cruce EMA 9/21 en la dirección del trade
   - Volumen > MA-20 (en la vela del cruce o la siguiente
     si va en la misma dirección)
   - Precio del lado correcto del VWAP (sesión UTC)
   - Veto RSI: no long con RSI>70; no short con RSI<30
     ni con RSI sobre su SMA-14
   - Filtro 1D: ADX(14) > 20 y precio del lado correcto
     de la EMA 50 diaria

 Modos:
   python bot_senales.py --test   -> manda mensaje de prueba
   python bot_senales.py --once   -> un escaneo y termina (GitHub Actions)
   python bot_senales.py          -> loop continuo cada 15 min (VPS/Railway)
============================================================
"""

import os
import sys
import io
import csv
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

PARES = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT"]

ADX_MIN        = 20     # filtro de rango en 1D
RSI_VETO_LONG  = 70
RSI_VETO_SHORT = 30
VENTANA_FRESCA_MIN = 16 # en modo --once: solo alerta si la vela cerró hace <16 min
LOG_FILE   = "registro_senales.csv"
STATE_FILE = "estado.json"

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/klines"


# ---------------- DATOS ----------------
def traer_velas(par: str, intervalo: str, limite: int = 300) -> pd.DataFrame:
    r = requests.get(BINANCE_FAPI, params={
        "symbol": par, "interval": intervalo, "limit": limite
    }, timeout=15)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "trades", "tbv", "tqv", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    # descartar la vela en curso: solo velas cerradas
    ahora = datetime.now(timezone.utc)
    df = df[df["close_time"] <= ahora]
    return df[["open", "high", "low", "close", "volume", "close_time"]]


# ---------------- INDICADORES ----------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm  = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def vwap_sesion(df: pd.DataFrame):
    """VWAP anclado a la sesión UTC + bandas de 1 y 2 desvíos (ponderados)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    fecha = df.index.date
    pv  = (tp * df["volume"]).groupby(fecha).cumsum()
    vv  = df["volume"].groupby(fecha).cumsum().replace(0, 1e-10)
    vwap = pv / vv
    var = ((tp - vwap) ** 2 * df["volume"]).groupby(fecha).cumsum() / vv
    sd = var ** 0.5
    return vwap, sd


def analizar_par(par: str) -> dict | None:
    """Devuelve dict con la señal si el sistema completo se alinea, si no None."""
    h1 = traer_velas(par, "1h", 300)
    d1 = traer_velas(par, "1d", 300)
    if len(h1) < 60 or len(d1) < 60:
        return None

    # --- 1h ---
    h1["ema9"], h1["ema21"] = ema(h1["close"], 9), ema(h1["close"], 21)
    h1["volma"] = h1["volume"].rolling(20).mean()
    h1["rsi"] = rsi(h1["close"]); h1["rsisma"] = h1["rsi"].rolling(14).mean()
    h1["vwap"], h1["sd"] = vwap_sesion(h1)

    u, p = h1.iloc[-1], h1.iloc[-2]  # última cerrada y anterior

    cruce_arr_u = u.ema9 > u.ema21 and p.ema9 <= p.ema21
    cruce_arr_p = p.ema9 > p.ema21 and h1.iloc[-3].ema9 <= h1.iloc[-3].ema21
    cruce_abj_u = u.ema9 < u.ema21 and p.ema9 >= p.ema21
    cruce_abj_p = p.ema9 < p.ema21 and h1.iloc[-3].ema9 >= h1.iloc[-3].ema21

    vol_ok = u.volume > u.volma
    verde, roja = u.close > u.open, u.close < u.open

    # cruce en la última vela con volumen, o en la anterior con
    # confirmación de volumen posterior en la misma dirección
    senal_long_1h  = (cruce_arr_u and vol_ok) or (cruce_arr_p and vol_ok and verde)
    senal_short_1h = (cruce_abj_u and vol_ok) or (cruce_abj_p and vol_ok and roja)

    # VWAP como juez de lado
    senal_long_1h  = senal_long_1h  and u.close > u.vwap
    senal_short_1h = senal_short_1h and u.close < u.vwap

    # vetos RSI
    if u.rsi > RSI_VETO_LONG:
        senal_long_1h = False
    if u.rsi < RSI_VETO_SHORT or u.rsi > u.rsisma:
        senal_short_1h = False

    if not (senal_long_1h or senal_short_1h):
        return None

    # --- filtro 1D ---
    d1["ema50"], d1["ema200"] = ema(d1["close"], 50), ema(d1["close"], 200)
    d1["adx"] = adx(d1)
    ud = d1.iloc[-1]
    if ud.adx < ADX_MIN:
        return None
    if senal_long_1h and ud.close < ud.ema50:
        return None
    if senal_short_1h and ud.close > ud.ema50:
        return None

    lado = "LONG" if senal_long_1h else "SHORT"
    regimen = "alcista" if ud.ema50 > ud.ema200 else "bajista"

    # niveles sugeridos (bandas VWAP + swing de 12 velas)
    if lado == "LONG":
        sl  = min(u.vwap - 2 * u.sd, h1["low"].iloc[-12:].min())
        tp1, tp2 = u.vwap + u.sd, u.vwap + 2 * u.sd
    else:
        sl  = max(u.vwap + 2 * u.sd, h1["high"].iloc[-12:].max())
        tp1, tp2 = u.vwap - u.sd, u.vwap - 2 * u.sd

    riesgo = abs(u.close - sl)
    # si el precio ya superó las bandas, usar targets por múltiplos de R
    margen = riesgo * 0.5
    if lado == "LONG" and tp1 < u.close + margen:
        tp1, tp2 = u.close + 1.5 * riesgo, u.close + 2.0 * riesgo
    elif lado == "SHORT" and tp1 > u.close - margen:
        tp1, tp2 = u.close - 1.5 * riesgo, u.close - 2.0 * riesgo
    rr1 = abs(tp1 - u.close) / riesgo if riesgo > 0 else 0
    rr2 = abs(tp2 - u.close) / riesgo if riesgo > 0 else 0

    return {
        "par": par, "lado": lado, "precio": u.close,
        "vwap": u.vwap, "rsi": u.rsi, "vol_x": u.volume / u.volma,
        "adx_1d": ud.adx, "regimen": regimen,
        "sl": sl, "tp1": tp1, "tp2": tp2, "rr1": rr1, "rr2": rr2,
        "cierre_vela": h1.iloc[-1]["close_time"], "df": h1,
    }


# ---------------- CHART ----------------
def generar_chart(s: dict) -> bytes:
    import mplfinance as mpf
    df = s["df"].iloc[-100:].copy()
    df.columns = [c.capitalize() for c in df.columns]

    ap = [
        mpf.make_addplot(df["Ema9"],  color="#e74c3c", width=1),
        mpf.make_addplot(df["Ema21"], color="#f39c12", width=1),
        mpf.make_addplot(df["Vwap"],  color="#3498db", width=1.6),
        mpf.make_addplot(df["Vwap"] + df["Sd"],     color="#2ecc71", width=0.7, linestyle="--"),
        mpf.make_addplot(df["Vwap"] - df["Sd"],     color="#2ecc71", width=0.7, linestyle="--"),
        mpf.make_addplot(df["Vwap"] + 2 * df["Sd"], color="#e67e22", width=0.7, linestyle="--"),
        mpf.make_addplot(df["Vwap"] - 2 * df["Sd"], color="#e67e22", width=0.7, linestyle="--"),
    ]
    # marcador de señal en la última vela
    marca = pd.Series(float("nan"), index=df.index)
    if s["lado"] == "LONG":
        marca.iloc[-1] = df["Low"].iloc[-1] * 0.998
        ap.append(mpf.make_addplot(marca, type="scatter", marker="^", markersize=140, color="#2ecc71"))
    else:
        marca.iloc[-1] = df["High"].iloc[-1] * 1.002
        ap.append(mpf.make_addplot(marca, type="scatter", marker="v", markersize=140, color="#e74c3c"))

    estilo = mpf.make_mpf_style(base_mpf_style="nightclouds", gridstyle=":")
    buf = io.BytesIO()
    fig, _ = mpf.plot(
        df, type="candle", style=estilo, addplot=ap, volume=True,
        title=f"\n{s['par']} 1h - {s['lado']}",
        returnfig=True, figsize=(12, 7), tight_layout=True,
    )
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------- TELEGRAM ----------------
def enviar_foto(caption: str, png: bytes):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
        data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", png, "image/png")}, timeout=30)
    r.raise_for_status()

def enviar_texto(txt: str):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": txt, "parse_mode": "HTML"}, timeout=15)
    r.raise_for_status()

def armar_caption(s: dict) -> str:
    e = "\U0001F7E2" if s["lado"] == "LONG" else "\U0001F534"
    f = lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return (
        f"{e} <b>{s['lado']} {s['par']}</b> | 1h\n"
        f"Precio: {f(s['precio'])} | VWAP: {f(s['vwap'])}\n"
        f"Vol: {s['vol_x']:.1f}x MA-20 | RSI: {s['rsi']:.0f}\n"
        f"1D: ADX {s['adx_1d']:.0f} \u2713 | r\u00e9gimen {s['regimen']} (EMA 200/50)\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"SL sugerido: {f(s['sl'])}\n"
        f"TP1: {f(s['tp1'])} (RR 1:{s['rr1']:.1f}) | TP2: {f(s['tp2'])} (RR 1:{s['rr2']:.1f})\n"
        f"\u26A0\uFE0F Esperar retest en 15m antes de entrar.\n"
        f"Sin retest en 4 velas de 15m = trade cancelado."
    )


# ---------------- REGISTRO / ESTADO ----------------
def registrar(s: dict):
    nuevo = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["fecha_utc", "par", "lado", "precio", "sl", "tp1", "tp2",
                        "rr1", "rr2", "vol_x", "rsi", "adx_1d", "regimen_1d",
                        "hubo_retest", "resultado_R"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="minutes"),
                    s["par"], s["lado"], round(s["precio"], 4), round(s["sl"], 4),
                    round(s["tp1"], 4), round(s["tp2"], 4), round(s["rr1"], 2),
                    round(s["rr2"], 2), round(s["vol_x"], 2), round(s["rsi"], 1),
                    round(s["adx_1d"], 1), s["regimen"], "", ""])

def cargar_estado() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as fh:
            return json.load(fh)
    return {}

def guardar_estado(e: dict):
    with open(STATE_FILE, "w") as fh:
        json.dump(e, fh)


# ---------------- CICLO ----------------
def escanear(modo_once: bool):
    estado = cargar_estado()
    ahora = datetime.now(timezone.utc)
    for par in PARES:
        try:
            s = analizar_par(par)
        except Exception as ex:
            print(f"[{par}] error: {ex}")
            continue
        if s is None:
            print(f"[{par}] sin se\u00f1al")
            continue

        clave = f"{par}_{s['cierre_vela'].isoformat()}"
        if estado.get(par) == clave:
            print(f"[{par}] se\u00f1al ya enviada")
            continue
        # en modo --once (sin estado persistente) solo alertar velas frescas
        if modo_once:
            mins = (ahora - s["cierre_vela"]).total_seconds() / 60
            if mins > VENTANA_FRESCA_MIN:
                print(f"[{par}] se\u00f1al vieja ({mins:.0f} min), se omite")
                continue

        print(f"[{par}] SE\u00d1AL {s['lado']} -> enviando")
        try:
            enviar_foto(armar_caption(s), generar_chart(s))
            registrar(s)
            estado[par] = clave
        except Exception as ex:
            print(f"[{par}] error enviando: {ex}")
    guardar_estado(estado)


def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en las variables de entorno.")

    if "--test" in sys.argv:
        enviar_texto("\u2705 Bot de se\u00f1ales TEO conectado. Sistema: EMA 9/21 + Vol MA-20 + VWAP + veto RSI + filtro ADX 1D.\nPares: " + ", ".join(PARES))
        print("Mensaje de prueba enviado.")
        return

    if "--once" in sys.argv:
        escanear(modo_once=True)
        return

    # loop continuo: escanea a los minutos :01, :16, :31, :46
    print("Bot iniciado en modo loop.")
    while True:
        escanear(modo_once=False)
        ahora = datetime.now(timezone.utc)
        # resumen diario 17:00 UTC = 14:00 Argentina
        if ahora.hour == 17 and ahora.minute < 15:
            hoy = ahora.date().isoformat()
            est = cargar_estado()
            if est.get("resumen") != hoy:
                enviadas_hoy = any(hoy in v for k, v in est.items() if k != "resumen")
                if not enviadas_hoy:
                    enviar_texto("\U0001F4CB 14:00 ART - Sesi\u00f3n sin se\u00f1ales v\u00e1lidas. Sesi\u00f3n cerrada seg\u00fan regla. D\u00eda de replay/backtesting.")
                est["resumen"] = hoy
                guardar_estado(est)
        faltan = 900 - (ahora.minute % 15) * 60 - ahora.second + 60
        time.sleep(max(60, faltan))


if __name__ == "__main__":
    main()
