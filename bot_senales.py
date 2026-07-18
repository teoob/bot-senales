"""
============================================================
 BOT DE SEÑALES - Sistema TEO (Estrategia P: cruce de OBV)
 Analiza SOL, ETH, BTC, BNB, XRP (futuros perpetuos Binance)
 cada 5 min y manda señales por Telegram con captura del chart.

 Sistema (1h, velas cerradas) - ganador del walk-forward, con
 ventaja positiva confirmada fuera de muestra en los 5 pares:
   - OBV cruza por ENCIMA de su propia EMA-20 -> LONG
   - OBV cruza por DEBAJO de su propia EMA-20 -> SHORT
   - Filtro 1D: precio del lado correcto de la EMA 50 diaria
   - SL: 1.5x ATR(14) | TP1: 1.5x ATR (parcial) | TP2: 2x ATR

 Modos:
   python bot_senales.py --test   -> manda mensaje de prueba
   python bot_senales.py --once   -> un escaneo y termina (GitHub Actions)
   python bot_senales.py          -> loop continuo cada 5 min (Railway/VPS)
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
import numpy as np
from datetime import datetime, timezone, timedelta

# ---------------- CONFIG ----------------
TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

PARES = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT"]

MAKER_FEE_POR_LADO = 0.0002  # 0.02% - arancel maker Binance Futures USDT-M, usuario regular

VENTANA_FRESCA_MIN = 16 # en modo --once: solo alerta si la vela cerró hace <16 min
LOG_FILE   = "registro_senales.csv"
STATE_FILE = "estado.json"

# Espejo publico de datos de Binance, sin restriccion geografica
# (fapi.binance.com bloquea IPs de EEUU con error 451; este endpoint
#  esta documentado por Binance para uso de bots/servicios externos)
BINANCE_DATA = "https://data-api.binance.vision/api/v3/klines"


# ---------------- DATOS ----------------
def traer_velas(par: str, intervalo: str, limite: int = 300) -> pd.DataFrame:
    r = requests.get(BINANCE_DATA, params={
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

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    direccion = np.sign(df["close"].diff()).fillna(0)
    return (direccion * df["volume"]).cumsum()


def analizar_par(par: str) -> dict | None:
    """Devuelve dict con la señal si el sistema (Estrategia P) se alinea, si no None."""
    h1 = traer_velas(par, "1h", 300)
    d1 = traer_velas(par, "1d", 300)
    if len(h1) < 60 or len(d1) < 60:
        return None

    # --- 1h: OBV y su EMA-20 (el gatillo) ---
    h1["ema9"], h1["ema21"] = ema(h1["close"], 9), ema(h1["close"], 21)  # solo para el chart
    h1["obv"] = obv(h1)
    h1["obv_ema"] = ema(h1["obv"], 20)
    h1["atr"] = atr(h1)

    u, p = h1.iloc[-1], h1.iloc[-2]  # última vela cerrada y la anterior

    cruce_arriba = u.obv > u.obv_ema and p.obv <= p.obv_ema
    cruce_abajo  = u.obv < u.obv_ema and p.obv >= p.obv_ema

    # --- filtro 1D: tendencia (unico filtro extra de la Estrategia P) ---
    d1["ema50"] = ema(d1["close"], 50)
    ud = d1.iloc[-1]

    senal_long  = cruce_arriba and ud.close > ud.ema50
    senal_short = cruce_abajo  and ud.close < ud.ema50

    if not (senal_long or senal_short):
        return None

    lado = "LONG" if senal_long else "SHORT"

    # niveles por ATR (SL 1.5x, TP1 1.5x parcial, TP2 2x) - igual que en el backtest
    riesgo = 1.5 * u.atr
    if riesgo <= 0 or riesgo / u.close < 0.0005:
        return None
    if lado == "LONG":
        sl, tp1, tp2 = u.close - riesgo, u.close + riesgo, u.close + 2.0 * u.atr
    else:
        sl, tp1, tp2 = u.close + riesgo, u.close - riesgo, u.close - 2.0 * u.atr
    rr1 = abs(tp1 - u.close) / riesgo
    rr2 = abs(tp2 - u.close) / riesgo

    # --- ROI de cada nivel: % de movimiento de precio hasta ahi,
    # neto de comision (ida + vuelta, tarifa maker) ---
    comision_ida_vuelta_pct = 2 * MAKER_FEE_POR_LADO * 100  # en puntos porcentuales
    signo = 1 if lado == "LONG" else -1
    roi_sl  = signo * (sl  - u.close) / u.close * 100 - comision_ida_vuelta_pct
    roi_tp1 = signo * (tp1 - u.close) / u.close * 100 - comision_ida_vuelta_pct
    roi_tp2 = signo * (tp2 - u.close) / u.close * 100 - comision_ida_vuelta_pct

    return {
        "par": par, "lado": lado, "precio": u.close,
        "obv_x_ema": (u.obv - u.obv_ema) / abs(u.obv_ema) if u.obv_ema != 0 else 0,
        "sl": sl, "tp1": tp1, "tp2": tp2, "rr1": rr1, "rr2": rr2,
        "roi_sl": roi_sl, "roi_tp1": roi_tp1, "roi_tp2": roi_tp2,
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
        mpf.make_addplot(df["Obv"],     panel=2, color="#3498db", width=1.6, ylabel="OBV"),
        mpf.make_addplot(df["Obv_ema"], panel=2, color="#e67e22", width=1),
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
        returnfig=True, figsize=(12, 8), tight_layout=True,
        panel_ratios=(3, 1, 1),
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

def telegram_get(metodo: str, params: dict) -> dict:
    r = requests.get(f"https://api.telegram.org/bot{TOKEN}/{metodo}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------- COMANDOS (consulta manual desde Telegram) ----------------
def responder_estado(comando: str):
    if comando in ("/start", "/ayuda", "/help"):
        enviar_texto(
            "\U0001F916 Bot de se\u00f1ales TEO\n\n"
            "Comandos:\n"
            "/hoy o /estado - estado actual de los 5 pares\n\n"
            "Las se\u00f1ales v\u00e1lidas te llegan solas, con chart, apenas se detectan "
            "(chequeo autom\u00e1tico cada ~5 min)."
        )
        return

    ahora = (datetime.now(timezone.utc) + timedelta(hours=-3)).strftime("%H:%M")
    lineas = [f"\U0001F4CA <b>Estado actual</b> ({ahora} ART)"]
    alguna = False
    for par in PARES:
        try:
            s = analizar_par(par)
        except Exception as ex:
            lineas.append(f"\u26AA {par}: error consultando ({ex})")
            continue
        if s:
            alguna = True
            e = "\U0001F7E2" if s["lado"] == "LONG" else "\U0001F534"
            lineas.append(f"{e} {par}: SE\u00d1AL {s['lado']} (te llega la captura aparte)")
        else:
            lineas.append(f"\u26AA {par}: sin se\u00f1al")
    lineas.append("" )
    lineas.append("Hay se\u00f1al activa \u2192 revis\u00e1 el mensaje con el chart." if alguna
                   else "Ninguno cumple los filtros ahora mismo. Reintento autom\u00e1tico en ~5 min.")
    enviar_texto("\n".join(lineas))


def procesar_comandos(estado: dict) -> dict:
    """Revisa si mandaste /hoy, /estado, etc. desde el ultimo chequeo y responde."""
    offset = estado.get("update_offset", 0)
    try:
        data = telegram_get("getUpdates", {"offset": offset, "timeout": 0})
    except Exception as ex:
        print(f"error consultando comandos: {ex}")
        return estado

    for u in data.get("result", []):
        estado["update_offset"] = u["update_id"] + 1
        msg = u.get("message", {})
        texto = (msg.get("text") or "").strip().lower()
        chat = str(msg.get("chat", {}).get("id", ""))
        if chat != str(CHAT_ID) or not texto.startswith("/"):
            continue
        print(f"comando recibido: {texto}")
        try:
            responder_estado(texto.split()[0])
        except Exception as ex:
            print(f"error respondiendo comando: {ex}")
    return estado

def armar_caption(s: dict) -> str:
    e = "\U0001F7E2" if s["lado"] == "LONG" else "\U0001F534"
    f = lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    fr = lambda x: f"{x:+.2f}".replace(".", ",")
    return (
        f"{e} <b>{s['lado']} {s['par']}</b> | 1h | Estrategia P (OBV)\n"
        f"Precio: {f(s['precio'])}\n"
        f"OBV vs su EMA-20: {s['obv_x_ema']*100:+.1f}%\n"
        f"1D: tendencia confirmada (precio vs EMA50 diaria)\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"SL:  {f(s['sl'])}  \u2192 ROI {fr(s['roi_sl'])}%\n"
        f"TP1: {f(s['tp1'])}  \u2192 ROI {fr(s['roi_tp1'])}% (RR 1:{s['rr1']:.2f}, parcial 50%)\n"
        f"TP2: {f(s['tp2'])}  \u2192 ROI {fr(s['roi_tp2'])}% (RR 1:{s['rr2']:.2f})\n"
        f"<i>ROI = % de movimiento de precio, neto de comisi\u00f3n maker ida+vuelta (0,04%)</i>\n"
        f"Entrada validada en backtest sin retest previo - pod\u00e9s ejecutar directo."
    )


# ---------------- REGISTRO / ESTADO ----------------
def registrar(s: dict):
    nuevo = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["fecha_utc", "par", "lado", "precio", "sl", "tp1", "tp2",
                        "rr1", "rr2", "obv_x_ema", "roi_sl_pct", "roi_tp1_pct", "roi_tp2_pct",
                        "resultado_R"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="minutes"),
                    s["par"], s["lado"], round(s["precio"], 4), round(s["sl"], 4),
                    round(s["tp1"], 4), round(s["tp2"], 4), round(s["rr1"], 2),
                    round(s["rr2"], 2), round(s["obv_x_ema"], 4),
                    round(s["roi_sl"], 3), round(s["roi_tp1"], 3), round(s["roi_tp2"], 3), ""])

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
        enviar_texto("\u2705 Bot de se\u00f1ales TEO conectado. Sistema: Estrategia P (cruce de OBV + filtro de tendencia 1D).\nPares: " + ", ".join(PARES))
        print("Mensaje de prueba enviado.")
        return

    if "--once" in sys.argv:
        estado = cargar_estado()
        estado = procesar_comandos(estado)
        guardar_estado(estado)
        escanear(modo_once=True)
        return

    # loop continuo (Railway/VPS): responde comandos casi al instante,
    # y escanea los pares cada 5 minutos
    print("Bot iniciado en modo loop.")
    ultimo_escaneo = 0.0
    ultimo_resumen_dia = ""
    while True:
        estado = cargar_estado()
        estado = procesar_comandos(estado)
        guardar_estado(estado)

        ahora_ts = time.time()
        if ahora_ts - ultimo_escaneo >= 300:  # 5 minutos
            escanear(modo_once=False)
            ultimo_escaneo = ahora_ts

            ahora = datetime.now(timezone.utc)
            if ahora.hour == 17 and ahora.minute < 5:  # 14:00 ART
                hoy = ahora.date().isoformat()
                if ultimo_resumen_dia != hoy:
                    est = cargar_estado()
                    enviadas_hoy = any(hoy in str(v) for k, v in est.items()
                                        if k not in ("resumen", "update_offset"))
                    if not enviadas_hoy:
                        enviar_texto("\U0001F4CB 14:00 ART - Sesi\u00f3n sin se\u00f1ales v\u00e1lidas. "
                                     "Sesi\u00f3n cerrada seg\u00fan regla. D\u00eda de replay/backtesting.")
                    ultimo_resumen_dia = hoy

        time.sleep(10)  # chequeo de comandos, casi instant\u00e1neo


if __name__ == "__main__":
    main()
