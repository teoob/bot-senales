"""
============================================================
 BOT DE SEÑALES - Detector de Divergencias RSI (Sistema TEO)
 Analiza SOL, ETH, BTC, BNB, XRP (futuros perpetuos Binance)
 cada 5 min y manda por Telegram las divergencias de RSI(14)
 que se van confirmando, con foto del chart de la temporalidad
 correspondiente y aviso de si el sesgo es LONG o SHORT.

 Temporalidades escaneadas: 1m, 1h, 4h, 1d
   - Divergencia ALCISTA de RSI  -> sesgo LONG
   - Divergencia BAJISTA de RSI  -> sesgo SHORT

 NOTA: 1m esta pensado para TESTEAR que el bot funciona end-to-end.
 Para operar en serio conviene mirar 1h/4h/1d, ya que en 1m el
 ruido genera muchas divergencias que no son operables.

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

STATE_FILE = "estado.json"

# --- Config del detector de divergencias RSI ---
DIV_LOG_FILE   = "registro_divergencias.csv"
DIV_TIMEFRAMES = ["1m", "1h", "4h", "1d"]  # orden de escaneo
DIV_RSI_LEN    = 14
DIV_SMA_LEN    = 14
DIV_LB_LEFT    = 5
DIV_LB_RIGHT   = 5     # velas de confirmacion del pivot (retraso inevitable)
DIV_RANGE_MIN  = 5     # separacion minima entre pivots, en barras
DIV_RANGE_MAX  = 60    # separacion maxima entre pivots, en barras

# ventana de frescura por temporalidad, para modo --once (GitHub Actions)
DIV_VENTANA_FRESCA_MIN = {"1m": 6, "1h": 16, "4h": 60, "1d": 180}

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
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_g = ganancia.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_p = perdida.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = avg_g / avg_p
    r = 100 - (100 / (1 + rs))
    return r.fillna(50)


# ---------------- MODULO: DIVERGENCIAS RSI (1M / 1H / 4H / 1D) ----------------
def _pivots(serie: pd.Series, lb_left: int, lb_right: int):
    """
    Pivots locales de una serie (usado sobre el RSI).
    Un pivot en la posicion i recien se puede CONOCER en i+lb_right
    (mismo criterio que ta.pivothigh/pivotlow de Pine: no repinta).
    Devuelve dos arrays booleanos (pivot_high, pivot_low) indexados por posicion.
    """
    n = len(serie)
    vals = serie.values
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for i in range(lb_left, n - lb_right):
        ventana = vals[i - lb_left: i + lb_right + 1]
        centro = vals[i]
        if centro == ventana.max() and (ventana == centro).sum() == 1:
            ph[i] = True
        if centro == ventana.min() and (ventana == centro).sum() == 1:
            pl[i] = True
    return ph, pl


def detectar_divergencia_tf(df: pd.DataFrame) -> dict | None:
    """
    Busca, en la ULTIMA barra donde ya se puede confirmar un pivot
    (posicion len-1-lb_right), si hay una divergencia regular de RSI
    contra el pivot anterior del mismo tipo.

    Devuelve None si no hay divergencia recien confirmada en esa barra,
    o un dict con los datos para armar el mensaje y el chart.
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"], DIV_RSI_LEN)
    n = len(df)
    idx_confirma = n - 1 - DIV_LB_RIGHT  # ultima posicion con pivot ya confirmable
    if idx_confirma < DIV_LB_LEFT + DIV_RANGE_MIN:
        return None

    ph, pl = _pivots(df["rsi"], DIV_LB_LEFT, DIV_LB_RIGHT)

    resultado = None

    # --- pivot low en idx_confirma: buscar divergencia alcista (sesgo LONG) ---
    if pl[idx_confirma]:
        anteriores = [i for i in range(0, idx_confirma) if pl[i]]
        if anteriores:
            prev = anteriores[-1]
            gap = idx_confirma - prev
            if DIV_RANGE_MIN <= gap <= DIV_RANGE_MAX:
                precio_now, precio_prev = df["low"].iloc[idx_confirma], df["low"].iloc[prev]
                rsi_now, rsi_prev = df["rsi"].iloc[idx_confirma], df["rsi"].iloc[prev]
                if precio_now < precio_prev and rsi_now > rsi_prev:
                    resultado = {
                        "tipo": "alcista", "sesgo": "LONG",
                        "idx_a": prev, "idx_b": idx_confirma,
                        "precio_a": precio_prev, "precio_b": precio_now,
                        "rsi_a": rsi_prev, "rsi_b": rsi_now,
                        "vela_confirma": df.iloc[idx_confirma]["close_time"],
                    }

    # --- pivot high en idx_confirma: buscar divergencia bajista (sesgo SHORT) ---
    if resultado is None and ph[idx_confirma]:
        anteriores = [i for i in range(0, idx_confirma) if ph[i]]
        if anteriores:
            prev = anteriores[-1]
            gap = idx_confirma - prev
            if DIV_RANGE_MIN <= gap <= DIV_RANGE_MAX:
                precio_now, precio_prev = df["high"].iloc[idx_confirma], df["high"].iloc[prev]
                rsi_now, rsi_prev = df["rsi"].iloc[idx_confirma], df["rsi"].iloc[prev]
                if precio_now > precio_prev and rsi_now < rsi_prev:
                    resultado = {
                        "tipo": "bajista", "sesgo": "SHORT",
                        "idx_a": prev, "idx_b": idx_confirma,
                        "precio_a": precio_prev, "precio_b": precio_now,
                        "rsi_a": rsi_prev, "rsi_b": rsi_now,
                        "vela_confirma": df.iloc[idx_confirma]["close_time"],
                    }

    if resultado is None:
        return None

    resultado["df"] = df
    return resultado


def analizar_divergencias(par: str) -> list[dict]:
    """Escanea 1m, 1H, 4H y 1D. Devuelve lista de divergencias recien confirmadas."""
    encontradas = []
    for tf in DIV_TIMEFRAMES:
        try:
            df = traer_velas(par, tf, 300)
        except Exception as ex:
            print(f"[{par}][{tf}] error trayendo velas: {ex}")
            continue
        if len(df) < 80:
            continue
        div = detectar_divergencia_tf(df)
        if div:
            div["par"] = par
            div["tf"] = tf
            encontradas.append(div)
    return encontradas


def generar_chart_divergencia(div: dict) -> bytes:
    """Chart con panel de precio (velas) y panel de RSI, marcando los dos
    pivots de la divergencia y la linea que los conecta en ambos paneles."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    df = div["df"].iloc[-120:].copy()
    df = df.reset_index()
    n = len(df)
    # reindexar idx_a / idx_b al recorte de las ultimas 120 velas
    offset = len(div["df"]) - n
    idx_a = div["idx_a"] - offset
    idx_b = div["idx_b"] - offset

    color_up, color_down = "#26a69a", "#ef5350"
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
        facecolor="#131722"
    )
    for ax in (ax_price, ax_rsi):
        ax.set_facecolor("#131722")
        ax.tick_params(colors="#d1d4dc", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2a2e39")

    # --- velas ---
    for i in range(n):
        o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        color = color_up if c >= o else color_down
        ax_price.plot([i, i], [l, h], color=color, linewidth=1)
        ax_price.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-9),
                                      facecolor=color, edgecolor=color))

    # --- RSI ---
    ax_rsi.plot(range(n), df["rsi"], color="#ba68c8", linewidth=1.4, label="RSI 14")
    ax_rsi.axhline(70, color="#ef5350", linewidth=0.7, linestyle="--")
    ax_rsi.axhline(30, color="#26a69a", linewidth=0.7, linestyle="--")
    ax_rsi.set_ylim(0, 100)

    # --- marcar pivots y linea de divergencia ---
    col_div = "#26a69a" if div["tipo"] == "alcista" else "#ef5350"
    precio_puntos = [div["precio_a"], div["precio_b"]]
    rsi_puntos = [div["rsi_a"], div["rsi_b"]]
    xs = [idx_a, idx_b]

    ax_price.plot(xs, precio_puntos, color=col_div, linewidth=2, marker="o", markersize=5)
    ax_rsi.plot(xs, rsi_puntos, color=col_div, linewidth=2, marker="o", markersize=5)

    marca = "▲" if div["tipo"] == "alcista" else "▼"
    ax_price.annotate(f"{marca} {div['sesgo']}",
                       xy=(idx_b, precio_puntos[1]),
                       xytext=(idx_b, precio_puntos[1]),
                       color=col_div, fontsize=11, fontweight="bold",
                       xycoords="data", textcoords="data",
                       ha="right", va="bottom" if div["tipo"] == "alcista" else "top")

    ax_price.set_title(f"{div['par']}  {div['tf'].upper()}  -  Divergencia {div['tipo'].upper()} de RSI  ->  {div['sesgo']}",
                        color="#d1d4dc", fontsize=12, loc="left")
    ax_price.set_xlim(-2, n + 1)
    ax_rsi.set_xlim(-2, n + 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def armar_caption_divergencia(div: dict) -> str:
    emoji_sesgo = "\U0001F7E2" if div["sesgo"] == "LONG" else "\U0001F534"
    flecha = "\U0001F53C" if div["sesgo"] == "LONG" else "\U0001F53D"
    f = lambda x: f"{x:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return (
        f"{emoji_sesgo} <b>{div['par']}</b> | {div['tf'].upper()}\n"
        f"{flecha} <b>Divergencia {div['tipo'].upper()} de RSI -> sesgo {div['sesgo']}</b>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Pivot anterior: precio {f(div['precio_a'])} | RSI {div['rsi_a']:.1f}\n"
        f"Pivot actual:   precio {f(div['precio_b'])} | RSI {div['rsi_b']:.1f}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"<i>Confirmada con {DIV_LB_RIGHT} velas de retraso respecto al pivot real. "
        f"Es contexto, no una senal automatizada: cruzala con POC/VAH/VAL, Fibonacci "
        f"y price action antes de operar.</i>"
    )


def registrar_divergencia(div: dict):
    nuevo = not os.path.exists(DIV_LOG_FILE)
    with open(DIV_LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["fecha_utc", "par", "tf", "tipo", "sesgo", "precio_a", "precio_b",
                        "rsi_a", "rsi_b"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="minutes"),
                    div["par"], div["tf"], div["tipo"], div["sesgo"],
                    round(div["precio_a"], 6), round(div["precio_b"], 6),
                    round(div["rsi_a"], 2), round(div["rsi_b"], 2)])


def escanear_divergencias(estado: dict, modo_once: bool) -> dict:
    ahora = datetime.now(timezone.utc)
    for par in PARES:
        try:
            divs = analizar_divergencias(par)
        except Exception as ex:
            print(f"[{par}] error analizando divergencias: {ex}")
            continue

        for div in divs:
            clave = f"div_{par}_{div['tf']}_{div['vela_confirma'].isoformat()}"
            if estado.get(clave):
                continue  # ya avisada

            if modo_once:
                mins = (ahora - div["vela_confirma"]).total_seconds() / 60
                ventana = DIV_VENTANA_FRESCA_MIN.get(div["tf"], 16)
                if mins > ventana:
                    continue

            print(f"[{par}][{div['tf']}] divergencia {div['tipo']} ({div['sesgo']}) -> enviando")
            try:
                enviar_foto(armar_caption_divergencia(div), generar_chart_divergencia(div))
                registrar_divergencia(div)
                estado[clave] = True
            except Exception as ex:
                print(f"[{par}][{div['tf']}] error enviando divergencia: {ex}")
    return estado


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
            "\U0001F916 Bot de divergencias RSI - TEO\n\n"
            "Comandos:\n"
            "/hoy o /estado - ultimas divergencias detectadas hoy\n\n"
            "Escaneo automatico cada ~5 min en 1m/1h/4h/1d para "
            "SOL, ETH, BTC, BNB y XRP. Te llega el chart con la "
            "divergencia marcada y el sesgo (LONG/SHORT)."
        )
        return

    ahora = (datetime.now(timezone.utc) + timedelta(hours=-3)).strftime("%H:%M")
    lineas = [f"\U0001F4CA <b>Divergencias detectadas hoy</b> ({ahora} ART)"]
    if os.path.exists(DIV_LOG_FILE):
        hoy = datetime.now(timezone.utc).date().isoformat()
        with open(DIV_LOG_FILE) as fh:
            filas = [r for r in csv.DictReader(fh) if r["fecha_utc"].startswith(hoy)]
        if filas:
            for r in filas[-15:]:
                e = "\U0001F7E2" if r["sesgo"] == "LONG" else "\U0001F534"
                lineas.append(f"{e} {r['par']} {r['tf'].upper()} - {r['sesgo']} ({r['fecha_utc'][11:16]} UTC)")
        else:
            lineas.append("Todavia ninguna divergencia detectada hoy.")
    else:
        lineas.append("Todavia no hay registro de divergencias.")
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


# ---------------- ESTADO ----------------
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
    estado = escanear_divergencias(estado, modo_once)
    guardar_estado(estado)


def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en las variables de entorno.")

    if "--test" in sys.argv:
        enviar_texto("\u2705 Bot de divergencias RSI conectado. "
                     "Escaneando 1m/1h/4h/1d en: " + ", ".join(PARES))
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
    while True:
        estado = cargar_estado()
        estado = procesar_comandos(estado)
        guardar_estado(estado)

        ahora_ts = time.time()
        if ahora_ts - ultimo_escaneo >= 300:  # 5 minutos
            escanear(modo_once=False)
            ultimo_escaneo = ahora_ts

        time.sleep(10)  # chequeo de comandos, casi instantaneo


if __name__ == "__main__":
    main()
