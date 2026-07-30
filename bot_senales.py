"""
============================================================
 BOT DE SEÑALES - Divergencias RSI + Soportes/Resistencias
 Analiza SOL, ETH, BTC, BNB, XRP (futuros perpetuos Binance)
 en 1H, 4H y 1D, cada 5 min, y manda por Telegram:

  A) Divergencias de RSI(14) confirmadas -> LONG o SHORT,
     con chart de la temporalidad y aviso si coincide con
     un soporte/resistencia (confluencia = señal mas fuerte).
  B) Avisos de proximidad a soporte/resistencia (por su cuenta,
     sin necesidad de que haya divergencia), con el RSI actual
     como contexto.

 Todo se avisa de forma INCREMENTAL: cada condicion dispara su
 propio aviso a medida que se va dando, no se espera a que las
 3 coincidan (divergencia + zona + RSI extremo).

 Todos los horarios que ve el usuario van en hora Argentina (ART,
 UTC-3). Internamente todo se calcula en UTC.

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
ART = timezone(timedelta(hours=-3))  # Argentina, fijo (sin horario de verano)

# --- Config del detector de divergencias RSI ---
DIV_LOG_FILE   = "registro_divergencias.csv"
DIV_TIMEFRAMES = ["1h", "4h", "1d"]  # orden de escaneo (de menor a mayor)
DIV_RSI_LEN    = 14
DIV_LB_LEFT    = 5
DIV_LB_RIGHT   = 5     # velas de confirmacion del pivot (retraso inevitable)
DIV_RANGE_MIN  = 5     # separacion minima entre pivots, en barras
DIV_RANGE_MAX  = 60    # separacion maxima entre pivots, en barras

# ventana de frescura por temporalidad, para modo --once (GitHub Actions)
DIV_VENTANA_FRESCA_MIN = {"1h": 16, "4h": 60, "1d": 180}
# cuantas barras confirmables hacia atras revisar (para no perder confirmaciones
# entre corrida y corrida del bot)
DIV_VENTANA_BARRAS = {"1h": 10, "4h": 6, "1d": 3}

# --- Config de Soportes / Resistencias ---
SR_LOG_FILE     = "registro_niveles.csv"
SR_LB           = 10      # pivot lookback en precio (izq y der)
SR_TOL_PCT      = {"1h": 0.4, "4h": 0.6, "1d": 1.0}   # % para agrupar pivots en la misma zona
SR_MAX_NIVELES  = 6       # niveles mas relevantes por par/TF (por cantidad de toques)
SR_PROXIMIDAD_PCT = {"1h": 0.35, "4h": 0.5, "1d": 0.8}  # % de distancia para avisar "cerca de zona"
SR_COOLDOWN_HORAS = 4     # no repetir el mismo aviso de proximidad antes de este tiempo

# Espejo publico de datos de Binance, sin restriccion geografica
# (fapi.binance.com bloquea IPs de EEUU con error 451; este endpoint
#  esta documentado por Binance para uso de bots/servicios externos)
BINANCE_DATA = "https://data-api.binance.vision/api/v3/klines"


# ---------------- UTILIDADES ----------------
def a_hora_art(dt: datetime) -> str:
    """Convierte un datetime UTC a texto en hora Argentina."""
    return dt.astimezone(ART).strftime("%d/%m %H:%M")


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
    ahora = datetime.now(timezone.utc)
    df = df[df["close_time"] <= ahora]  # solo velas cerradas
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


def _pivots(serie: pd.Series, lb_left: int, lb_right: int):
    """Pivots locales (maximos/minimos) de una serie, con confirmacion
    con retraso de lb_right barras (no repinta)."""
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


# ---------------- MODULO: DIVERGENCIAS RSI ----------------
def detectar_divergencias_tf(df: pd.DataFrame, ventana_reciente: int = 10) -> list[dict]:
    """Divergencias regulares de RSI confirmadas en las ultimas
    `ventana_reciente` barras confirmables (no solo la ultima)."""
    df = df.copy()
    df["rsi"] = rsi(df["close"], DIV_RSI_LEN)
    n = len(df)
    ultima_confirmable = n - 1 - DIV_LB_RIGHT
    if ultima_confirmable < DIV_LB_LEFT + DIV_RANGE_MIN:
        return []

    ph, pl = _pivots(df["rsi"], DIV_LB_LEFT, DIV_LB_RIGHT)
    desde = max(DIV_LB_LEFT, ultima_confirmable - ventana_reciente + 1)
    resultados = []

    for idx_confirma in range(desde, ultima_confirmable + 1):
        if pl[idx_confirma]:
            anteriores = [i for i in range(0, idx_confirma) if pl[i]]
            if anteriores:
                prev = anteriores[-1]
                gap = idx_confirma - prev
                if DIV_RANGE_MIN <= gap <= DIV_RANGE_MAX:
                    precio_now, precio_prev = df["low"].iloc[idx_confirma], df["low"].iloc[prev]
                    rsi_now, rsi_prev = df["rsi"].iloc[idx_confirma], df["rsi"].iloc[prev]
                    if precio_now < precio_prev and rsi_now > rsi_prev:
                        resultados.append({
                            "tipo": "alcista", "sesgo": "LONG",
                            "idx_a": prev, "idx_b": idx_confirma,
                            "precio_a": precio_prev, "precio_b": precio_now,
                            "rsi_a": rsi_prev, "rsi_b": rsi_now,
                            "vela_confirma": df.iloc[idx_confirma]["close_time"],
                            "df": df,
                        })
                        continue

        if ph[idx_confirma]:
            anteriores = [i for i in range(0, idx_confirma) if ph[i]]
            if anteriores:
                prev = anteriores[-1]
                gap = idx_confirma - prev
                if DIV_RANGE_MIN <= gap <= DIV_RANGE_MAX:
                    precio_now, precio_prev = df["high"].iloc[idx_confirma], df["high"].iloc[prev]
                    rsi_now, rsi_prev = df["rsi"].iloc[idx_confirma], df["rsi"].iloc[prev]
                    if precio_now > precio_prev and rsi_now < rsi_prev:
                        resultados.append({
                            "tipo": "bajista", "sesgo": "SHORT",
                            "idx_a": prev, "idx_b": idx_confirma,
                            "precio_a": precio_prev, "precio_b": precio_now,
                            "rsi_a": rsi_prev, "rsi_b": rsi_now,
                            "vela_confirma": df.iloc[idx_confirma]["close_time"],
                            "df": df,
                        })

    return resultados


# ---------------- MODULO: SOPORTES / RESISTENCIAS ----------------
def detectar_niveles(df: pd.DataFrame, tf: str) -> list[dict]:
    """
    Detecta zonas de soporte/resistencia agrupando los pivots de
    precio (high para resistencia, low para soporte) que caen dentro
    de la misma tolerancia %. Devuelve las zonas mas tocadas.
    """
    tol = SR_TOL_PCT.get(tf, 0.5) / 100
    ph, pl = _pivots(df["high"], SR_LB, SR_LB)
    ph_low, pl_low = _pivots(df["low"], SR_LB, SR_LB)  # mismos pivots, distinto precio de referencia

    candidatos_res = [df["high"].iloc[i] for i in range(len(df)) if ph[i]]
    candidatos_sop = [df["low"].iloc[i] for i in range(len(df)) if pl_low[i]]

    def agrupar(precios, tipo):
        zonas = []
        for p in sorted(precios):
            asignado = False
            for z in zonas:
                if abs(p - z["precio"]) / z["precio"] <= tol:
                    z["precio"] = (z["precio"] * z["toques"] + p) / (z["toques"] + 1)
                    z["toques"] += 1
                    asignado = True
                    break
            if not asignado:
                zonas.append({"precio": p, "toques": 1, "tipo": tipo})
        zonas.sort(key=lambda z: z["toques"], reverse=True)
        return zonas[:SR_MAX_NIVELES]

    return agrupar(candidatos_res, "resistencia") + agrupar(candidatos_sop, "soporte")


def revisar_proximidad(par: str, tf: str, df: pd.DataFrame, estado: dict, ahora: datetime) -> dict:
    """Avisa si el precio actual esta cerca de un soporte/resistencia
    de esa temporalidad. Usa cooldown para no repetir el mismo aviso."""
    niveles = detectar_niveles(df, tf)
    precio_actual = df["close"].iloc[-1]
    rsi_actual = rsi(df["close"], DIV_RSI_LEN).iloc[-1]
    umbral = SR_PROXIMIDAD_PCT.get(tf, 0.5) / 100

    for nivel in niveles:
        dist_pct = abs(precio_actual - nivel["precio"]) / precio_actual
        if dist_pct > umbral:
            continue

        clave = f"sr_{par}_{tf}_{nivel['tipo']}_{round(nivel['precio'], 4)}"
        ultimo = estado.get(clave)
        if ultimo:
            horas_desde = (ahora - datetime.fromisoformat(ultimo)).total_seconds() / 3600
            if horas_desde < SR_COOLDOWN_HORAS:
                continue

        contexto_rsi = ""
        if nivel["tipo"] == "resistencia" and rsi_actual >= 60:
            contexto_rsi = " (RSI alto, contexto de agotamiento)"
        elif nivel["tipo"] == "soporte" and rsi_actual <= 40:
            contexto_rsi = " (RSI bajo, contexto de agotamiento)"

        emoji = "\U0001F534" if nivel["tipo"] == "resistencia" else "\U0001F7E2"
        f = lambda x: f"{x:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
        msg = (
            f"{emoji} <b>{par}</b> | {tf.upper()}\n"
            f"Precio acercandose a {nivel['tipo'].upper()} ({nivel['toques']} toques)\n"
            f"Precio actual: {f(precio_actual)} | Nivel: {f(nivel['precio'])} "
            f"(dist. {dist_pct*100:.2f}%)\n"
            f"RSI actual: {rsi_actual:.1f}{contexto_rsi}\n"
            f"\U0001F550 {a_hora_art(ahora)} ART"
        )
        try:
            enviar_texto(msg)
            estado[clave] = ahora.isoformat()
            registrar_nivel(par, tf, nivel, precio_actual, rsi_actual, ahora)
        except Exception as ex:
            print(f"[{par}][{tf}] error enviando aviso de proximidad: {ex}")

    return estado


def registrar_nivel(par, tf, nivel, precio_actual, rsi_actual, ahora):
    nuevo = not os.path.exists(SR_LOG_FILE)
    with open(SR_LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["fecha_utc", "par", "tf", "tipo", "nivel", "precio_actual", "rsi"])
        w.writerow([ahora.isoformat(timespec="minutes"), par, tf, nivel["tipo"],
                    round(nivel["precio"], 6), round(precio_actual, 6), round(rsi_actual, 2)])


def buscar_confluencia(par: str, precio_ref: float, tf_origen: str, estado_niveles: dict) -> str:
    """
    Chequea si precio_ref esta cerca de un soporte/resistencia en la
    misma temporalidad o en una mayor (1h->[1h,4h,1d], 4h->[4h,1d], 1d->[1d]).
    Devuelve un texto para agregar al caption de la divergencia, o "".
    """
    orden = {"1h": 0, "4h": 1, "1d": 2}
    tfs_a_mirar = [tf for tf in DIV_TIMEFRAMES if orden[tf] >= orden[tf_origen]]
    hallazgos = []
    for tf in tfs_a_mirar:
        niveles = estado_niveles.get((par, tf))
        if niveles is None:
            continue
        umbral = SR_PROXIMIDAD_PCT.get(tf, 0.5) / 100
        for nivel in niveles:
            dist_pct = abs(precio_ref - nivel["precio"]) / precio_ref
            if dist_pct <= umbral:
                hallazgos.append(f"{nivel['tipo'].upper()} en {tf.upper()} ({nivel['toques']} toques)")
    if not hallazgos:
        return ""
    return "\n\u26A0\uFE0F <b>Confluencia:</b> " + " | ".join(hallazgos)


# ---------------- CHART ----------------
def generar_chart_divergencia(div: dict) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    df = div["df"].iloc[-120:].copy()
    df = df.reset_index()
    n = len(df)
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

    for i in range(n):
        o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        color = color_up if c >= o else color_down
        ax_price.plot([i, i], [l, h], color=color, linewidth=1)
        ax_price.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-9),
                                      facecolor=color, edgecolor=color))

    ax_rsi.plot(range(n), df["rsi"], color="#ba68c8", linewidth=1.4, label="RSI 14")
    ax_rsi.axhline(70, color="#ef5350", linewidth=0.7, linestyle="--")
    ax_rsi.axhline(30, color="#26a69a", linewidth=0.7, linestyle="--")
    ax_rsi.set_ylim(0, 100)

    col_div = "#26a69a" if div["tipo"] == "alcista" else "#ef5350"
    precio_puntos = [div["precio_a"], div["precio_b"]]
    rsi_puntos = [div["rsi_a"], div["rsi_b"]]
    xs = [idx_a, idx_b]

    ax_price.plot(xs, precio_puntos, color=col_div, linewidth=2, marker="o", markersize=5)
    ax_rsi.plot(xs, rsi_puntos, color=col_div, linewidth=2, marker="o", markersize=5)

    marca = "▲" if div["tipo"] == "alcista" else "▼"
    ax_price.annotate(f"{marca} {div['sesgo']}",
                       xy=(idx_b, precio_puntos[1]), xytext=(idx_b, precio_puntos[1]),
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


def armar_caption_divergencia(div: dict, nota_confluencia: str = "") -> str:
    emoji_sesgo = "\U0001F7E2" if div["sesgo"] == "LONG" else "\U0001F534"
    flecha = "\U0001F53C" if div["sesgo"] == "LONG" else "\U0001F53D"
    f = lambda x: f"{x:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return (
        f"{emoji_sesgo} <b>{div['par']}</b> | {div['tf'].upper()}\n"
        f"{flecha} <b>Divergencia {div['tipo'].upper()} de RSI -> sesgo {div['sesgo']}</b>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Pivot anterior: precio {f(div['precio_a'])} | RSI {div['rsi_a']:.1f}\n"
        f"Pivot actual:   precio {f(div['precio_b'])} | RSI {div['rsi_b']:.1f}\n"
        f"\U0001F550 Confirmada: {a_hora_art(div['vela_confirma'])} ART"
        f"{nota_confluencia}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"<i>Confirmada con {DIV_LB_RIGHT} velas de retraso respecto al pivot real "
        f"(inherente al metodo, no es un error). Cruzala con Fibonacci y price "
        f"action antes de operar.</i>"
    )


def registrar_divergencia(div: dict):
    nuevo = not os.path.exists(DIV_LOG_FILE)
    with open(DIV_LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["fecha_utc", "hora_art", "par", "tf", "tipo", "sesgo",
                        "precio_a", "precio_b", "rsi_a", "rsi_b"])
        w.writerow([div["vela_confirma"].isoformat(), a_hora_art(div["vela_confirma"]),
                    div["par"], div["tf"], div["tipo"], div["sesgo"],
                    round(div["precio_a"], 6), round(div["precio_b"], 6),
                    round(div["rsi_a"], 2), round(div["rsi_b"], 2)])


def escanear_divergencias(estado: dict, modo_once: bool) -> dict:
    ahora = datetime.now(timezone.utc)
    cache_niveles = {}  # (par, tf) -> lista de niveles, para reusar en confluencia y proximidad

    for par in PARES:
        dfs_por_tf = {}
        for tf in DIV_TIMEFRAMES:
            try:
                dfs_por_tf[tf] = traer_velas(par, tf, 300)
            except Exception as ex:
                print(f"[{par}][{tf}] error trayendo velas: {ex}")

        # precalculo niveles de S/R por TF (se reusan para confluencia y proximidad)
        for tf, df in dfs_por_tf.items():
            if len(df) >= 80:
                cache_niveles[(par, tf)] = detectar_niveles(df, tf)

        # --- A) divergencias ---
        for tf, df in dfs_por_tf.items():
            if len(df) < 80:
                continue
            ventana = DIV_VENTANA_BARRAS.get(tf, 10)
            divs = detectar_divergencias_tf(df, ventana_reciente=ventana)
            for div in divs:
                div["par"], div["tf"] = par, tf
                clave = f"div_{par}_{tf}_{div['vela_confirma'].isoformat()}"
                if estado.get(clave):
                    continue
                if modo_once:
                    mins = (ahora - div["vela_confirma"]).total_seconds() / 60
                    if mins > DIV_VENTANA_FRESCA_MIN.get(tf, 16):
                        continue

                nota = buscar_confluencia(par, div["precio_b"], tf, cache_niveles)
                print(f"[{par}][{tf}] divergencia {div['tipo']} ({div['sesgo']}){' + confluencia' if nota else ''} -> enviando")
                try:
                    enviar_foto(armar_caption_divergencia(div, nota), generar_chart_divergencia(div))
                    registrar_divergencia(div)
                    estado[clave] = True
                except Exception as ex:
                    print(f"[{par}][{tf}] error enviando divergencia: {ex}")

        # --- B) proximidad a soporte/resistencia, independiente de la divergencia ---
        for tf, df in dfs_por_tf.items():
            if len(df) < 80:
                continue
            estado = revisar_proximidad(par, tf, df, estado, ahora)

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
            "\U0001F916 Bot de divergencias RSI + Soportes/Resistencias - TEO\n\n"
            "Comandos:\n"
            "/hoy o /estado - ultimas divergencias detectadas hoy (con chart)\n\n"
            "Escaneo automatico cada ~5 min en 1h/4h/1d para "
            "SOL, ETH, BTC, BNB y XRP. Avisa divergencias (con sesgo LONG/SHORT), "
            "proximidad a soporte/resistencia, y confluencias entre ambas."
        )
        return

    ahora_art = datetime.now(timezone.utc).astimezone(ART).strftime("%d/%m %H:%M")
    if not os.path.exists(DIV_LOG_FILE):
        enviar_texto(f"\U0001F4CA <b>Divergencias hoy</b> ({ahora_art} ART)\nTodavia no hay registro.")
        return

    hoy_art = datetime.now(ART).date().isoformat()
    with open(DIV_LOG_FILE) as fh:
        filas = [r for r in csv.DictReader(fh) if r.get("hora_art", "").startswith(hoy_art[8:10] + "/" + hoy_art[5:7])]

    if not filas:
        enviar_texto(f"\U0001F4CA <b>Divergencias hoy</b> ({ahora_art} ART)\nTodavia ninguna divergencia detectada hoy.")
        return

    enviar_texto(f"\U0001F4CA <b>Divergencias hoy</b> ({ahora_art} ART) - {len(filas)} en total, mandando charts:")

    # reconstruye y manda el chart de cada divergencia del dia (ultimas 10 para no saturar)
    for r in filas[-10:]:
        try:
            df = traer_velas(r["par"], r["tf"], 300)
            df["rsi"] = rsi(df["close"], DIV_RSI_LEN)
            vela_confirma = pd.to_datetime(r["fecha_utc"])
            # busca la posicion de esa vela en el df actual
            coincidencias = df.index[df["close_time"] == vela_confirma]
            if len(coincidencias) == 0:
                continue
            idx_b = df.index.get_loc(coincidencias[0])
            # idx_a: la barra donde el precio tenia el valor precio_a guardado (aprox., por cercania)
            precio_a = float(r["precio_a"])
            ventana_busqueda = df.iloc[max(0, idx_b - DIV_RANGE_MAX):idx_b]
            col = "low" if r["tipo"] == "alcista" else "high"
            idx_a = (ventana_busqueda[col] - precio_a).abs().idxmin()
            idx_a = df.index.get_loc(idx_a)

            div = {
                "par": r["par"], "tf": r["tf"], "tipo": r["tipo"], "sesgo": r["sesgo"],
                "idx_a": idx_a, "idx_b": idx_b,
                "precio_a": precio_a, "precio_b": float(r["precio_b"]),
                "rsi_a": float(r["rsi_a"]), "rsi_b": float(r["rsi_b"]),
                "vela_confirma": vela_confirma.tz_localize("UTC") if vela_confirma.tzinfo is None else vela_confirma,
                "df": df,
            }
            enviar_foto(armar_caption_divergencia(div), generar_chart_divergencia(div))
        except Exception as ex:
            print(f"error reconstruyendo chart de {r['par']}/{r['tf']}: {ex}")


def procesar_comandos(estado: dict) -> dict:
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
        enviar_texto("\u2705 Bot conectado. Divergencias RSI + Soportes/Resistencias. "
                     "Escaneando 1h/4h/1d en: " + ", ".join(PARES))
        print("Mensaje de prueba enviado.")
        return

    if "--once" in sys.argv:
        estado = cargar_estado()
        estado = procesar_comandos(estado)
        guardar_estado(estado)
        escanear(modo_once=True)
        return

    print("Bot iniciado en modo loop.")
    ultimo_escaneo = 0.0
    while True:
        estado = cargar_estado()
        estado = procesar_comandos(estado)
        guardar_estado(estado)

        ahora_ts = time.time()
        if ahora_ts - ultimo_escaneo >= 300:
            escanear(modo_once=False)
            ultimo_escaneo = ahora_ts

        time.sleep(10)


if __name__ == "__main__":
    main()
