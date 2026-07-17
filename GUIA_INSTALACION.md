# Bot de Señales TEO — Guía de instalación

Tiempo total: ~30 minutos. No hace falta saber programar, solo seguir los pasos.

---

## Paso 1 — Crear el bot en Telegram (5 min)

1. Abrí Telegram y buscá **@BotFather** (el verificado, con tilde azul).
2. Mandale `/newbot`.
3. Nombre visible: `Señales TEO` (o el que quieras).
4. Username: tiene que terminar en `bot`, ej: `senales_teo_bot`.
5. BotFather te devuelve el **TOKEN** (formato `1234567890:AAE...`). Guardalo, es la llave del bot. No lo compartas ni lo subas a ningún lado público.

## Paso 2 — Obtener tu chat_id (2 min)

1. Abrí un chat con tu bot recién creado y mandale cualquier mensaje ("hola").
2. En el navegador entrá a (reemplazando TOKEN por el tuyo):
   `https://api.telegram.org/botTOKEN/getUpdates`
3. Buscá en la respuesta `"chat":{"id":123456789` — ese número es tu **CHAT_ID**.

## Paso 3 — Probarlo en tu PC (5 min)

Necesitás Python 3.10 o superior instalado (python.org si no lo tenés).

```
# en la carpeta donde descargaste los archivos:
pip install -r requirements.txt

# Windows (PowerShell):
$env:TELEGRAM_BOT_TOKEN="tu_token"
$env:TELEGRAM_CHAT_ID="tu_chat_id"
python bot_senales.py --test

# Linux/Mac:
export TELEGRAM_BOT_TOKEN="tu_token"
export TELEGRAM_CHAT_ID="tu_chat_id"
python bot_senales.py --test
```

Si te llega el mensaje "✅ Bot de señales TEO conectado" a Telegram, está todo bien.
Después probá un escaneo real: `python bot_senales.py --once`
(si ningún par tiene señal en ese momento, no manda nada — es lo esperado).

## Paso 4 — Dejarlo corriendo 24/7

### Opción A: GitHub Actions (GRATIS, recomendada para empezar)

GitHub ejecuta el escaneo cada 15 minutos en sus servidores. Sin VPS, sin PC prendida.

1. Creá una cuenta en github.com si no tenés.
2. Creá un repositorio nuevo → nombre `bot-senales` → **Private**.
   > Nota: en repos privados el plan gratis incluye 2.000 minutos/mes de Actions
   > y este bot usa ~1 min por corrida (~2.900/mes si corre siempre). Opciones:
   > hacé el repo **público** (minutos ilimitados; el código no tiene nada
   > secreto, los tokens van en Secrets) o cambiá el cron a `*/20` (~2.100 min).
3. Subí estos 3 archivos al repo: `bot_senales.py`, `requirements.txt`,
   y `bot.yml` — este último **dentro de la carpeta** `.github/workflows/`
   (creala con "Add file → Create new file" escribiendo
   `.github/workflows/bot.yml` como nombre).
4. En el repo: Settings → Secrets and variables → Actions → "New repository secret":
   - `TELEGRAM_BOT_TOKEN` → tu token
   - `TELEGRAM_CHAT_ID` → tu chat_id
5. Pestaña Actions → workflow "bot-senales" → "Run workflow" para probarlo a mano.
6. Listo. Corre solo cada ~15 min y además va guardando `registro_senales.csv`
   en el repo con el historial de todas las señales (tu base para el post-mortem).

Limitación conocida: GitHub a veces demora los crons 3-10 min en horas pico.
Para señales de 1h con entrada por retest de 15m, no te afecta.

### Opción B: Servidor propio (Railway/VPS, ~USD 5/mes)

Para cuando quieras latencia constante y el resumen diario de las 14:00:

1. Subí el código a GitHub (igual que arriba, sin el bot.yml).
2. En railway.app: New Project → Deploy from GitHub repo.
3. Variables → agregá `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`.
4. Settings → Start Command: `python bot_senales.py` (sin --once: modo loop).
5. El bot escanea cada 15 min y manda el resumen "sesión cerrada" a las 14:00 ART.

---

## Qué hace y qué NO hace

HACE: escanea SOL, ETH, BTC, BNB y XRP en 1h cada 15 min con el sistema completo
(cruce EMA 9/21 + volumen MA-20 + lado del VWAP + vetos RSI + filtro ADX/EMA50 en 1D),
manda captura del chart con niveles de SL/TP calculados, y registra todo en CSV.

NO HACE: no ejecuta trades (deliberado), no verifica el retest de 15m (eso es tuyo),
y no reemplaza tu chequeo del contexto diario. Es tu asistente, no tu reemplazo.

## Ajustes rápidos (editando bot_senales.py)

- Pares: línea `PARES = [...]`
- Umbral ADX: `ADX_MIN = 20`
- Vetos RSI: `RSI_VETO_LONG / RSI_VETO_SHORT`

Regla de oro: cualquier cambio de parámetros cuenta como cambio de sistema.
No se toca hasta los 50 trades registrados.
