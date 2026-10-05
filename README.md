# Bolsa Millonaria 2026 — Monitor, Radar y Bot (Plan Definitivo v5)

Sistema de **alertas y recomendaciones** para el concurso de trii (5-oct a 6-nov de 2026). Te dice si la caída de tu acción es ruido o un
cambio real, cuál es el mejor relevo y si la Regla Maestra recomienda cambiar. **trii no tiene API: el sistema NUNCA envía órdenes.**

> La estrategia anterior (setups A/B/C y `signals.py`) está retirada en [legacy/](legacy/). El análisis de backtest sigue en `run_fase*.py` y `reports/`.

## 1) Qué hace cada pieza

| Pieza | Cuándo corre | Qué hace |
|---|---|---|
| `monitor.py` | cada 15 min en horario de bolsa | Calcula el semáforo de tu activo (y del banco) y **avisa sólo si cambia el color** |
| `radar.py` | 19:30, lunes a jueves | Resumen completo: semáforo, banco top 7, catalizadores de 10 días hábiles, Regla Maestra, recordatorio de la micro-compra de mañana. Es la medición oficial del cierre (decide NEGRO) |
| `bot.py` | siempre encendido | Responde en Telegram: comandos, texto libre ("compré 300 argos a 21500") y botones |
| **`vigia.py`** | **continuo (nube o PC)** | **Todo en un solo proceso:** el bot al instante + noticias de la BVC cada 2 min + monitor cada 15 min + radar 19:30. Es el modo recomendado (sección 6d y 7) |
| `run_estudio_noticias.py` | cuando quieras | Estudio de eventos: qué pasó con la acción después de cada anuncio oficial (alimenta lo que dicen los avisos de noticias) |
| `motor_hoy.py` | cuando quieras | Corre todo el motor con datos en vivo y lo imprime (no toca tu estado) |
| `check_fuentes.py` | cuando quieras | Verifica que cada fuente de datos responda |

### Semáforo
🟢 VERDE normal · 🔵 AZUL cae el mercado, no tu activo · 🟡 AMARILLO caída propia sin noticia/volumen que la confirme · 🔴 ROJO caída propia + noticia negativa +
volumen ≥ 1,5× · ⚫ NEGRO un ROJO que no recuperó el 50 % de la caída al cierre del 2.º día. Si el corte es mañana, ROJO se trata como NEGRO.
**Validación (Fase 3):** en acciones grandes, después de NEGRO la caída *no* continuó en promedio (2015–2026), por eso `regla_maestra.cambio_por_negro` quedó en
`false`: NEGRO se muestra como información y no baja el listón de la Regla Maestra. Detalle y límites: `reports/validacion_semaforo.html`.
Umbrales en `config.yaml` (sección `semaforo`).

### Cómo se le habla al bot
**Sin comandos (lo más cómodo):** escribe como te salga y el bot te muestra lo que entendió.
* `compré 300 argos a 21500` · `compré 20 millones de ecopetrol` · `compré 8 meta` (sin precio usa el de hoy)
* `vendí tesla` · `vendí 100 argos` · `vendí la mitad de meta`
* `voy 3,5` (tu rentabilidad; el corte se conserva) · `el corte está en 8` · `voy 3,5 y el corte está en 8`
* `me equivoqué` deshace el último registro de compra o venta.

No importa si el precio va **en pesos o en dólares** ni si pones **el total en vez del precio por acción**: el bot compara con la cotización de hoy, elige la
lectura que cuadra y lo dice ("tomé 48.500 como precio en PESOS"). Si ninguna cuadra, **no guarda** y pregunta. Reconoce nombres (`argos`, `bancolombia`), los de
trii (`nuco`, `uberco`) y errores de tecleo. Las preguntas y los planes ("¿qué compro?", "voy a comprar…") nunca registran nada.

**Con botones:** `/menu` (Semáforo · Mi cartera · Noticias BVC · Qué comprar · Compré · Vendí · Mi ranking · Cómo voy). Los avisos de noticias traen botones propios.

**Comandos:** `/comprar` (qué acción de la BVC comprar, qué esperar y por cuánto tiempo) · `/noticias [ACCIÓN]` · `/semaforo [ACCIÓN]` · `/cartera` · `/estado` ·
`/detalle [ACCIÓN]` · `/banco` · `/base` · `/catalizadores` · `/informe` · `/actualizar` · `/rank` · `/compra` · `/venta` · `/deshacer` · `/pos` · `/op` · `/ayuda`.
Los mensajes están en lenguaje sencillo; los números técnicos viven en `/detalle`. Titulares en inglés traducidos por MyMemory (≈ 5.000 caracteres/día).
En la **semana 5**, `/rank <mi> <rent_del_#1> [<rent_del_#2>]`: si vas #1 y das el #2, el sistema puede sugerir vender el último día con ventaja ≥ 8 pp.

**Sólo BVC (`modo.solo_bvc`, decisión del 5-oct-2026):** relevos, `/base`, `/comprar` y avisos de noticias trabajan sólo con acciones de la BVC
(`banco.grupos_candidatos: [local]`). Lo que ya tengas de EE. UU. se sigue vigilando (semáforo y cartera) y se puede vender, pero no se recomienda comprar nada de allá.

## 2) Puesta en marcha

```powershell
cd "RUTA\A\LA\CARPETA\estrategia"
python -m pip install yfinance pandas numpy scipy pyarrow pyyaml python-dotenv requests python-telegram-bot vaderSentiment pytest
python -m pytest -q                       # deben pasar todos
copy .env.example .env                     # y completa los 3 valores (ver abajo)
python check_fuentes.py                    # ¿responden las fuentes?
python radar.py --dry-run                  # ver el radar sin enviar
python monitor.py --prueba                 # envía una alerta de prueba a tu Telegram
python bot.py --probar                     # verifica el token del bot
powershell -ExecutionPolicy Bypass -File .\programar_tareas.ps1     # instala las tareas (monitor, radar, bot)
```

`.env` (nunca lo subas a git ni lo compartas; ya está en `.gitignore`):
* `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`: crea el bot con **@BotFather** (`/newbot`), escríbele un mensaje y abre `https://api.telegram.org/bot<TOKEN>/getUpdates` para ver tu `chat id`.
* `FINNHUB_API_KEY`: gratis en https://finnhub.io/register (noticias y fechas de reportes).
* `ALPHAVANTAGE_API_KEY` (opcional): respaldo de noticias, 25 consultas/día.

Programador de tareas (lo hace `programar_tareas.ps1`): monitor cada 15 min de lunes a viernes 08:30–16:00, radar lunes a jueves 19:30 y el bot al iniciar sesión
(`iniciar_bot.bat` lo reinicia si se cae). El monitor no hace nada si el mercado de trii está cerrado o es festivo (12-oct y 2-nov), así que el mismo horario sirve en octubre (hasta 15:00) y noviembre (hasta 16:00).
`-Simular` muestra qué haría y `-Quitar` borra las tareas. Salidas en `alertas.log` y `bot.log`.

## 3) Rutina del concurso
* **Cada noche 19:30:** lee el radar → `/rank <mi_rent> <objetivo>` con tus números de la app → si hay **noche de decisión** y dice CAMBIAR, programa el cambio.
* **Mañana siguiente 08:45 (09:45 desde el 3-nov):** sólo si hay cambio: ejecuta con orden límite y registra con `/pos`.
* **Micro-compra:** el radar te dice si mañana toca (cuenta tus 4 operaciones por semana). Regístrala con `/op`.
* Durante el día no tienes que hacer nada: el monitor te avisa sólo si cambia el color.

## 4) ¿Y si algo sale mal? (respaldos incorporados)

| Qué falla | Qué hace el sistema | Qué haces tú |
|---|---|---|
| Finnhub (noticias) cae o sin clave | Cadena de respaldo: Yahoo RSS → Google News RSS → Alpha Vantage (si hay clave) → caché vencida. Si todas caen: avisa "FUENTE CAÍDA: noticias" y usa un **proxy** (volumen ≥ 2× + hueco de apertura) para presumir mala noticia; los mensajes llevan la etiqueta **SIN DATOS DE NOTICIAS** | Nada; en una alerta roja verifica el titular en Yahoo Finance |
| Acciones locales | Noticias por Google News en español (cobertura parcial, se advierte). Si no hay datos de noticias, **no se recomiendan** como relevo (si eso vaciara el banco, se relaja y quedan marcadas) | — |
| Precios (Yahoo) caen | Usa la última caché; tras 3 corridas seguidas sin datos avisa "FUENTE CAÍDA: precios" (y "recuperada" al volver) | Mira trii a mano |
| IV de opciones no disponible | El MEE usa σ de 20 días en la misma base de tiempo | — |
| Telegram falla o no hay internet | El mensaje se guarda (hasta 20) y se reenvía en la siguiente corrida con la hora original | — |
| `state.json` dañado | Restaura la copia `.bak` (última versión buena) y lo avisa; si ambas se dañan, error claro | Revisa `data/state.json` |
| Monitor, radar y bot escriben a la vez | Candado de escritura: ninguna actualización se pierde | — |
| El monitor o el radar fallan por un error | Te mandan un mensaje de fallo (el radar incluye el respaldo manual de 3 min) | Usa el respaldo manual |
| El bot se cae | `iniciar_bot.bat` lo reinicia en 15 s; el bot tampoco muere por un error de datos (responde con el error) | `/actualizar` |
| Fechas de reportes distintas entre fuentes | Se muestran como "fecha incierta: finnhub 20/10 / yfinance 21/10 ⚠️ verificar" | Confírmala en la web de la empresa |
| **PC apagado** | Las tareas se ejecutan al encenderlo, pero **no hay alertas mientras esté apagado ni el bot responde** | Usa la nube (GitHub Actions, Fase 6) o mira trii a mano |
| Datos con ticks erróneos de Yahoo | Se limpian automáticamente (barras con mechas absurdas o cierres que saltan y revierten) | — |

## 5) Archivos que importan
* [config.yaml](config.yaml): todos los parámetros (umbrales del semáforo, Regla Maestra, cortes, horarios, cachés, respaldos).
* `data/state.json`: tu posición, rentabilidades, cambios y operaciones (se actualiza con los comandos del bot).
* [ASSUMPTIONS.md](ASSUMPTIONS.md): supuestos y limitaciones (puntos 61 en adelante: sistema de monitoreo).
* Backtest y análisis anteriores: `run_fase1.py` … `run_fase6.py`, `reports/informe_backtest.html`.
* Validación del semáforo y de la Regla Maestra: `python run_validacion.py estudio|noticias|liga|bases|informe` → `reports/validacion_semaforo.html`.

## 6) Informe HTML a pedido y uso en grupo
* **Los avisos automáticos (monitor y radar) NO traen archivo.** El archivo HTML con toda la información fundamentada se manda **sólo cuando alguien lo pide**: escribe `/informe`, o agrega la palabra `html` a cualquier consulta (`/estado html`, `/semaforo html`, `/banco html`…). El bot responde con el mensaje corto y, además, el archivo `informe_AAAAMMDD_HHMM.html` (ábrelo con el navegador del celular o del PC; no necesita internet).
* El informe explica cada número del semáforo, los pasos de la Regla Maestra, el banco de relevo completo (y **por qué** se descartó cada acción), noticias con su tono, fechas que vienen, la comparación de base y lo que dice la validación histórica.
* **`/base`** compara **todas** las acciones líquidas de EE. UU. y de la BVC por cuánto se espera que se muevan hasta el 6-nov (50 % opciones + 50 % últimos 60 días) y dice si alguna supera a tu base por el 20 % que se exige para justificar el cambio. Así no se olvidan las demás acciones.
* **Grupo:** agrega el bot al grupo de Telegram; cualquiera del grupo puede consultar (`/estado`, `/informe`…). Los avisos automáticos llegan al chat de `TELEGRAM_CHAT_ID`: si quieres que lleguen al grupo, pon en `.env` el **chat id del grupo** (es un número negativo; lo ves en `getUpdates` después de escribir en el grupo).
* En `config.yaml`, sección `bot`: `responder_a: todos` (cualquier chat) o `solo_mi_chat`; `escritura_solo_mi_chat: true` deja `/rank`, `/pos` y `/op` (que **cambian el seguimiento**) sólo para tu chat aunque los demás puedan consultar; `enfriamiento_pesado_s: 45` evita que alguien repita consultas grandes seguidas.
* Con `responder_a: todos`, **cualquiera que encuentre el bot puede usarlo**, gastar sus consultas de datos y (si no activas `escritura_solo_mi_chat`) cambiar tu `/rank`, `/pos` y `/op`. Si no quieres eso, activa esa opción.

## 6a) Cartera de varias acciones
Si compras más de una acción, registra **cada compra** con `/compra ACCIÓN CANTIDAD PRECIO` (por ejemplo `/compra TSLA 55 372`, `/compra META 8 740,16`, `/compra GRUPOARGOS 500 21.960`). Cada una cuenta como una operación de actividad y **no** gasta un cambio. `/cartera` muestra tus acciones con precios de hoy, rentabilidad y peso; `/venta ACCIÓN [CANTIDAD]` registra ventas; `/pos` queda solo para cuando cambias **toda** tu cartera a una sola acción.
* La **acción principal** (la de mayor monto) es la que vigilan el semáforo, las noticias y la Regla Maestra; el monitor avisa además el **color de cada una de las otras**.
* La Regla Maestra compara los candidatos contra el **movimiento esperado del conjunto** (cartera ponderada, 20 días) y **no te recomienda lo que ya tienes**.
* Aproximaciones: el valor en pesos de las acciones de EE. UU. usa la TRM de Yahoo (la de trii puede diferir) y la rentabilidad del conjunto es aproximada; tu ranking oficial lo da trii con `/rank`. Las noticias y novedades automáticas solo se siguen para la acción principal.

## 6b) Avisos automáticos: qué te avisa el bot por sí solo
Además del radar de las 19:30, el monitor (cada 15 min en horario de trii) avisa **cuando ocurre algo nuevo**; cada tipo recuerda lo que ya avisó y no se repite:
| Novedad | Cuándo avisa | Anti-spam |
|---|---|---|
| 🚦 Cambio de color de tu acción (y relevos que entran/salen de ROJO/NEGRO) | al cambiar | enfriamiento de 30 min; las escaladas pasan |
| 📰 Noticia nueva importante de tu acción | titular de las últimas 6 h claramente malo (o muy bueno) que no se había visto | la 1.ª vez solo toma una "línea base" (no avisa lo ya existente); 20 min entre avisos; máx. 3 titulares |
| ⚖️ La Regla Maestra cambia de opinión | pasa a recomendar cambiar (o deja de hacerlo); requiere tu `/rank` | 60 min; si la alerta de color ya trajo la decisión, no se repite |
| 📊 / 🏦 / 🏁 Fechas | reporte de resultados de tu acción hoy o en la próxima sesión, evento macro del día (p. ej. FED), corte hoy/mañana | una sola vez cada una |
| 📈 Subida fuerte | tu acción sube con z ≥ 2 | una vez al día |
| 🔎 Comparación de base | (en el radar de las 19:30) otra acción supera a la base por ≥ 20 %, o deja de hacerlo | solo cuando cambia |
| ⚠️ Salud de fuentes | una fuente cae o se recupera | 60 min |
Todo se ajusta en `config.yaml → monitor.eventos` (`activo: false` apaga los avisos de novedades; el tope `max_por_corrida` limita cuántos salen juntos). Estos avisos **no** traen archivo; para el detalle escribe `/informe`.

## 6e) Liquidez real en trii (se revisa antes de cada resultado)
trii ejecuta tus órdenes en la Bolsa de Colombia. Las acciones de EE. UU. que muestra (TSLACO, NVDACO, METACO…) se negocian **muy poco** aquí aunque en Nueva York
muevan miles de millones: el 5-oct TSLACO negoció 42 acciones en todo el día. `src/liquidez.py` mide lo que de verdad se negocia en el libro local (mediana de 20
sesiones y días sin negociación) y clasifica: **buena** (≥ $2.000 millones al día), **justa** (≥ $500 millones: sólo con orden límite) o **mala** (mejor no operarla).
* `/comprar` y los relevos sólo incluyen acciones con liquidez buena.
* Al registrar una compra el bot te dice la liquidez de esa acción y qué parte del volumen diario sería tu orden.
* `/cartera` y `/semaforo` te recuerdan lo que tienes y se negocia poco (salir puede costar caro).
* Regla fija: **orden límite siempre, nunca "a mercado"**. Umbrales en `config.yaml → liquidez`.

## 6d) Vigía de noticias de la BVC (cada 2 minutos)
`src/noticias_bvc.py`. Cada 2 minutos, las 24 horas, lee en paralelo (≈ 1 segundo):
1. **Superfinanciera · Información relevante** (lo que cada emisor está obligado a publicar: es la fuente oficial y la más rápida),
2. RSS de Valora Analitik, La República, Portafolio, Semana, El Tiempo y El Colombiano,
3. Google News (una búsqueda agrupada por ronda, rotando).

De cada titular nuevo decide **de qué empresa de la BVC habla** (21 emisores en `config.yaml → noticias_bvc.emisores`; los nombres que son palabras corrientes —Éxito,
Mineros, ISA, PEI— exigen contexto de bolsa; los resúmenes de la jornada se descartan), **qué tipo de noticia es** (OPA, recompra, dividendo, resultados, sanción,
contrato/hallazgo, compra o venta de un negocio, calificación, índice…) y **hacia dónde empuja**. El puntaje (0–1) sale del peso del tipo, la fiabilidad de la
fuente, si varias fuentes la repiten y, sobre todo, **si el precio ya lo confirma** (se movió ≥ 1 vez lo normal en el mismo sentido). Sólo se avisa lo que pasa
`umbral_alto` (0,60). Anti-spam: la primera ronda es sólo línea base, cada titular se avisa una vez, 60 min entre avisos de la misma empresa y sentido, máximo 2 por ronda,
y una mala noticia de una acción que **no** tienes sólo se avisa si es muy fuerte.

**El aviso dice:** qué pasó y quién lo publicó · tipo e impacto · cómo va el precio · **qué hacer** · **qué esperar** (con casos reales medidos) · **cuánto tiempo**
(máximo de sesiones, fecha límite y tope de pérdida) · la BVC con más movimiento esperado · botones (✅ La compré / 🛒 Qué comprar / 💼 Mi cartera).

**Lo que dice la evidencia (léelo: cambia lo que cabría esperar).** `python run_estudio_noticias.py` midió 1.883 anuncios oficiales de 20 emisores (oct-2023 a oct-2026):
* Un anuncio "con tipo" mueve fuerte la acción (≥ 2 veces lo normal) sólo el 11 % de las veces; un día cualquiera con anuncio, el 9 %. Casi ningún tipo es "de alto impacto" por sí solo: por eso el bot exige la confirmación del precio.
* **Después de un día de subida fuerte por un anuncio, la acción NO siguió subiendo:** −0,6 % en 3 sesiones y −1,25 % en 5 (IC95 −2,35 a −0,30) frente al mercado. Perseguir el salto pierde.
* Comprar en la apertura siguiente a un anuncio positivo dejó +0,25 % en 3 sesiones; entrar y salir cuesta ≈ 1,3 %.
* Después de una caída fuerte por un anuncio hubo rebote parcial: +0,9 % en 3 sesiones. No vender por pánico.

Por eso el bot **sólo recomienda COMPRAR si, en los casos parecidos, lo que vino después supera el costo incluso en el extremo bajo del intervalo de confianza**; con los
datos de hoy eso no ocurre, así que el aviso dirá "no la persigas" o "no compres sólo por esta noticia" y te mostrará los números. Si el estudio cambia, la recomendación
cambia sola. Para decidir **qué comprar**, `/comprar` ordena las acciones líquidas de la BVC por el movimiento esperado hasta el próximo corte (lo que el plan usa para
remontar) y aplica la Regla Maestra con tu ranking. Límites: precios de Yahoo con ≈ 15 min de retraso; el estudio usa precios diarios y sólo anuncios oficiales.

## 6c) Dashboard (Streamlit)
`iniciar_app.bat` (o `python -m streamlit run app.py`) abre el tablero en el navegador y se actualiza solo cada 5 minutos. Pestañas: **Hoy** (semáforo grande, métricas, gráfico del día en σ y la decisión), **Relevos** (banco completo y por qué se descartó cada acción), **Calculadora** (escribe tu rentabilidad y la del corte y aplica la misma Regla Maestra del bot), **Noticias** (con tono y traducción), **Fechas** y **Base** (¿hay una acción mejor que la tuya, BVC o EE. UU.?). El botón "Actualizar ahora" borra la caché y recalcula; "Preparar informe completo" descarga el HTML.
**Desde el celular:** en la misma red wifi abre la "Network URL" que muestra la ventana. **En la nube (Streamlit Community Cloud):** sube el repositorio a GitHub, entra a https://share.streamlit.io, elige `app.py` y en *Advanced settings → Secrets* escribe `FINNHUB_API_KEY = "tu_clave"`. La app **no necesita el token de Telegram ni tu posición** (la calculadora trabaja con los números que escribas). Una app pública puede abrirla cualquiera que tenga el enlace: no incluye datos personales, pero compártelo con cuidado y considera restringir el acceso en *Share*.

## 7) Nube: GitHub Actions (cuando el PC está apagado)
**Modo actual: el Vigía** (`.github/workflows/vigia.yml` → `python vigia.py`). Un solo proceso continuo que contesta el chat al instante, revisa noticias de la BVC cada
2 minutos, corre el monitor cada 15 y manda el radar a las 19:30. **No depende del "cron" de GitHub** (en este repositorio los horarios programados nunca se dispararon):
cada eslabón vive 5 h 30 min y, al terminar, guarda el estado y lanza el siguiente.

* **Encender:** pestaña *Actions* → "Vigía…" → *Run workflow*. **Apagar:** abre la ejecución en curso → *Cancel run* (al cancelar no se relanza).
* **Si muere al arrancar** (menos de `vigia.vida_minima_s`) no se relanza, para no entrar en un bucle, y te avisa por Telegram.
* `resucitar.yml` es una red de seguridad con cron (si algún día GitHub lo dispara): vuelve a encender el vigía si la cadena se rompió, salvo que tú lo hayas apagado.
* `monitor.yml`, `radar.yml` y `bot.yml` quedaron **sólo como botón manual** (modo antiguo): úsalos únicamente con el vigía apagado.
* **Un solo lector:** Telegram admite un proceso leyendo el bot. No corras `vigia.py` ni `bot.py` en el PC mientras la nube esté encendida (error 409 y mensajes perdidos).
  En el PC, `python vigia.py` hace lo mismo que la nube (déjalo abierto y apaga el de la nube).

**Consumo (importante).** El vigía corre las 24 horas: ≈ 1.440 minutos de Actions por día.
| Repositorio | Minutos gratis | Alcanza para |
|---|---|---|
| **Privado** (plan gratuito) | 2.000 al mes | menos de 2 días de vigía |
| **Público** | ilimitados | todo el concurso |

Con el repositorio **público** cualquiera puede leer el código y la rama `estado`; por eso el estado se guarda **cifrado**: crea una clave con
`python nube/sincronizar_estado.py clave`, guárdala como secreto `ESTADO_CLAVE` y, antes de hacerlo público, borra la rama `estado` antigua (tiene versiones sin cifrar)
para que se cree de nuevo cifrada. Las claves (Telegram, Finnhub) viven en los *Secrets* y nunca son visibles. GitHub puede limitar procesos que abusen de Actions; un bot
personal es un uso menor, pero no hay garantía de servicio.

**Secretos** (*Settings → Secrets and variables → Actions*): `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY` (opcional),
`ESTADO_CLAVE` (sólo repositorio público). *Settings → Actions → General → Workflow permissions → Read and write permissions.*

Límites reales: Yahoo puede bloquear direcciones de GitHub (el sistema usa su respaldo y avisa); la Superfinanciera podría hacer lo mismo (el vigía sigue con las otras
fuentes y avisa si todas caen); los precios llegan con ≈ 15 min de retraso; en el relevo entre eslabones hay ≈ 1 minuto sin bot (los mensajes que escribas en ese minuto
se contestan al arrancar el siguiente).
