"""Vigía de MERCADO: noticias macro y financieras de las bolsas, leídas de las fuentes que el usuario pidió, con qué acciones y sectores se benefician o se
afectan y por qué. Complementa a `noticias_bvc` (noticias propias de cada empresa colombiana) con dos cosas que ese módulo no veía:

  · noticias de TEMA (petróleo, tasas de la Fed, inflación, dólar, Brasil, Medio Oriente, inteligencia artificial, bancos de EE. UU.…), y
  · noticias de las empresas GLOBALES que se negocian en la Bolsa de Colombia (Nubank, Petrobras, Tesla, Nvidia…).

Fuentes (config `noticias_mercado`): Bloomberg Línea (RSS y la página "Las noticias del día"), Bloomberg (canales públicos de mercados, economía, política,
tecnología e industrias), Valora Analitik, Investing.com (canales RSS), Yahoo Finanzas (portada y titulares por acción), La República y la Superfinanciera.
bloomberg.com/latinamerica e investing.com/news rechazan a los programas (error 403): no se fuerzan; se usan sus canales RSS públicos.

Cada tema es conocimiento del negocio escrito a mano: a qué sector le pega, en qué sentido y POR QUÉ. No es una predicción: dice a quién expone la noticia.
La dirección (sube/baja, escalada/calma) se lee del titular; si el titular no la dice, se avisa así y se listan los expuestos sin signo."""
from __future__ import annotations

import datetime as dt
import html as _html
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import noticias_bvc as N

UTC = dt.timezone.utc

# ------------------------------------------------------------------ dirección genérica (titulares en español e inglés, ya sin tildes y en minúsculas)
SUBE = (r"\b(sube|suben|subio|subieron|subira|alza|al alza|repunt\w+|dispar\w+|escala|escalan|trep\w+|avanz\w+|se acelera|acelero|aceler\w+|supera|superan|supero|record|maximo|maximos|"
        r"rall\w+|jump\w*|surge\w*|soar\w*|rise\w*|rose|rising|gain\w*|climb\w*|spike\w*|higher|highs?|boost\w*)\b")
BAJA = (r"\b(baja|bajan|bajo|bajaron|cae|caen|cayo|cayeron|caida|desplom\w+|retroced\w+|hund\w+|se enfria|enfrio|cede|ceden|cedio|a la baja|minimo|minimos|recort\w+|desaceler\w+|"
        r"fren\w+|prohib\w+|restric\w+|limita\w*|pierde fuerza|loses steam|flee|curb\w*|halt\w*|ban|bans|"
        r"fall\w*|fell|drop\w*|slide\w*|slid|slump\w*|plunge\w*|tumble\w*|sink\w*|sank|lower|lows?|decline\w*|ease\w*|cool\w*|weak\w*)\b")

# (signo cuando el tema "sube", sector, acciones, por qué). Las acciones son las del concurso: locales y globales de la Bolsa de Colombia.
T = dict
TEMAS: list[dict[str, Any]] = [
    T(clave="petroleo", nombre="Petróleo", factor="petroleo", que="el precio del petróleo",
      patrones=[r"\bpetroleo\b", r"\bcrudo\b", r"\bbrent\b", r"\bwti\b", r"\bopep\b", r"\bopec\b", r"\bcrude\b",
                r"\boil (price|prices|jump\w*|fall\w*|rise\w*|surge\w*|slide\w*|drop\w*|rall\w+|market|futures|supply|output|demand|climb\w*|slump\w*|soar\w*|tumble\w*|steadies|settles)"],
      efectos=[(+1, "petróleo y gas", ["ECOPETROL", "PBR", "CNEC"], "venden crudo: cada dólar más en el barril es ingreso directo"),
               (-1, "consumo y transporte", ["TERPEL", "UBER", "F", "EXITO", "WMT"], "el combustible más caro sube sus costos y le quita plata al consumidor"),
               (-1, "tecnológicas y bolsa en general", ["CSPX", "EQQQ"], "petróleo caro alimenta la inflación y aleja las bajas de tasas")]),
    T(clave="tasas_eeuu", nombre="Tasas de interés en EE. UU. (Fed)", factor="nasdaq", que="las tasas de interés en EE. UU.", signo_factor=-1,
      patrones=[r"\bfed\b", r"reserva federal", r"federal reserve", r"\bpowell\b", r"\bfomc\b", r"treasury yields?", r"bond yields?", r"rendimiento\w* de(l| los) (tesoro|bonos)", r"bonos del tesoro", r"mercado de bonos"],
      arriba=r"(sub\w+ (de |las )?tasas|alza de tasas|hike|hawkish|higher for longer|endurec\w+|rendimientos? (suben|al alza|en maximos)|yields? (rise|jump|surge|climb|spike)|venta masiva de bonos|bond (rout|selloff|squeeze)|caida del mercado de bonos|bonos (caen|se desploman|se hunden))",
      abajo=r"(recort\w+ (de |las )?tasas|baj\w+ (de |las )?tasas|rate cut|cuts? rates|dovish|flexibiliz\w+|rendimientos? (bajan|caen|ceden)|yields? (fall|drop|ease|slide|retreat))",
      efectos=[(-1, "tecnológicas y de crecimiento", ["NVDA", "MSFT", "META", "AMZN", "GOOGL", "AAPL", "TSLA", "NUCO", "UBER", "EQQQ"], "valen por ganancias futuras: con tasas altas esas ganancias valen menos hoy"),
               (+1, "bancos de EE. UU.", ["JPM", "BAC", "C"], "cobran más por prestar; ganan margen con tasas altas"),
               (-1, "oro", ["MINEROS"], "el oro no paga intereses: pierde atractivo cuando los bonos rinden más"),
               (-1, "bolsa colombiana y peso", ["PFBCOLOM", "BCOLOMBIA", "ECOPETROL", "ISA", "GEB"], "con tasas altas en EE. UU. la plata sale de países emergentes y sube el dólar")]),
    T(clave="inflacion_eeuu", nombre="Inflación y empleo en EE. UU.", factor="nasdaq", que="la inflación en EE. UU.", signo_factor=-1,
      patrones=[r"\bcpi\b", r"\bipc\b.*(ee\.? ?uu|estados unidos)", r"inflacion (en|de) (ee\.? ?uu|estados unidos)", r"us inflation", r"\bpce\b", r"nominas no agricolas", r"nonfarm", r"payrolls", r"jobless claims", r"subsidio\w* por desempleo"],
      efectos=[(-1, "tecnológicas y de crecimiento", ["NVDA", "MSFT", "META", "AMZN", "GOOGL", "AAPL", "TSLA", "NUCO", "EQQQ", "CSPX"], "inflación alta obliga a la Fed a mantener tasas altas, y eso les baja el valor"),
               (+1, "bancos de EE. UU.", ["JPM", "BAC", "C"], "tasas altas por más tiempo les sostienen el margen"),
               (-1, "bolsa colombiana y peso", ["PFBCOLOM", "BCOLOMBIA", "ECOPETROL"], "dólar fuerte y menos apetito por países emergentes")]),
    T(clave="tasas_colombia", nombre="Tasas e inflación en Colombia", factor="colombia", que="las tasas o la inflación en Colombia", signo_factor=-1,
      patrones=[r"banco de la republica", r"\bbanrep\b", r"junta del emisor", r"inflacion (en|de) colombia", r"\bdane\b.*(inflacion|ipc)", r"(inflacion|ipc).*\b(dane|colombia)\b", r"tasa de intervencion"],
      arriba=r"(sub\w+ (de |las )?tasas|alza de tasas|inflacion (sube|subio|se acelera|acelero|repunta|no encuentra techo|volvio a acelerarse)|aceler\w+)",
      abajo=r"(recort\w+ (de |las )?tasas|baj\w+ (de |las )?tasas|inflacion (baja|bajo|cede|cedio|se desacelera|desacelero))",
      efectos=[(-1, "servicios públicos y empresas con deuda", ["CELSIA", "GEB", "ISA", "PROMIGAS", "GRUPOARGOS", "PFGRUPOARG"], "tienen mucha deuda: con tasas altas pagan más intereses"),
               (-1, "construcción e inmobiliario", ["CEMARGOS", "PFCEMARGOS", "PEI"], "crédito caro frena la vivienda y las obras"),
               (-1, "consumo", ["EXITO", "NUTRESA"], "la inflación y el crédito caro le quitan poder de compra a los hogares"),
               (0, "bancos", ["PFBCOLOM", "BCOLOMBIA", "BOGOTA", "AVAL", "PFAVAL", "BHI", "CORFICOLCF"], "efecto mixto: ganan margen, pero prestan menos y sube la cartera en mora")]),
    T(clave="dolar", nombre="Dólar en Colombia", factor="dolar", que="el dólar frente al peso",
      patrones=[r"dolar en colombia", r"\btrm\b", r"peso colombiano", r"dolar (hoy )?(abre|cierra|cerro|se dispara|se desploma|rompe|supera|cae|sube|retrocede|frena)"],
      efectos=[(+1, "exportadoras con ingresos en dólares", ["ECOPETROL", "MINEROS", "CNEC"], "venden en dólares y pagan costos en pesos"),
               (+1, "acciones globales en trii", ["NUCO", "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "PBR"], "se negocian en pesos: si el dólar sube, su precio en pesos sube aunque en Nueva York no se muevan"),
               (-1, "importadoras y consumo", ["EXITO", "TERPEL", "NUTRESA"], "compran afuera en dólares: sus costos suben")]),
    T(clave="oro", nombre="Oro", factor="oro", que="el precio del oro",
      patrones=[r"\boro\b", r"\bgold\b", r"metales preciosos", r"\bbullion\b"],
      efectos=[(+1, "minería de oro", ["MINEROS"], "vende oro: su ingreso sube casi a la par del precio")]),
    T(clave="brasil", nombre="Brasil (elecciones y mercado)", factor="brasil", que="el mercado de Brasil",
      patrones=[r"\bibovespa\b", r"\bbolsonaro\b", r"\blula\b", r"real brasil\w+", r"brazilian real", r"\brunoff\b", r"balotaje",
                r"\b(brasil|brazil\w*)\b.*\b(elecci\w+|election\w*|segunda vuelta|bolsa|mercados?|acciones|stocks|markets?|banco central|central bank|economia|economy|tasas|rates|selic|fiscal|encuestas|polls?)\b",
                r"\b(elecci\w+|election\w*|segunda vuelta|bolsa|mercados?|acciones|stocks|markets?|economia|economy)\b.*\b(brasil|brazil\w*|brasilen\w+)\b"],
      efectos=[(+1, "empresas brasileñas en trii", ["NUCO", "PBR"], "Nubank y Petrobras ganan o pierden con el ánimo del mercado brasileño y con el real")]),
    T(clave="geopolitica", nombre="Tensión geopolítica (Medio Oriente, guerra)", factor="petroleo", que="la tensión geopolítica", peso_abajo=2,
      patrones=[r"medio oriente", r"middle east", r"estrecho de ormuz", r"\bhormuz\b", r"\bguerra\b(?! comercial| arancelaria| de precios)", r"(?<!trade )\bwar\b", r"misil\w*", r"missile\w*",
                r"\b(iran|israel|huties|houthi\w*|hezbola|hezbollah|hamas|rusia|russia|ucrania|ukraine|taiwan|arabia saudita|saudi)\b.*\b(ataque\w*|attack\w*|guerra|war|conflict\w*|misil\w*|sancion\w*|sanctions?|tension\w*|tregua|ceasefire|cese al fuego|paz|peace|nuclear|bombarde\w+)\b",
                r"\b(ataque\w*|attack\w*|sancion\w*|sanctions?|tregua|ceasefire|cese al fuego)\b.*\b(iran|israel|huties|houthi\w*|rusia|russia|ucrania|ukraine|arabia saudita|saudi|riad|riyadh)\b"],
      arriba=r"(ataque\w*|ataca\w*|bombarde\w+|misil\w*|escalada|escala|intercambio de fuego|exchange fire|strike\w*|attack\w*|retalia\w+|represalia\w*|invasion|invade\w*|sancion\w*|sanctions?|tension\w* (sube|aumenta|crece)|temor)",
      abajo=r"(tregua|cese al fuego|ceasefire|acuerdo de paz|peace (deal|talks)|truce|desescalada|de-escalat\w+|close to over|cerca de terminar|no atacara|discusiones productivas|productive (talks|discussions)|pactaron frenar)",
      efectos=[(+1, "petróleo y gas", ["ECOPETROL", "PBR", "CNEC"], "el miedo a que falte crudo sube su precio"),
               (+1, "oro (refugio)", ["MINEROS"], "con miedo, la plata busca refugio en el oro"),
               (-1, "bolsas y acciones de riesgo", ["NUCO", "NVDA", "TSLA", "META", "AMZN", "UBER", "CSPX", "EQQQ"], "en un susto se venden primero las acciones más movidas"),
               (-1, "consumo y transporte", ["TERPEL", "UBER", "F"], "combustible más caro")]),
    T(clave="riesgo_colombia", nombre="Finanzas públicas y política económica de Colombia", factor="colombia", que="la confianza en las finanzas de Colombia",
      patrones=[r"regla fiscal", r"deficit fiscal", r"reforma tributaria", r"ley de financiamiento", r"calificacion (de|crediticia de|soberana de) colombia", r"(fitch|moody'?s|s&p|standard).*colombia", r"\bfmi\b.*colombia",
                r"colombia.*\bfmi\b", r"\bafp\b", r"fondos de pensiones", r"reforma pensional", r"presupuesto general", r"deuda publica", r"minhacienda", r"ministerio de hacienda"],
      arriba=r"(mejora|sube la calificacion|aprob\w+|respaldo|upgrade|perspectiva (positiva|estable)|cumpl\w+|aval\w+|apoyo)",
      abajo=r"(rebaja|baja la calificacion|degrada|downgrade|perspectiva negativa|incumpl\w+|suspende la regla|advierte|advirtio|riesgo|alerta|preocupa\w+|presion\w+|deterior\w+)",
      efectos=[(+1, "bancos y grupos financieros", ["PFBCOLOM", "BCOLOMBIA", "BOGOTA", "AVAL", "PFAVAL", "GRUPOSURA", "PFSURA", "CORFICOLCF"], "son los que más bonos del gobierno tienen y los que más dependen del costo de la deuda del país"),
               (+1, "empresas con dueño estatal", ["ECOPETROL", "ISA", "GEB"], "un gobierno con las cuentas apretadas les pide más dividendos o les cambia las reglas"),
               (+1, "bolsa colombiana en general", ["HCOLSEL", "GRUPOARGOS", "CEMARGOS"], "la confianza en el país mueve a todo el mercado local y al dólar")]),
    T(clave="ia_chips", nombre="Inteligencia artificial y chips", factor="nasdaq", que="el ánimo por la inteligencia artificial",
      patrones=[r"\bchips?\b", r"semiconductor\w*", r"\bopenai\b", r"\btsmc\b", r"\bnvidia\b", r"\basml\b", r"\bai[- ](boom|bubble|trade|rally|stocks?|spending|related|demand|capex)", r"centros? de datos", r"data cent(er|re)s?",
                r"\b(auge|burbuja|boom|gasto|inversion\w*|acciones|riesgo|apuesta\w*) (de|en|por) (la )?(ia|inteligencia artificial)\b",
                r"\b(ia|inteligencia artificial)\b.*\b(mercados?|bolsa|inversion\w+|acciones|wall street)\b"],
      efectos=[(+1, "chips y nube", ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "EQQQ"], "son los que venden y usan la infraestructura de inteligencia artificial"),
               (+1, "Nasdaq en general", ["AAPL", "TSLA", "CSPX"], "pesan mucho en los índices: los arrastran en la misma dirección")]),
    T(clave="bancos_eeuu", nombre="Bancos de EE. UU.", factor="sp500", que="los resultados de los bancos de EE. UU.",
      patrones=[r"bank earnings", r"resultados de (los )?bancos", r"bancos de (ee\.? ?uu|estados unidos|wall street)", r"\bjpmorgan (reporta|reports|earnings|profit|gana)", r"wall street banks", r"big banks"],
      efectos=[(+1, "bancos de EE. UU.", ["JPM", "BAC", "C"], "los resultados de uno se leen como señal para todos"),
               (+1, "pagos", ["V", "MA"], "si la gente gasta y paga sus tarjetas, también ganan las redes de pago")]),
    T(clave="consumo_eeuu", nombre="Consumo en EE. UU.", factor="sp500", que="el consumo en EE. UU.",
      patrones=[r"retail sales", r"ventas minoristas", r"consumer (sentiment|confidence|spending)", r"confianza del consumidor", r"gasto del consumidor"],
      efectos=[(+1, "consumo y entretenimiento", ["WMT", "AMZN", "KO", "DIS"], "viven de lo que gastan los hogares"),
               (+1, "pagos", ["V", "MA"], "cobran por cada compra con tarjeta"),
               (+1, "autos", ["F", "TSLA"], "un carro es la compra que primero se aplaza cuando el consumidor se asusta")]),
    T(clave="aranceles", nombre="Aranceles y comercio", factor="sp500", que="las barreras al comercio", signo_factor=-1,
      patrones=[r"\barancel\w*\b", r"\btariffs?\b", r"guerra comercial", r"trade war", r"\bdumping\b"],
      arriba=r"(impon\w+|impuso|nuevos? arancel\w*|sube\w* arancel\w*|new tariffs?|raises? tariffs?|amenaza\w*|threat\w*)",
      abajo=r"(suspend\w+|retira\w*|rebaja\w*|acuerdo comercial|trade deal|pausa|exenci\w+|cut\w* tariffs?|strike deal)",
      efectos=[(-1, "autos y fabricantes", ["F", "TSLA", "GE"], "arman con piezas importadas: el arancel les sube el costo"),
               (-1, "tecnológicas con fábricas en Asia", ["AAPL", "NVDA"], "producen afuera y venden en EE. UU."),
               (-1, "consumo", ["WMT", "AMZN"], "venden mercancía importada: o suben precios o pierden margen")]),
    T(clave="energia_colombia", nombre="Energía eléctrica en Colombia", factor="colombia", que="las reglas o el clima para la energía en Colombia",
      patrones=[r"\bcreg\b", r"tarifas? de energia", r"embalses", r"fenomeno del? nino", r"racionamiento", r"precio de bolsa de energia", r"cargo por confiabilidad", r"generacion (hidraulica|termica|electrica)"],
      efectos=[(0, "energía eléctrica", ["CELSIA", "GEB", "ISA", "PROMIGAS"], "son las que generan, transportan y venden la energía: cualquier cambio de reglas o de lluvias les mueve el ingreso")]),
    T(clave="china", nombre="China", factor="petroleo", que="la economía de China",
      patrones=[r"\byuan\b", r"(china|chino|china's|pekin|beijing).*(economia|economy|estimulo|stimulus|pib|gdp|crecimiento|growth|exportaciones|exports|deflac\w+|deflation|demanda|demand|inmobiliari\w+|property)",
                r"(economia|estimulo|pib|crecimiento|exportaciones|stimulus|economy) (de |from |in )?(china|chino)"],
      efectos=[(+1, "materias primas", ["ECOPETROL", "PBR", "MINEROS", "CNEC"], "China es el mayor comprador de materias primas: si crece, suben"),
               (+1, "tecnológicas que venden allá", ["AAPL", "TSLA", "NVDA"], "una parte grande de sus ventas viene de China")]),
]
POR_CLAVE = {t["clave"]: t for t in TEMAS}

# ------------------------------------------------------------------ empresas globales que se negocian en la Bolsa de Colombia
# (acción, cómo la nombran los titulares, sector). Los bancos y casas de bolsa aparecen a diario OPINANDO de otras empresas ("JPMorgan sube el objetivo de…"):
# sólo cuentan como noticia propia si el titular habla de SUS resultados o de SU acción (`propia`).
PROPIA = r"(resultados|ganancia\w*|utilidad\w*|earnings|profit\w*|revenue|ingresos|acciones? de|shares|stock|despid\w+|layoff\w*|multa|fine[sd]?|demanda|lawsuit|ceo|reporta|reports?)"
PROPIA_CERCA = r"\W+(\w+\W+){0,2}(earnings|profit\w*|results|shares|stock|resultados|ganancia\w*|utilidad\w*|gana|gano|pierde|perdio)\b"       # "JPMorgan reports record earnings"
GLOBALES: list[tuple[str, list[str], str, bool]] = [
    ("NUCO", [r"\bnubank\b", r"nu holdings", r"\bnu colombia\b", r"\bnu (stock|shares)\b"], "brasil y fintech", False),
    ("PBR", [r"\bpetrobras\b"], "petróleo y gas", False),
    ("NVDA", [r"\bnvidia\b"], "chips e inteligencia artificial", False),
    ("MSFT", [r"\bmicrosoft\b"], "tecnología", False),
    ("AAPL", [r"\bapple\b", r"\biphone\b"], "tecnología", False),
    ("AMZN", [r"\bamazon\b"], "tecnología y consumo", False),
    ("GOOGL", [r"\bgoogle\b", r"\balphabet\b"], "tecnología", False),
    ("META", [r"\bmeta platforms\b", r"\bfacebook\b", r"\binstagram\b", r"\bwhatsapp\b", r"\bzuckerberg\b", r"\bmeta (reporta|gana|pierde|stock|shares|anuncia|lanza)\b", r"acciones de meta"], "tecnología", False),
    ("TSLA", [r"\btesla\b"], "autos", False),
    ("UBER", [r"\buber\b"], "movilidad", False),
    ("WMT", [r"\bwalmart\b"], "consumo", False),
    ("KO", [r"coca[- ]cola"], "consumo", False),
    ("DIS", [r"\bdisney\b"], "entretenimiento", False),
    ("F", [r"\bford (motor|reporta|reports|recall|llama|stock|shares|gana|pierde|ev\b)", r"acciones de ford"], "autos", False),
    ("GE", [r"general electric", r"\bge aerospace\b", r"\bge vernova\b"], "industria", False),
    ("PFE", [r"\bpfizer\b"], "salud", False),
    ("JNJ", [r"johnson & johnson", r"johnson and johnson", r"\bj&j\b"], "salud", False),
    ("V", [r"\bvisa inc\b", r"visa y mastercard", r"visa and mastercard", r"acciones de visa", r"\bvisa (stock|shares|reporta|reports)\b"], "pagos", False),
    ("MA", [r"\bmastercard\b"], "pagos", False),
    ("JPM", [r"\bjpmorgan\b", r"\bjp morgan\b"], "bancos de EE. UU.", True),
    ("BAC", [r"bank of america", r"\bbofa\b"], "bancos de EE. UU.", True),
    ("C", [r"\bcitigroup\b", r"\bciti\b"], "bancos de EE. UU.", True),
]
SECTOR_GLOBAL = {t: s for t, _, s, _ in GLOBALES}
BUENA = (r"\b(sube|subio|se dispara|disparo|gana|gano|supera|supero|record|mejora|eleva|elevo|aprueba|aprobo|acuerdo|alianza|recompra|"
         r"beats?|tops?|surge\w*|jump\w*|soar\w*|rall\w+|gain\w*|upgrade\w*|raises?|record|wins?|approval|buyback|partnership|strong)\b")
MALA = (r"\b(cae|cayo|se desploma|desplomo|pierde|perdio|recorta|recorto|rebaja|demanda|multa|investigacion|retira|despid\w+|incumpl\w+|"
        r"miss\w*|falls?|fell|drop\w*|slide\w*|slump\w*|plunge\w*|downgrade\w*|cuts?|lawsuit|probe|recall\w*|fine[sd]?|layoff\w*|weak|warning|warns?)\b")
RUIDO = re.compile(r"(precio del dolar hoy|dolar hoy|a como esta|casas de cambio|horoscopo|en vivo|minuto a minuto|loteria|pico y placa|resultados? (del |de la )?(baloto|chance|futbol|partido)|"
                   r"quien es quien|receta|cupon|black friday|hot sale|como (hacer|ganar|invertir)|mejores tarjetas|estilo de vida|serie de|pelicula|concierto|que es y como|"
                   r"price target|precio objetivo de|forecast &|analyst predictions|stock forecast|"
                   r"^\d+ \w+|is it (still )?(a|too late to) buy|should you buy|still a buy|better buy|time to buy|stock (is|has) (up|down|surged|soared|jumped) \d|justify (the|its) valuation|"
                   r"here'?s why|what to (know|watch|expect)|ripe for|income seekers|protect your wealth|prediction for|motley fool|para aprender|herramientas de|mejores aplicaciones|"
                   r"trabajar en|empleos abiertos|manijas|\bcv\b)")


# ------------------------------------------------------------------ clasificar
def direccion(titulo_norm: str, tema: dict[str, Any]) -> int:
    """+1 si el titular dice que el tema SUBE (o se agrava), −1 si baja (o se calma), 0 si no lo dice. Cada tema puede traer sus propias palabras."""
    a = len(re.findall(tema.get("arriba") or SUBE, titulo_norm))
    b = len(re.findall(tema.get("abajo") or BAJA, titulo_norm)) * int(tema.get("peso_abajo", 1))      # un titular de tregua casi siempre nombra también la guerra
    if tema.get("arriba") and not a and not b:                                         # temas con palabras propias: de respaldo, las genéricas
        a, b = len(re.findall(SUBE, titulo_norm)), len(re.findall(BAJA, titulo_norm))
    return (a > b) - (b > a)


def temas_de(titulo: str) -> list[tuple[dict[str, Any], int]]:
    """Temas de mercado que toca el titular, cada uno con su dirección. Máximo dos (el titular más cargado no debe volverse una lista de todo)."""
    t = N.norm(titulo)
    hallados = [(tema, sum(1 for p in tema["patrones"] if re.search(p, t))) for tema in TEMAS]
    hallados = sorted([h for h in hallados if h[1]], key=lambda h: -h[1])[:2]
    return [(tema, direccion(t, tema)) for tema, _ in hallados]


def globales_de(titulo: str) -> list[str]:
    """Acciones globales de la Bolsa de Colombia de las que habla el titular (no las que sólo opinan en él)."""
    t = N.norm(titulo)
    out = []
    for ticker, patrones, _, exige in GLOBALES:
        if any(re.search(p + (PROPIA_CERCA if exige else ""), t) or (exige and re.search(r"(acciones|resultados|utilidad\w*|ganancias) de (\w+ ){0,2}" + p, t)) for p in patrones):
            out.append(ticker)
    return out[:3]


def tono(titulo: str) -> int:
    t = N.norm(titulo)
    a, b = len(re.findall(BUENA, t)), len(re.findall(MALA, t))
    return (a > b) - (b > a)


def analizar(item: N.Item, permitidas: set[str] | None = None) -> dict[str, Any] | None:
    """Lo que la noticia significa para las acciones: {tipo, temas, direccion, sectores[{nombre, signo, tickers, porque}], beneficiadas, afectadas, expuestas}.
    None si la noticia no toca ningún tema de mercado ni a una empresa global. `permitidas` = acciones del concurso (las demás no se nombran)."""
    if RUIDO.search(N.norm(item.titulo)):
        return None
    ok = (lambda xs: [x for x in xs if permitidas is None or x in permitidas])          # noqa: E731
    empresas = ok(globales_de(item.titulo))
    temas = temas_de(item.titulo)
    if not empresas and not temas:
        return None
    sectores: list[dict[str, Any]] = []
    if empresas:                                                                       # noticia de una empresa: ella primero, luego su sector de rebote
        s = tono(item.titulo)
        for e in empresas:
            sectores.append(dict(nombre=SECTOR_GLOBAL[e], signo=s, tickers=[e], porque="es la empresa de la noticia", directa=True))
        pares = ok([t for t, _, sec, _ in GLOBALES if sec == SECTOR_GLOBAL[empresas[0]] and t not in empresas])
        if pares:
            sectores.append(dict(nombre=f"otras de {SECTOR_GLOBAL[empresas[0]]}", signo=s, tickers=pares, directa=False,
                                 porque="lo que le pasa a una se lee como señal para sus pares (efecto más débil)"))
    for tema, d in temas:
        for signo, nombre, tickers, porque in tema["efectos"]:
            tk = ok([t for t in tickers if t not in empresas])
            if tk:
                sectores.append(dict(nombre=nombre, signo=(signo * d) if signo else 0, base=signo, mixto=signo == 0, tickers=tk, porque=porque, directa=False,
                                     tema=tema["clave"], que=tema["que"]))
    if not sectores:
        return None
    vistos: set[str] = set()
    benef, afect, exp = [], [], []
    for s in sectores:                                                                 # una acción aparece una sola vez: en el primer sector que la nombra
        s["tickers"] = [t for t in s["tickers"] if not (t in vistos or vistos.add(t))]
        (benef if s["signo"] > 0 else afect if s["signo"] < 0 else exp).extend(s["tickers"])
    sectores = [s for s in sectores if s["tickers"]]
    dirs = [d for _, d in temas]
    return dict(tipo="empresa" if empresas else "tema", empresas=empresas, temas=[t["clave"] for t, _ in temas], nombres=[t["nombre"] for t, _ in temas],
                direccion=(dirs[0] if dirs else tono(item.titulo)), sectores=sectores, beneficiadas=benef, afectadas=afect, expuestas=exp,
                factor=(temas[0][0].get("factor") if temas else None), que=(temas[0][0]["que"] if temas else None))


# ------------------------------------------------------------------ leer las fuentes
def items_de_pagina(texto: str, fuente: str, base: str, ahora: dt.datetime) -> list[N.Item]:
    """Titulares de una página de portada (sin RSS): cada enlace a una nota con su título. La página no trae la hora de cada nota: se usa la de lectura
    (la primera vez que se ve; lo ya visto se descarta por su huella)."""
    out, vistos = [], set()
    for href, cuerpo in re.findall(r'<a[^>]+href="(/[^"#?]+)"[^>]*>(.*?)</a>', texto or "", flags=re.S):
        titulo = " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", cuerpo)).split())
        if len(titulo) < 35 or href in vistos or href.count("/") < 3 or re.search(r"/(video|videos|opinion|especiales|estilo-de-vida|autor|tags)/", href):
            continue
        vistos.add(href)
        out.append(N.Item(ahora.astimezone(UTC), titulo, "", fuente, base.rstrip("/") + href, "pagina"))
    return out


class Lector:
    """Lee las fuentes de mercado en paralelo. Reutiliza el lector de `noticias_bvc` (peticiones condicionales, tolerancia a fallos) y le añade
    páginas de portada y los titulares por acción de Yahoo Finanzas."""

    def __init__(self, cfg: dict[str, Any], http: Any = None, env: dict[str, str] | None = None):
        self.cfg, self.m = cfg, cfg["noticias_mercado"]
        self.base = N.Lector(cfg, http, env)
        self.ronda = 0
        self.fallos: dict[str, int] = {}
        u = cfg["universe"]
        from .data_loader import yahoo_symbol
        self.simbolos = [yahoo_symbol(t, cfg) for t in list(u.get("mgc", [])) + list(u.get("local", []))]

    def _rss(self, f: dict[str, Any]) -> list[N.Item]:
        return self.base.rss(f)

    def _pagina(self, f: dict[str, Any], ahora: dt.datetime) -> list[N.Item]:
        r = self.base._get(f["url"], condicional=False)
        return items_de_pagina(r.content.decode("utf-8", "replace"), f["nombre"], f.get("base") or f["url"], ahora)

    def _yahoo(self) -> list[N.Item]:
        y = self.m["yahoo_titulares"]
        n = max(int(y.get("por_ronda", 8)), 1)
        k = (self.ronda // max(int(y.get("cada_rondas", 1)), 1)) * n % max(len(self.simbolos), 1)
        lote = (self.simbolos + self.simbolos)[k:k + n]
        r = self.base._get(y["url"], {"s": ",".join(lote), "region": "US", "lang": "en-US"}, condicional=False)
        return N.items_de_rss(r.content.decode("utf-8", "replace"), "Yahoo Finance")

    def todo(self, ahora: dt.datetime, completo: bool = False) -> tuple[list[N.Item], dict[str, bool]]:
        tareas: list[tuple[str, Callable[[], list[N.Item]]]] = []
        toca = lambda f: completo or self.ronda % max(int(f.get("cada_rondas", 1)), 1) == 0      # noqa: E731
        for f in self.m.get("rss", []):
            if toca(f):
                tareas.append((f["nombre"], lambda f=f: self._rss(f)))
        for f in self.m.get("paginas", []):
            if toca(f):
                tareas.append((f["nombre"], lambda f=f: self._pagina(f, ahora)))
        y = self.m.get("yahoo_titulares") or {}
        if y.get("activo") and self.simbolos and toca(y):
            tareas.append(("Yahoo Finance (por acción)", self._yahoo))
        if self.m.get("superfinanciera", True) and self.cfg["noticias_bvc"]["fuentes"]["sfc"]["activo"] and toca({"cada_rondas": self.m.get("sfc_cada_rondas", 2)}):
            tareas.append(("Superfinanciera", lambda: self.base.sfc(ahora)))
        self.ronda += 1
        activas = [(n, fn) for n, fn in tareas if completo or self.fallos.get(n, 0) < 3 or self.ronda % 10 == 0]

        def una(par: tuple[str, Callable[[], list[N.Item]]]) -> tuple[str, list[N.Item] | None]:
            try:
                return par[0], par[1]()
            except Exception:                                                          # noqa: BLE001 — una fuente caída no tumba a las demás
                return par[0], None
        items: list[N.Item] = []
        salud: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=max(len(activas), 1)) as ex:
            for nombre, res in ex.map(una, activas):
                salud[nombre] = res is not None
                self.fallos[nombre] = 0 if res is not None else self.fallos.get(nombre, 0) + 1
                items += res or []
        return items, salud

    def resumen_de(self, url: str) -> str:
        """Descripción de la nota (la que el propio medio pone para compartir), cuando la fuente no trae resumen. '' si no se puede."""
        try:
            r = self.base._get(url, condicional=False)
            t = r.content.decode("utf-8", "replace")[:200000]
            m = re.search(r'<meta[^>]+(?:property|name)="(?:og:description|description)"[^>]+content="([^"]{30,})"', t) or \
                re.search(r'<meta[^>]+content="([^"]{30,})"[^>]+(?:property|name)="(?:og:description|description)"', t)
            return N.limpiar(m.group(1)) if m else ""
        except Exception:                                                              # noqa: BLE001
            return ""


def recortar(texto: str, n: int = 320) -> str:
    """El resumen en una o dos frases completas (hasta `n` letras)."""
    t = " ".join((texto or "").split())
    if len(t) <= n:
        return t
    corte = max(t.rfind(". ", 0, n), t.rfind("; ", 0, n))
    return (t[:corte + 1] if corte > n * 0.5 else t[:n].rsplit(" ", 1)[0] + "…").strip()


# ------------------------------------------------------------------ una ronda
def ronda(lector: Any, mem: Any, ahora: dt.datetime, cfg: dict[str, Any], locales: Callable[[N.Item], bool] | None = None) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Lee las fuentes y devuelve las noticias NUEVAS de mercado ya analizadas (cada una con su `item`), más la salud de las fuentes.
    La primera vez sólo toma la línea base (nada de lo que ya estaba publicado se avisa). `locales(item)` = la noticia es de una empresa colombiana que
    ya avisa `noticias_bvc`: aquí no se repite, salvo que además toque un tema de mercado."""
    m = cfg["noticias_mercado"]
    d = mem.d.setdefault("mercado", {})
    d.setdefault("vistos", {})
    d.setdefault("temas", {})
    d.setdefault("titulos", [])
    d.setdefault("ultimas", [])
    items, salud = lector.todo(ahora)
    d["salud"] = {**d.get("salud", {}), **salud}
    ahora_utc = ahora.astimezone(UTC)
    primera = not d.get("base")
    u = cfg["universe"]
    permitidas = set(u.get("local", [])) | set(u.get("mgc", [])) | set(u.get("etf", []))
    nuevos = []
    for it in items:
        h = N.huella(it.titulo)
        if h not in d["vistos"]:
            nuevos.append((h, it))
    if primera:
        if any(salud.values()):
            for h, _ in nuevos:
                d["vistos"][h] = ahora_utc.isoformat(timespec="seconds")
            d["base"] = ahora_utc.isoformat(timespec="seconds")
            mem.sucia = True
        return [], salud
    limite = ahora_utc - dt.timedelta(minutes=m["ventana_min"])
    hace_12h = (ahora_utc - dt.timedelta(hours=12)).isoformat(timespec="seconds")
    d["titulos"] = [x for x in d["titulos"] if x[0] >= hace_12h]
    out: list[dict[str, Any]] = []
    marca = ahora_utc.isoformat(timespec="seconds")
    for h, it in sorted(nuevos, key=lambda p: -p[1].ts.timestamp()):
        if len(out) >= m["max_por_ronda"]:
            break                                                                      # lo que no cupo queda sin marcar: sale en la ronda siguiente
        d["vistos"][h] = marca
        mem.sucia = True
        if it.ts < limite or it.ts > ahora_utc + dt.timedelta(minutes=10):
            continue
        a = analizar(it, permitidas)
        if a is None or (locales and locales(it) and not a["temas"]):
            continue
        if N.es_repetida(it.titulo, [x[1] for x in d["titulos"]], m["parecido_repetida"]):
            continue                                                                   # la misma noticia en otro medio
        clave = (a["temas"][0] if a["temas"] else "emp:" + a["empresas"][0]) + f"|{a['direccion']:+d}"
        ult = d["temas"].get(clave)
        if ult and (ahora_utc - dt.datetime.fromisoformat(ult)).total_seconds() / 60 < m["enfriamiento_tema_min"]:
            continue                                                                   # mismo tema y misma dirección hace un momento
        d["temas"][clave] = marca
        d["titulos"].append([marca, sorted(N.palabras_clave(it.titulo))])
        a["item"] = it
        out.append(a)
        d["ultimas"].append(dict(ts=marca, titulo=it.titulo[:220], fuente=it.fuente, url=it.url, nombres=a["nombres"], empresas=a["empresas"], direccion=a["direccion"],
                                 beneficiadas=a["beneficiadas"][:8], afectadas=a["afectadas"][:8], expuestas=a["expuestas"][:8]))
    d["ultimas"] = d["ultimas"][-30:]
    lim = (ahora_utc - dt.timedelta(days=3)).isoformat()
    d["vistos"] = {k: v for k, v in d["vistos"].items() if v >= lim}
    return out, salud
