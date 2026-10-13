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
    T(clave="energia_colombia", nombre="Energía en Colombia (riesgo de apagón)", factor="colombia", que="el riesgo de apagón o de racionamiento", peso_abajo=2,
      patrones=[r"\bapagon(es)?\b", r"racionamiento", r"crisis energetica", r"(riesgo\w*|seguridad|emergencia|deficit|escasez) (energetic\w+|de energia|electric\w+)", r"sistema electrico", r"sector electrico",
                r"\bcreg\b", r"tarifas? de (la )?energia", r"embalses", r"fenomeno del? nino", r"precio de bolsa de energia", r"cargo por confiabilidad", r"opcion tarifaria",
                r"generacion (hidraulica|termica|electrica|de energia)", r"(termicas|hidroelectricas|generadoras|comercializadoras) (de energia)?", r"energia electrica", r"servicio de energia",
                r"\bel nino\b.*\b(energ\w+|embalse\w*|apagon\w*|racionamiento|sequia|medidas|plan|crisis|riesgo\w*|gobierno|presidente|estrategia)\b",
                r"\b(energ\w+|apagon\w*|racionamiento|sequia|medidas|plan|crisis|riesgo\w*|gobierno|presidente|estrategia)\b.*\bel nino\b"],
      arriba=r"(apagon\w*|racionamiento|crisis|alerta\w*|riesgo\w*|deficit|escasez|advierte|advirtio|se quedaria sin|emergencia|sequia|nos puede apagar|sin energia)",
      abajo=r"(plan|medidas|evitar|evita|garantiz\w+|solucion\w*|arregl\w+|destrab\w+|inversion\w*|invertir\w*|traslado|proteger|protege|promet\w+|enfrentar|afrontar|contingencia|fecha limite|"
            r"estaran en marcha|optimizar|aprob\w+|refuerz\w+|nueva capacidad|entran? en operacion)",
      efectos=[(0, "energía eléctrica: redes y generación", ["ISA", "CELSIA", "GEB"],
                "ISA lleva la energía por las líneas de transmisión, y Celsia y GEB la generan y la venden. Cuando el gobierno pone un plan con plata y con fecha, se destraban líneas "
                "y plantas que estaban frenadas y baja el riesgo que más castiga a estas acciones: que les intervengan los precios o que haya racionamiento. Si en cambio sólo crece "
                "la alarma, el efecto es mixto: la energía escasa se vende más cara, pero una hidroeléctrica sin agua genera menos y aumenta la presión para controlar tarifas", +1),
               (+1, "gas para las plantas térmicas", ["PROMIGAS", "CNEC", "ECOPETROL"],
                "cuando falta agua en los embalses, las plantas térmicas se prenden más y queman más gas: sube la demanda para quien lo produce y lo transporta", 0),
               (-1, "industria que gasta mucha energía", ["CEMARGOS", "PFCEMARGOS", "ENKA", "NUTRESA"],
                "el cemento, los textiles y los alimentos gastan mucha energía: si escasea, la pagan más cara o tienen que parar hornos y líneas de producción"),
               (-1, "comercio y consumo", ["EXITO"], "los cortes de luz y las tarifas altas le quitan ventas a las tiendas y plata al bolsillo de los hogares")]),
    T(clave="hidrocarburos_colombia", nombre="Petróleo y gas en Colombia (reglas del sector)", factor="petroleo", que="el apoyo del gobierno a producir petróleo y gas",
      patrones=[r"\bfracking\b", r"exploracion (de |y produccion de )?(hidrocarburos|petroleo|gas)", r"contratos (petroleros|de exploracion)", r"\banh\b", r"hectareas para exploracion", r"regasificadora",
                r"importacion(es)? de gas", r"(deficit|escasez|reservas) de gas", r"gas natural (en|de) colombia", r"precio de la gasolina", r"precio del acpm", r"\bfepc\b", r"regalias (petroleras|de hidrocarburos)",
                r"(hidrocarburos|petroleo|gas).*\b(gobierno|presidente|minenergia|ministerio de minas|decreto)\b.*colombia", r"\b(minenergia|ministerio de minas)\b"],
      arriba=r"(reserva\w*|habilit\w+|aprob\w+|autoriz\w+|reactiv\w+|impuls\w+|destrab\w+|adjudic\w+|firma\w*|nuevos contratos|invert\w+|inversion\w*|hallazgo|descubr\w+|levanta\w* (la )?prohib\w+|permit\w+)",
      abajo=r"(prohib\w+|suspend\w+|fren\w+|cancel\w+|no firmara|moratoria|deficit|escasez|sancion\w*|congel\w+|sube el precio|alza)",
      efectos=[(+1, "petróleo y gas", ["ECOPETROL", "CNEC"],
                "son las que buscan y sacan petróleo y gas en Colombia: más áreas para explorar y reglas que lo permitan significan más reservas y más producción en los próximos años; "
                "una prohibición o un freno hace lo contrario"),
               (+1, "transporte y distribución de gas", ["PROMIGAS", "TERPEL"],
                "Promigas mueve el gas por sus tubos y Terpel vende los combustibles: viven del volumen que pasa por su red, así que más producción y más oferta les conviene"),
               (+1, "empresas con dueño estatal", ["ISA", "GEB"], "lo que el gobierno decide con Ecopetrol, que es dueña de la mayoría de ISA, termina reflejándose en el grupo")]),
    T(clave="infraestructura_colombia", nombre="Obras, vivienda e infraestructura en Colombia", factor="colombia", que="el impulso a las obras y la vivienda",
      patrones=[r"megaproyecto\w*", r"megaobra\w*", r"licencia\w* ambiental\w*", r"concesion(es)? (vial\w*|aeroportuari\w+|de vias)", r"\b(4g|5g) vial\w*", r"vias (4g|5g)", r"(tarifas?|alza|aumento|cobro) de(l| los)? peajes?", r"obras? (publicas?|de infraestructura|civiles)",
                r"infraestructura (vial|de transporte)", r"subsidios? de vivienda", r"mi casa ya", r"vivienda (de interes social|nueva|vis)", r"desarrollo urbano", r"obras del metro", r"\bani\b", r"\binvias\b",
                r"aeropuerto el dorado", r"conexion vial", r"sector (de la )?construccion", r"ventas de vivienda", r"licencias de construccion"],
      arriba=r"(aprob\w+|aprueb\w+|destrab\w+|aceler\w+|impuls\w+|invert\w+|inversion\w*|adjudic\w+|reactiv\w+|elimin\w+ (la )?(medida|norma|restriccion)|quit\w+ demoras|avanza\w*|inici\w+|financia\w*|subsidi\w+|crec\w+|repunt\w+)",
      abajo=r"(fren\w+|suspend\w+|cancel\w+|retras\w+|demor\w+|paraliz\w+|problemas|cae\w*|caida|desplom\w+|sin recursos|desfinanci\w+|bloque\w+|recort\w+)",
      efectos=[(+1, "cemento y construcción", ["CEMARGOS", "PFCEMARGOS"],
                "cada obra y cada casa nueva es cemento y concreto vendido: cuando el gobierno destraba licencias o pone subsidios de vivienda, sube el volumen que despachan; si las obras se frenan, baja"),
               (+1, "concesiones y obras", ["GRUPOARGOS", "PFGRUPOARG", "CORFICOLCF", "PFCORFICOL"],
                "Grupo Argos (con Odinsa) y Corficolombiana son dueños de vías y aeropuertos en concesión: ganan cuando los proyectos avanzan y se aprueban, y pierden con retrasos, bloqueos o cambios en los peajes"),
               (+1, "bancos que financian las obras", ["PFBCOLOM", "BCOLOMBIA", "BOGOTA", "PFAVAL"], "prestan la plata para construir y para comprar vivienda: más obras son más crédito")]),
    T(clave="consumo_colombia", nombre="Bolsillo de los hogares en Colombia", factor="colombia", que="la plata que tienen los hogares para gastar",
      patrones=[r"salario minimo", r"\biva\b", r"dia sin iva", r"ventas (del comercio|minoristas|de las tiendas)", r"confianza del consumidor", r"\bremesas\b", r"\bdesempleo\b", r"canasta familiar", r"costo de vi(da|vir)",
                r"gasto de los hogares", r"impuesto\w* (saludable\w*|a (las )?bebidas|a (los )?alimentos)"],
      arriba=r"(sube el salario|aumento del salario|alza del salario|crec\w+|repunt\w+|mejor\w+|baja (el )?(iva|desempleo)|reduc\w+ (el )?(iva|desempleo)|dia sin iva|record|aument\w+ (las )?(ventas|remesas))",
      abajo=r"(cae\w*|caida|baj\w+ (las )?ventas|sube (el )?(iva|desempleo)|nuevo impuesto|mas impuestos|encarec\w+|golpea\w*|apriet\w+|se desploma|pierde poder)",
      efectos=[(+1, "comercio y alimentos", ["EXITO", "NUTRESA"],
                "venden lo que los hogares compran todos los días: con más ingreso o menos impuestos la gente gasta más en sus tiendas y productos; con impuestos nuevos o desempleo, gasta menos"),
               (+1, "bancos (crédito de consumo)", ["PFBCOLOM", "BCOLOMBIA", "BOGOTA", "PFAVAL", "BHI"], "un hogar con más ingreso pide más crédito y lo paga a tiempo; uno apretado se atrasa en las cuotas")]),
    T(clave="banca_colombia", nombre="Bancos y crédito en Colombia", factor="colombia", que="las condiciones para los bancos",
      patrones=[r"tasa de usura", r"4x1000", r"cuatro por mil", r"cartera (vencida|en mora|de credito)", r"utilidades de (los )?bancos", r"sistema financiero", r"inversiones forzosas", r"sector bancario",
                r"credito (de consumo|hipotecario|de vivienda)", r"desembolsos", r"superfinanciera.*\b(bancos|entidades financieras|cartera)\b"],
      arriba=r"(crec\w+|repunt\w+|mejor\w+|sube la tasa de usura|aument\w+ (las )?utilidades|baja la mora|elimin\w+ (el )?4x1000|recuper\w+)",
      abajo=r"(cae\w*|caida|deterior\w+|sube la mora|aument\w+ (la )?(mora|cartera vencida)|baja la tasa de usura|inversiones forzosas|nuevo impuesto|sobretasa|perdidas)",
      efectos=[(+1, "bancos y grupos financieros", ["PFBCOLOM", "BCOLOMBIA", "BOGOTA", "AVAL", "PFAVAL", "BHI", "CORFICOLCF", "GRUPOSURA", "PFSURA"],
                "su negocio es prestar y cobrar: ganan cuando el crédito crece y la gente paga, y pierden cuando sube la mora o cuando el gobierno les pone topes, impuestos o inversiones obligatorias")]),
    T(clave="mineria_colombia", nombre="Minería en Colombia", factor="oro", que="las condiciones para la minería legal",
      patrones=[r"\bmineria\b", r"titulos? minero\w*", r"codigo de minas", r"regalias mineras", r"oro (ilegal|de aluvion)", r"contrabando de oro", r"explotacion ilegal de oro", r"sector minero"],
      arriba=r"(formaliz\w+|aprob\w+|impuls\w+|combat\w+|intervino|intervien\w+|golpe\w* a|decomis\w+|aument\w+ (la )?produccion|nuevos titulos)",
      abajo=r"(prohib\w+|suspend\w+|fren\w+|nuevo impuesto|mas regalias|sube\w* (las )?regalias|bloque\w+|ataque\w*|aument\w+ (de )?(la )?(explotacion|mineria) ilegal)",
      efectos=[(+1, "minería de oro", ["MINEROS"],
                "Mineros saca oro en Antioquia y en Nicaragua: le ayuda que se combata la minería ilegal, que compite con ella y le crea problemas de seguridad, y le pesa cualquier impuesto, "
                "regalía o freno nuevo a los títulos mineros")]),
    T(clave="china", nombre="China", factor="petroleo", que="la economía de China",
      patrones=[r"\byuan\b", r"(china|chino|china's|pekin|beijing).*(economia|economy|estimulo|stimulus|pib|gdp|crecimiento|growth|exportaciones|exports|deflac\w+|deflation|demanda|demand|inmobiliari\w+|property)",
                r"(economia|estimulo|pib|crecimiento|exportaciones|stimulus|economy) (de |from |in )?(china|chino)"],
      efectos=[(+1, "materias primas", ["ECOPETROL", "PBR", "MINEROS", "CNEC"], "China es el mayor comprador de materias primas: si crece, suben"),
               (+1, "tecnológicas que venden allá", ["AAPL", "TSLA", "NVDA"], "una parte grande de sus ventas viene de China")]),
]
POR_CLAVE = {t["clave"]: t for t in TEMAS}

# Por qué importa cada tema y qué mirar en la semana: va en el aviso, debajo de "Cómo va el mercado". Explica el mecanismo; no promete un resultado.
EXPLICA = {
    "petroleo": "El precio del petróleo es el ingreso de las petroleras y, a la vez, un costo para casi todo lo demás. Para Colombia pesa doble: Ecopetrol es la acción más negociada "
                "y el petróleo es el principal producto de exportación, así que también mueve el dólar. Esta semana conviene mirar si el movimiento se sostiene más de un día: los saltos "
                "por un susto suelen devolverse cuando el susto pasa.",
    "tasas_eeuu": "Las tasas de interés de Estados Unidos son el precio del dinero en el mundo. Cuando suben, la plata sale de las acciones más riesgosas y de los países emergentes hacia "
                  "los bonos, y eso pega primero en las tecnológicas, en Nubank y en la bolsa colombiana; cuando bajan, pasa lo contrario. Lo que hay que mirar en la semana son los datos "
                  "de inflación y empleo de Estados Unidos, que son los que mueven esas tasas.",
    "inflacion_eeuu": "La inflación y el empleo de Estados Unidos deciden qué hace la Reserva Federal con las tasas. Un dato más alto de lo esperado hace pensar en tasas altas por más "
                      "tiempo, y eso baja las acciones de crecimiento y fortalece el dólar; un dato bajo hace lo contrario. El efecto se ve el mismo día del dato y a veces se corrige al siguiente.",
    "tasas_colombia": "Las tasas del Banco de la República y la inflación definen cuánto cuesta el crédito en Colombia. Con tasas altas pagan más las empresas con mucha deuda (energía, "
                      "construcción) y los hogares compran menos; los bancos ganan margen pero prestan menos. En la semana importa si el dato sale por encima o por debajo de lo que el mercado esperaba.",
    "dolar": "El dólar cambia el valor en pesos de todo lo que viene de afuera. Las acciones globales que compras en trii se negocian en pesos: si el dólar sube, suben en pesos aunque en "
             "Nueva York no se muevan, y si baja, pierden. A las empresas colombianas que exportan les ayuda un dólar alto; a las que importan, uno bajo.",
    "oro": "El oro sube cuando hay miedo o cuando bajan las tasas, y baja cuando los bonos rinden más. Mineros vende oro, así que su ingreso se mueve casi a la par del precio. "
           "Un solo día de alza no hace tendencia: mira si el oro sostiene el movimiento varios días.",
    "brasil": "Nubank y Petrobras son empresas brasileñas: su precio depende del ánimo del mercado de Brasil y de su moneda, el real, más que de lo que pase en Colombia. En época de "
              "elecciones se mueven con cada encuesta y cada resultado, para arriba y para abajo, y esos movimientos pueden ser de varios puntos en un día.",
    "geopolitica": "Un conflicto sube el petróleo y el oro, porque se teme que falte crudo y la plata busca refugio, y baja las acciones más movidas, que son las primeras que se venden "
                   "en un susto. Una tregua hace lo contrario. Son movimientos rápidos y con frecuencia se devuelven en pocos días si la noticia no escala.",
    "riesgo_colombia": "La confianza en las cuentas del gobierno colombiano mueve a todo el mercado local a la vez: al dólar, a los bonos y a las acciones, sobre todo a los bancos (que tienen "
                       "muchos bonos del gobierno) y a las empresas con dueño estatal. Lo que da tranquilidad sube el mercado; lo que hace dudar de que el gobierno cumpla, lo baja.",
    "ia_chips": "La inteligencia artificial es lo que más ha movido a las bolsas de Estados Unidos. Las empresas que fabrican los chips y las que construyen los centros de datos pesan "
                "tanto en los índices que arrastran al resto. Las noticias sobre gasto, demanda o dudas de una burbuja mueven a todo el grupo al tiempo.",
    "bancos_eeuu": "Los grandes bancos de Estados Unidos reportan sus resultados casi el mismo día, y lo que dice uno se toma como señal para los demás y para la economía: si la gente "
                   "paga sus créditos y gasta con tarjeta, ganan también las redes de pago.",
    "consumo_eeuu": "El gasto de los hogares es dos tercios de la economía de Estados Unidos. Los datos de ventas y de confianza anticipan cómo les irá a las empresas de consumo, "
                    "a las de pagos con tarjeta y a los fabricantes de carros, que es lo primero que se aplaza cuando el consumidor se asusta.",
    "aranceles": "Un arancel es un impuesto a lo que entra de afuera: le sube el costo a quien fabrica con piezas importadas o vende mercancía traída de otros países. Un acuerdo que "
                 "los baja o los suspende hace lo contrario. El efecto depende de qué tan en serio se tome el anuncio: muchos se negocian o se aplazan.",
    "energia_colombia": "La energía de Colombia depende de las lluvias: cuando los embalses bajan, hay que prender plantas térmicas más caras y crece el riesgo de racionamiento. Para ISA, "
                        "Celsia y GEB lo que más pesa no es el clima sino lo que haga el gobierno: un plan con plata y con fecha destraba proyectos y les da reglas claras; una intervención "
                        "de precios o de tarifas les quita ingreso. En la semana hay que mirar si el anuncio trae decretos y recursos concretos o se queda en promesa.",
    "hidrocarburos_colombia": "Colombia vive del petróleo y se está quedando corta de gas. Cada decisión del gobierno sobre exploración, fracking o importación de gas cambia cuánto pueden "
                              "producir Ecopetrol y Canacol en los próximos años y cuánto gas pasa por la red de Promigas. Lo que hay que mirar es si el anuncio viene con contratos y áreas "
                              "concretas o es sólo una intención.",
    "infraestructura_colombia": "Las obras y la vivienda mueven a toda una cadena: el cemento, los dueños de las concesiones de vías y aeropuertos y los bancos que las financian. "
                                "Cuando el gobierno destraba licencias, pone subsidios o adjudica proyectos, esa cadena vende más; cuando una obra se frena o se queda sin plata, pasa lo contrario. "
                                "Entre el anuncio y la obra pasan meses: el mercado reacciona a la señal.",
    "consumo_colombia": "Lo que los hogares tienen para gastar define las ventas de las tiendas y de las empresas de alimentos, y también si la gente paga sus créditos. El salario mínimo, "
                        "el IVA, el desempleo y las remesas son los datos que más lo mueven.",
    "banca_colombia": "Los bancos son la mayor parte de la bolsa colombiana. Ganan cuando el crédito crece y la gente paga, y pierden cuando sube la mora o cuando el gobierno les pone "
                      "topes a las tasas, impuestos o inversiones obligatorias. Una noticia que cambie esas reglas mueve a todo el sector al tiempo.",
    "mineria_colombia": "La minería legal de oro compite con la ilegal y depende de las reglas del gobierno sobre títulos y regalías. A Mineros le ayuda que se controle la minería ilegal "
                        "y que haya reglas estables; le pesa cualquier impuesto nuevo o bloqueo en sus zonas.",
    "china": "China es el mayor comprador de materias primas del mundo. Si su economía crece o recibe estímulos, sube la demanda de petróleo y metales; si se frena, baja. "
             "También es un mercado grande para Apple, Tesla y Nvidia.",
}
EXPLICA_EMPRESA = ("Es una noticia propia de la empresa. Su efecto llega primero a su acción y, más débil, a las de su mismo sector, porque el mercado la lee como una señal de cómo "
                   "va el negocio de todas. En trii estas acciones se negocian en pesos, así que el dólar también les suma o les resta.")

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
                   r"here.?s (why|what|how)|what to (know|watch|expect)|what (could|would|does|it means)|could mean|means for|needs to show|^exclusive|\bgimmick\b|"
                   r"ripe for|income seekers|protect your wealth|prediction for|motley fool|para aprender|herramientas de|mejores aplicaciones|"
                   r"trabajar en|empleos abiertos|manijas|\bcv\b)")


# ------------------------------------------------------------------ clasificar
def direccion(titulo_norm: str, tema: dict[str, Any]) -> int:
    """+1 si el titular dice que el tema SUBE (o se agrava), −1 si baja (o se calma), 0 si no lo dice. Cada tema puede traer sus propias palabras."""
    a = len(re.findall(tema.get("arriba") or SUBE, titulo_norm))
    b = len(re.findall(tema.get("abajo") or BAJA, titulo_norm)) * int(tema.get("peso_abajo", 1))      # un titular de tregua casi siempre nombra también la guerra
    if tema.get("arriba") and not a and not b:                                         # temas con palabras propias: de respaldo, las genéricas
        a, b = len(re.findall(SUBE, titulo_norm)), len(re.findall(BAJA, titulo_norm))
    return (a > b) - (b > a)


OTRO_PAIS = (r"\b(cuba|venezuela|mexic\w+|argentin\w+|peru|chile|ecuador|bolivia|panama|espana|canada|uruguay|paraguay|brasil|brazil|india|china|rusia|ucrania|alemania|francia|"
             r"reino unido|japon|estados unidos|ee\.? ?uu\.?|milei|sheinbaum|fujimori|maduro)\b")


def temas_de(titulo: str) -> list[tuple[dict[str, Any], int]]:
    """Temas de mercado que toca el titular, cada uno con su dirección. Máximo dos (el titular más cargado no debe volverse una lista de todo)."""
    t = N.norm(titulo)
    ajeno = bool(re.search(OTRO_PAIS, t)) and not re.search(r"\bcolombia\w*\b", t)      # "Cuba enfrenta su crisis energética" no es un tema de Colombia
    hallados = [(tema, sum(1 for p in tema["patrones"] if re.search(p, t))) for tema in TEMAS if not (ajeno and tema["clave"].endswith("_colombia"))]
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
        for ef in tema["efectos"]:
            signo, nombre, tickers, porque = ef[:4]
            al_bajar = ef[4] if len(ef) > 4 else -signo                                # casi siempre es lo contrario; algunos sectores ganan sólo en un sentido
            tk = ok([t for t in tickers if t not in empresas])
            if tk:
                efecto = signo if d > 0 else (al_bajar if d < 0 else 0)
                base = signo if signo else (-1 if al_bajar > 0 else (1 if al_bajar < 0 else 0))      # para decir "gana si … sube/baja" cuando el titular no trae dirección
                sectores.append(dict(nombre=nombre, signo=efecto, base=base, mixto=(efecto == 0 and d != 0), tickers=tk, porque=porque, directa=False,
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
                factor=(temas[0][0].get("factor") if temas else None), que=(temas[0][0]["que"] if temas else None),
                explica=(EXPLICA.get(temas[0][0]["clave"], "") if temas else EXPLICA_EMPRESA))


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
        self.en_ingles = {f["nombre"] for f in self.m.get("rss", []) if f.get("idioma") == "en"} | {"Yahoo Finance", "Yahoo Finance (por acción)"}
        self._trad: dict[str, str] = {}

    def traducir(self, texto: str) -> str | None:
        """Inglés → español. Primero el traductor web de Google (el mismo punto público que usan las librerías de traducción de Python); si falla, MyMemory.
        Devuelve None si ninguno responde: el aviso sale entonces en inglés y lo dice. Lo ya traducido se recuerda."""
        t = " ".join((texto or "").split())
        if not t or not self.m.get("traducir", True):
            return None
        if t in self._trad:
            return self._trad[t] or None
        salida = ""
        try:
            r = self.base._get("https://translate.googleapis.com/translate_a/single", {"client": "gtx", "sl": "en", "tl": "es", "dt": "t", "q": t[:900]}, condicional=False)
            salida = "".join(seg[0] for seg in r.json()[0] if seg and seg[0]).strip()
        except Exception:                                                              # noqa: BLE001
            salida = ""
        if not salida:
            try:
                r = self.base._get("https://api.mymemory.translated.net/get", {"q": t[:480], "langpair": "en|es"}, condicional=False)
                j = r.json()
                cand = j["responseData"]["translatedText"]
                salida = cand.strip() if j.get("responseStatus") == 200 and "MYMEMORY WARNING" not in cand.upper() else ""
            except Exception:                                                          # noqa: BLE001
                salida = ""
        if len(self._trad) > 600:
            self._trad.clear()
        self._trad[t] = salida
        return salida or None

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
        g = self.m.get("google") or {}
        consultas = g.get("consultas") or []
        if g.get("activo") and consultas and completo:
            tareas.append(("Google Noticias (temas de Colombia)", lambda: [i for q in consultas for i in self.base.google(q)]))
        elif g.get("activo") and consultas and toca(g):
            q = consultas[(self.ronda // max(int(g.get("cada_rondas", 1)), 1)) % len(consultas)]
            tareas.append(("Google Noticias (temas de Colombia)", lambda: self.base.google(q)))
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


def resumen_limpio(texto: str, n: int = 260) -> str:
    """El resumen listo para leer, o '' si lo que trae la fuente no es un resumen sino una ficha pegada sin espacios ("StockCurrent Price: $608. 10Target Price…"),
    una lista de cifras o el mismo titular repetido."""
    t = " ".join((texto or "").split())
    if len(t) < 40:
        return ""
    pegadas = len(re.findall(r"[a-záéíóú][A-ZÁÉÍÓÚ]", t[:400]))                         # palabras pegadas: señal de texto extraído de una tabla
    cifras = sum(ch.isdigit() or ch in "$%~" for ch in t[:400]) / max(len(t[:400]), 1)
    if pegadas >= 3 or cifras > 0.14 or re.search(r"key stats|read more|click here|sign up|subscribe|cookies", t[:200], re.I):
        return ""
    return recortar(t, n)


def recortar(texto: str, n: int = 320) -> str:
    """El resumen en una o dos frases completas (hasta `n` letras)."""
    t = " ".join((texto or "").split())
    if len(t) <= n:
        return t
    corte = max(t.rfind(". ", 0, n), t.rfind("; ", 0, n))
    return (t[:corte + 1] if corte > n * 0.5 else t[:n].rsplit(" ", 1)[0] + "…").strip()


# ------------------------------------------------------------------ ¿qué tan importante es?
FUERTE = (r"\b(se dispara\w*|disparo|se desploma\w*|desplomo|se hunde\w*|hundio|colapsa\w*|crash|historic\w+|record|maximo historico|minimo historico|guerra|ataque\w*|bombarde\w+|invasion|"
          r"emergencia|crisis|default|quiebra|intervencion|interviene|expropia\w*|estado de excepcion|shock|panico|sorpres\w+|inesperad\w+|"
          r"soar\w*|plunge\w*|surge\w*|tumble\w*|slump\w*|rout|selloff|war|attack\w*|strike\w*|record high|record low|all-time|emergency|shock\w*|surprise\w*|unexpected\w*)\b")
MODERADA = (r"\b(sube|suben|subio|cae|caen|cayo|baja|bajan|alza|caida|repunt\w+|retroced\w+|avanz\w+|aceler\w+|recort\w+|advierte|advirtio|alerta\w*|riesgo\w*|tension\w*|"
            r"rise\w*|rose|fall\w*|fell|drop\w*|gain\w*|jump\w*|slide\w*|climb\w*|warn\w*|risk\w*|cut\w*|hike\w*)\b")
OFICIAL = (r"\b(decreto|decreta|aprueba|aprobo|aprobaron|sanciona|firma|firmo|anuncia|anuncio|decide|decidio|fija|fijo|sube las tasas|baja las tasas|recorta las tasas|mantiene las tasas|"
           r"dato de inflacion|\bipc\b|\bcpi\b|\bpib\b|resultados electorales|gana las elecciones|primera vuelta|segunda vuelta|"
           r"approves?|signs?|announces?|decides?|raises rates|cuts rates|holds rates|election results?)\b")
NIVELES = {"alto": ("🚨🚨🚨 NOTICIA MUY IMPORTANTE 🚨🚨🚨", "🔴🔴🔴 ALTA"), "medio": ("⚠️ NOTICIA IMPORTANTE", "🟠🟠⚪ MEDIA"), "bajo": ("📰 NOTICIA DE MERCADO", "⚪ baja (informativa)")}


def importancia(a: dict[str, Any], titulo: str, z_factor: float | None = None, n_medios: int = 1, tenidos: set[str] | None = None, m: dict[str, Any] | None = None) -> dict[str, Any]:
    """Qué tan importante es la noticia para el mercado, de 'bajo' (informativa) a 'alto' (muy importante), con los motivos en palabras.
    Puntos: el tono del titular (2 si habla de un salto, una crisis o un ataque; 1 si de una subida o caída normal) + 1 si dice hacia dónde va
    + lo que YA se está moviendo el mercado de ese tema (2 si es 2 veces lo normal o más; 1 si es 1 vez) + cuántos medios la traen (2 si son 3 o más; 1 si son 2)
    + 1 si es una decisión o un dato oficial + 1 si le pega a una acción que tienes. Con 5 o más es muy importante; con 3 o 4, importante."""
    t = N.norm(titulo)
    cortes = (m or {}).get("importancia") or {}
    puntos, motivos = 0, []
    if re.search(FUERTE, t):
        puntos += 2
        motivos.append("el titular habla de un movimiento o un hecho fuerte")
    elif re.search(MODERADA, t):
        puntos += 1
    directa = bool(a.get("direccion")) and bool(a.get("beneficiadas") or a.get("afectadas"))
    if directa:
        puntos += 1
        motivos.append("dice hacia dónde empuja")
    if z_factor is not None and abs(z_factor) >= 2:
        puntos += 2
        motivos.append(f"el mercado de ese tema ya se mueve {abs(z_factor):.1f} veces lo normal".replace(".", ","))
    elif z_factor is not None and abs(z_factor) >= 1:
        puntos += 1
        motivos.append("el mercado de ese tema ya se está moviendo")
    if n_medios >= 3:
        puntos += 2
        motivos.append(f"la traen {n_medios} medios")
    elif n_medios == 2:
        puntos += 1
        motivos.append("la traen 2 medios")
    if re.search(OFICIAL, t):
        puntos += 1
        motivos.append("es una decisión o un dato oficial")
    mias = [x for x in (a.get("beneficiadas", []) + a.get("afectadas", []) + a.get("expuestas", [])) if x in (tenidos or set())]
    if mias:
        puntos += 1
        motivos.insert(0, "le pega a " + ", ".join(mias[:3]) + ", que tienes")
    nivel = "alto" if puntos >= cortes.get("alto", 5) else ("medio" if puntos >= cortes.get("medio", 3) else "bajo")
    if nivel != "bajo" and not directa and not mias:
        nivel = "medio" if nivel == "alto" else "bajo"                                 # sin efecto directo (no dice a quién ni hacia dónde) baja un escalón
    return dict(nivel=nivel, puntos=puntos, motivos=motivos, titulo=NIVELES[nivel][0], etiqueta=NIVELES[nivel][1])


def medios_que_la_traen(item: N.Item, items: list[N.Item]) -> int:
    """Cuántos medios distintos traen esta misma noticia entre los titulares leídos (parecido de palabras o misma cifra)."""
    f = N.firma(item.titulo)
    medios = {item.fuente}
    for o in items:
        if o.fuente in medios:
            continue
        g = N.firma(o.titulo)
        if N.parecido(f["pal"], g["pal"]) >= 0.4 or (set(f["cif"]) & set(g["cif"]) and N.parecido(f["pal"], g["pal"]) >= 0.2):
            medios.add(o.fuente)
    return len(medios)


# ------------------------------------------------------------------ ¿ya se avisó esta historia?
def _nombres_de(empresas: list[str]) -> set[str]:
    """Las palabras con que se nombra a esas empresas (para no tomar 'Nvidia' como la pista de que dos noticias de Nvidia son la misma)."""
    out: set[str] = set()
    for t, patrones, _, _ in GLOBALES:
        if t in empresas:
            out |= set(re.findall(r"[a-z]{4,}", " ".join(patrones)))
    return out


def repetida(f: dict[str, list[str]], sujetos: set[str], direccion: int, historias: list[dict[str, Any]], ahora_utc: dt.datetime, m: dict[str, Any],
             propias: set[str] | None = None) -> str | None:
    """Por qué esta noticia es la MISMA que una ya avisada (o None si es nueva). Reglas, de la más segura a la más amplia:
      1. comparte la mitad de sus palabras con una ya avisada (la misma nota en otro medio);
      2. es del mismo tema o empresa y comparte una CIFRA ("5.000 millones", "4.160");
      3. es del mismo tema y nombra a los mismos protagonistas (Tencent, Zelenski, Riad) en las últimas 12 horas;
      4. es del mismo tema y comparte un tercio de sus palabras;
      5. es del mismo tema y va en la misma dirección (o no dice dirección) dentro del tiempo de enfriamiento: el mismo movimiento contado otra vez.
    Lo único que pasa dentro de ese tiempo es un GIRO del mismo tema (de subir a bajar o al revés), que sí es una noticia distinta."""
    ents = set(f["ent"]) - set(propias or ())
    for h in historias:
        edad_min = (ahora_utc - dt.datetime.fromisoformat(h["ts"])).total_seconds() / 60
        sim = N.parecido(f["pal"], h["pal"])
        if sim >= m["parecido_repetida"]:
            return "mismas palabras"
        if "*" not in h["sujetos"] and not (sujetos & set(h["sujetos"])):
            continue
        if set(f["cif"]) & set(h["cif"]):
            return "misma cifra"
        if edad_min <= 12 * 60 and ents & (set(h["ent"]) - set(propias or ())):
            return "mismos protagonistas"
        if sim >= m.get("parecido_mismo_tema", 0.34):
            return "parecida, mismo tema"
        if (h["dir"] == direccion or 0 in (h["dir"], direccion)) and edad_min < m["enfriamiento_tema_min"]:      # sin dirección clara cuenta como la misma: sólo un giro es noticia nueva
            return "mismo tema y dirección hace poco"
    return None


# ------------------------------------------------------------------ una ronda
def ronda(lector: Any, mem: Any, ahora: dt.datetime, cfg: dict[str, Any], locales: Callable[[N.Item], bool] | None = None,
          extra: list[N.Item] | None = None) -> tuple[list[dict[str, Any]], dict[str, bool]]:
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
    items = list(items) + list(extra or [])                                            # `extra`: titulares que ya leyó el vigía de empresas (Portafolio, Semana, El Tiempo, Google…)
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
    desde = (ahora_utc - dt.timedelta(hours=m.get("memoria_repetidas_h", 24))).isoformat(timespec="seconds")
    if "historias" not in d:                                                           # primera vez con esta memoria: lo que ya se avisó antes tampoco se repite
        d["historias"] = [dict(ts=x["ts"], sujetos=["*"], dir=x.get("direccion", 0), **{k: v for k, v in N.firma(x["titulo"]).items() if k in ("pal", "cif", "ent")})
                          for x in d.get("ultimas", [])]
    d["historias"] = [x for x in d["historias"] if x["ts"] >= desde]                   # lo ya avisado, para no repetirlo
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
        if a is None or (locales and locales(it)):
            continue                                                                   # la noticia de una empresa colombiana ya la avisa el vigía de empresas: aquí sería repetirla
        titulo_es = it.titulo
        if it.fuente in getattr(lector, "en_ingles", ()) and hasattr(lector, "traducir"):
            titulo_es = lector.traducir(it.titulo) or it.titulo                        # se compara en español: la misma noticia llega en los dos idiomas
        f = N.firma(titulo_es)
        sujetos = set(a["temas"]) | {"emp:" + e for e in a["empresas"]}
        if repetida(f, sujetos, a["direccion"], d["historias"], ahora_utc, m, propias=_nombres_de(a["empresas"])):
            continue                                                                   # la misma historia ya se avisó: no se repite
        d["historias"].append(dict(ts=marca, sujetos=sorted(sujetos), dir=a["direccion"], pal=f["pal"], cif=f["cif"], ent=f["ent"]))
        a["item"], a["titulo_es"] = it, titulo_es
        a["n_medios"] = medios_que_la_traen(it, items)
        out.append(a)
        d["ultimas"].append(dict(ts=marca, titulo=it.titulo[:220], fuente=it.fuente, url=it.url, nombres=a["nombres"], empresas=a["empresas"], direccion=a["direccion"],
                                 beneficiadas=a["beneficiadas"][:8], afectadas=a["afectadas"][:8], expuestas=a["expuestas"][:8]))
    d["ultimas"] = d["ultimas"][-30:]
    lim = (ahora_utc - dt.timedelta(days=3)).isoformat()
    d["vistos"] = {k: v for k, v in d["vistos"].items() if v >= lim}
    return out, salud
