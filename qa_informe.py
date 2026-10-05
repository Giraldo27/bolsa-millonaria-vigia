"""Revisión visual del informe con Playwright (solo para QA): capturas y comprobaciones automáticas."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HTML = Path(__file__).parent / "reports" / "informe_backtest.html"
OUT = Path(__file__).parent / "reports" / "qa"
OUT.mkdir(exist_ok=True)

CHECKS = """() => {
  const res = {};
  res.scrollW = document.documentElement.scrollWidth; res.innerW = window.innerWidth;
  res.plots = [...document.querySelectorAll('.js-plotly-plot')].map(d => ({id: d.id, svg: d.querySelectorAll('svg.main-svg').length,
     traces: (d.data||[]).length, w: d.getBoundingClientRect().width, h: d.getBoundingClientRect().height}));
  res.overflowTables = [...document.querySelectorAll('.tablewrap')].filter(t => t.scrollWidth > t.clientWidth + 2).length;
  res.wideEls = [...document.querySelectorAll('main *')].filter(e => e.getBoundingClientRect().right > window.innerWidth + 2 && !e.closest('.tablewrap') && !e.closest('nav')).slice(0, 8).map(e => e.tagName + '.' + (e.className||'') + '#' + (e.id||''));
  res.sections = [...document.querySelectorAll('section')].map(s => s.id + ':' + Math.round(s.getBoundingClientRect().height));
  res.rows = document.querySelectorAll('#tt tbody tr').length;
  res.english = [...document.body.innerText.matchAll(/\\b(Loading|Error|undefined|NaN|null|Traceback)\\b/g)].map(m => m[0]);
  return res;
}"""


def main() -> int:
    errs: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, vp in (("escritorio", dict(width=1366, height=900)), ("celular", dict(width=390, height=844))):
            ctx = b.new_context(viewport=vp, device_scale_factor=1, is_mobile=(name == "celular"))
            pg = ctx.new_page()
            pg.on("console", lambda m: errs.append(f"[{name}] consola {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errs.append(f"[{name}] pageerror: {e}"))
            pg.goto(HTML.as_uri())
            pg.wait_for_timeout(2500)
            r = pg.evaluate(CHECKS)
            print(f"\n=== {name} ===  ancho documento {r['scrollW']} / ventana {r['innerW']} · tablas con scroll horizontal: {r['overflowTables']} · filas bitácora: {r['rows']}")
            print("  gráficos:", [(x['id'], x['svg'], x['traces'], int(x['w']), int(x['h'])) for x in r['plots']])
            print("  elementos que se salen:", r['wideEls'])
            print("  texto sospechoso:", r['english'])
            print("  alto de secciones:", r['sections'])
            pg.screenshot(path=str(OUT / f"{name}_completo.png"), full_page=True)
            # capturas por sección
            for sid in [f"s{i}" for i in range(1, 16)]:
                el = pg.query_selector(f"#{sid}")
                if el:
                    el.screenshot(path=str(OUT / f"{name}_{sid}.png"))
            ctx.close()
        b.close()
    print("\nerrores de consola:", errs or "ninguno")
    return 0


if __name__ == "__main__":
    sys.exit(main())
