"""Self-contained HTML report for an optimizer run.

No external assets: inline CSS and an inline SVG Pareto chart, so the file can be opened, emailed or
archived as one artifact. The recommendation section is the same measured data the text report
prints; the HTML only changes presentation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json

from .report import OptimizerResult, _ctx, ok_rows
from .usecases import OBJECTIVE_LABELS

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       padding: 2rem; max-width: 1080px; margin-inline: auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #8884; padding-bottom: .25rem; }
.sub { opacity: .7; margin: 0 0 1rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
.card { border: 1px solid #8884; border-radius: 8px; padding: 1rem; }
.card h3 { margin: 0 0 .5rem; font-size: .95rem; text-transform: uppercase; letter-spacing: .04em; opacity: .8; }
.metric { font-size: 1.4rem; font-weight: 600; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; }
pre { background: #8881; padding: .6rem; border-radius: 6px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #8883; }
th { font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { display: inline-block; height: .7em; background: #4c8bf5; border-radius: 2px; vertical-align: middle; }
.ok { color: #2a7a2a; } .bad { color: #b3261e; }
.detail { opacity: .7; }
details { margin-top: 1rem; }
"""


def render_html(result: OptimizerResult, *, title: str | None = None) -> str:
    use_case = result.use_case
    fp = result.fingerprint
    heading = title or f"ninfer-optimizer — {use_case.name}"
    rows = result.rows
    ok = ok_rows(rows)
    parts: list[str] = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        f"<title>{escape(heading)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{escape(heading)}</h1>",
        f"<p class=sub>{escape(use_case.summary)} &middot; "
        f"{escape(fp['model']['name'])} on {escape(fp['hardware']['gpu'])} &middot; "
        f"objective {escape(OBJECTIVE_LABELS[use_case.objective])} &middot; "
        f"{len(ok)}/{len(rows)} configs produced a number &middot; "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>",
        "<div class=cards>",
        _card("Fastest", result.fastest, use_case),
        _card(f"Balanced (ctx &ge; {use_case.ctx_floor})", result.balanced, use_case),
        _card("Max context", result.max_context, use_case),
        "</div>",
    ]

    if result.confirmation is not None:
        predicted, row = result.confirmation
        error = abs(row.objective - predicted) / predicted * 100.0 if predicted > 0 else 0.0
        verdict = (
            "small gap: the additive model held"
            if error <= 5.0
            else "large gap: interactions or drift — trust the Pareto pick"
        )
        parts += [
            "<h2>Confirmation run</h2>",
            f"<p>Predicted <b>{predicted:.2f}</b>, measured <b>{row.objective:.2f}</b> "
            f"(status {escape(row.status)}), error <b>{error:.1f}%</b> — {escape(verdict)}.</p>",
        ]

    parts += ["<h2>Pareto frontier (context vs objective)</h2>", _pareto_svg(result)]
    parts += ["<h2>Main effects</h2>", _effects_table(result)]
    parts += ["<h2>All runs</h2>", _rows_table(rows, use_case)]
    parts += [
        "<h2>Fingerprint</h2><details><summary>machine, model and design</summary>",
        f"<pre>{escape(json.dumps(fp, indent=2))}</pre></details>",
        "</body></html>",
    ]
    return "\n".join(parts)


def _card(label: str, row, use_case) -> str:
    if row is None:
        return f"<div class=card><h3>{escape(label)}</h3><p class=detail>(none)</p></div>"
    extra = f" &middot; ttft {row.ttft_ms:.0f} ms" if row.ttft_ms > 0 else ""
    command = " ".join(_suggested(row, use_case))
    return (
        f"<div class=card><h3>{escape(label)}</h3>"
        f"<div class=metric>{row.objective:.2f}</div>"
        f"<p class=detail>pp {row.pp_tps:.1f} &middot; tg {row.tg_tps:.1f}{extra}</p>"
        f"<pre>{escape(command)}</pre></div>"
    )


def _suggested(row, use_case) -> list[str]:
    if use_case.driver == "server":
        return ["ninfer-serve", "<artifact>", "--max-concurrency", str(use_case.concurrency),
                *row.setting.render_serve()]
    return ["ninfer", "<artifact>", *row.setting.render_bench()]


def _pareto_svg(result: OptimizerResult) -> str:
    usable = ok_rows(result.rows)
    if not usable:
        return "<p class=detail>(no measured configurations)</p>"
    width, height, pad = 760.0, 300.0, 44.0
    xs = [max(1, _ctx(r.setting.values)) for r in usable]
    ys = [r.objective for r in usable]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span_x = (x1 - x0) or 1
    span_y = (y1 - y0) or 1

    def px(x: int) -> float:
        return pad + (x - x0) / span_x * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y0) / span_y * (height - 2 * pad)

    out = [f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" role="img">',
           f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#8886"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#8886"/>',
           f'<text x="{width/2:.0f}" y="{height-8:.0f}" text-anchor="middle" font-size="12">'
           f'context depth</text>',
           f'<text x="12" y="{height/2:.0f}" text-anchor="middle" font-size="12" '
           f'transform="rotate(-90 12 {height/2:.0f})">objective</text>']
    for row in usable:
        out.append(
            f'<circle cx="{px(max(1, _ctx(row.setting.values))):.1f}" cy="{py(row.objective):.1f}" '
            f'r="3" fill="#8888"/>'
        )
    frontier = sorted(result.frontier, key=lambda r: _ctx(r.setting.values))
    if len(frontier) >= 2:
        points = " ".join(
            f"{px(max(1, _ctx(r.setting.values))):.1f},{py(r.objective):.1f}" for r in frontier
        )
        out.append(f'<polyline points="{points}" fill="none" stroke="#4c8bf5" stroke-width="2"/>')
    for row in frontier:
        out.append(
            f'<circle cx="{px(max(1, _ctx(row.setting.values))):.1f}" cy="{py(row.objective):.1f}" '
            f'r="5" fill="#4c8bf5"/>'
        )
    out.append("</svg>")
    return "\n".join(out)


def _effects_table(result: OptimizerResult) -> str:
    if not result.effects:
        return "<p class=detail>(no main effects)</p>"
    top = max((e.range for e in result.effects), default=0.0) or 1.0
    out = ["<table><thead><tr><th>Factor</th><th class=num>Range</th><th>Level means</th>"
           "<th>Best</th></tr></thead><tbody>"]
    for effect in result.effects:
        width = 100.0 * effect.range / top
        out.append(
            f"<tr><td>{escape(effect.name)}</td><td class=num>{effect.range:.3f}</td>"
            f"<td><span class=bar style='width:{width:.1f}%'></span> "
            + ", ".join(
                f"{escape(level)}={mean:.2f}" for level, mean in zip(effect.levels, effect.means)
            )
            + f"</td><td>{escape(effect.best_level())}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _rows_table(rows, use_case) -> str:
    out = ["<table><thead><tr><th>Status</th><th class=num>Objective</th><th class=num>pp t/s</th>"
           "<th class=num>tg t/s</th><th class=num>TTFT ms</th><th class=num>Temp C</th>"
           "<th>Setting</th></tr></thead><tbody>"]
    for row in rows:
        cls = "ok" if row.ok else "bad"
        ttft = f"{row.ttft_ms:.0f}" if row.ttft_ms > 0 else ""
        temp = f"{row.temp_c:.0f}" if row.temp_c > 0 else ""
        out.append(
            f"<tr><td class={cls}>{escape(row.status)}</td>"
            f"<td class=num>{row.objective:.2f}</td><td class=num>{row.pp_tps:.1f}</td>"
            f"<td class=num>{row.tg_tps:.1f}</td><td class=num>{ttft}</td>"
            f"<td class=num>{temp}</td><td>{escape(row.setting.label())}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)
