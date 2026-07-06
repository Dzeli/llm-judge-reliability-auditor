"""HTML renderer utilities for the LLM Judge Reliability Auditor v4 UI.
Encapsulates all HTML template string generation, CSS color mapping, and visual components.
"""
from __future__ import annotations
import re
from models.report import AuditReport

# ── SVG and CSS Icons ──────────────────────────────────────────────────────────
INFO_ICON = "ℹ"
WARN_ICON = "⚠️"
CHECK_ICON = "✓"
CROSS_ICON = "✕"

EMPTY_STATE_HTML = """
<div class="empty-state">
  <div class="empty-icon">🔬</div>
  <div class="empty-title">Ready to Audit</div>
  <div class="empty-subtitle">
    Configure the parameters on the right and select your audit mode.<br>
    Click <strong>Run Audit</strong> to generate reliability profiles.
  </div>
  <div class="empty-chips">
    <span class="e-chip">📊 Position Bias</span>
    <span class="e-chip">✍️ Style & Verbosity</span>
    <span class="e-chip">🔄 Consistency Profiling</span>
    <span class="e-chip">📋 Rubric Sensitivity</span>
    <span class="e-chip">🔗 Reference Distortion</span>
  </div>
</div>
"""

def get_score_color(score: float) -> str:
    """Returns semantic hex color code based on score range."""
    if score >= 90:
        return "#4ade80"  # Green
    if score >= 75:
        return "#2dd4bf"  # Teal/Cyan
    if score >= 60:
        return "#fbbf24"  # Amber
    if score >= 45:
        return "#fb923c"  # Orange
    return "#f87171"      # Red

def get_bar_color(verdict: str) -> str:
    """Returns semantic color code based on categorical verdict."""
    return {
        "LOW": "#4ade80",
        "MEDIUM": "#fbbf24",
        "HIGH": "#f87171"
    }.get(verdict, "#7c3aed")

def get_verdict_chip(verdict: str) -> str:
    """Generates an HTML chip with custom styling for a given verdict class."""
    label = {"LOW": "✓ LOW", "MEDIUM": "⚠️ MED", "HIGH": "✕ HIGH"}.get(verdict, verdict)
    return f'<span class="v-chip v-{verdict}">{label}</span>'

def clean_evidence(text: str) -> tuple[str, str]:
    """Cleans evidence text strings and determines CSS classes for output formatting."""
    # Strip internal variant IDs (e.g. '(style::polished)')
    text = re.sub(r'\s*\([a-zA-Z0-9_:.\-]+::[a-zA-Z0-9_.\-]+\)', '', text)
    text = re.sub(r'^[a-zA-Z0-9_\-]+:\s*', '', text)
    
    css_class = ""
    low_text = text.lower()
    if "invariance failure" in low_text or "verdict changed" in low_text:
        css_class = "inv"
    elif "accuracy failure" in low_text or "wrong winner" in low_text:
        css_class = "acc"
        
    return text, css_class

def format_percentage(value: float | None) -> str:
    """Formats float/decimal values to human-friendly percentage strings."""
    return "—" if value is None else f"{value * 100:.0f}%"

def format_delta(value: float | None) -> str:
    """Formats float/decimal change delta to percentage point string."""
    return "—" if value is None else f"{value * 100:+.0f} pp"

def build_status_bar(message: str) -> str:
    """Builds an animated status progress bar with a pulsing indicator dot."""
    if not message:
        return ""
    return (
        f'<div class="status-bar">'
        f'  <div class="pulse-dot"></div>'
        f'  <span>{message}</span>'
        f'</div>'
    )

def build_model_card(model: str, report: AuditReport) -> str:
    """Generates the HTML card wrapper for a single audited model's results."""
    ms = report.metric_summary
    score = report.reliability_score
    grade = report.grade or "F"
    score_color = get_score_color(score)

    # 1. Score Row with radial progress gauge
    score_row = f"""
    <div class="score-row">
      <div class="gauge-wrap">
        <div class="gauge-arc" style="background: conic-gradient({score_color} {score * 3.6}deg, #16122d {score * 3.6}deg);">
          <div class="gauge-hole">
            <span class="g-score-num">{score:.0f}</span>
            <span class="g-score-den">/ 100</span>
          </div>
        </div>
      </div>
      <div class="score-meta">
        <span class="conf-pill c-{report.confidence_level}">{report.confidence_level.capitalize()} Confidence</span>
        <div class="score-detail">
          Reliability Grade: <strong style="color: {score_color};">{grade}</strong><br>
          Based on {report.n_cases} scenarios ({report.total_api_calls} queries)
        </div>
      </div>
    </div>"""

    # 2. Bias Dimension Progress Bars
    dim_rows = []
    for test_type, res in report.test_results.items():
        q = min(res.quality_score * 100, 100) if res.quality_score is not None else 50.0
        bc = get_bar_color(res.verdict)
        dim_rows.append(f"""
        <div class="dim-row">
          <span class="dim-lbl">{test_type.value}</span>
          <div class="dim-bar-bg"><div class="dim-bar-fill" style="width:{q:.0f}%;background:{bc};"></div></div>
          <span class="dim-pct">{q:.0f}%</span>
          {get_verdict_chip(res.verdict)}
        </div>""")

    dims_html = f"""
    <div class="dims-header">Bias Dimensions</div>
    {"".join(dim_rows)}"""

    # 3. Supplemental Metrics Strip
    strip_parts = []
    if ms.robust_accuracy is not None:
        strip_parts.append(f'<span class="metric-item">Robust Accuracy: <strong>{format_percentage(ms.robust_accuracy)}</strong></span>')
    if ms.stability is not None:
        strip_parts.append(f'<span class="metric-item">Stability: <strong>{format_percentage(ms.stability)}</strong></span>')
    if ms.consistency_quality is not None:
        strip_parts.append(f'<span class="metric-item">Consistency Quality: <strong>{format_percentage(ms.consistency_quality)}</strong></span>')
    if ms.consistency_profile:
        strip_parts.append(f'<span class="metric-item">Profile: <strong>{ms.consistency_profile.replace("_", " ")}</strong></span>')
    if ms.reference_helpfulness is not None:
        lbl = ms.reference_helpfulness_label or ""
        strip_parts.append(f'<span class="metric-item">Ref Delta: <strong>{format_delta(ms.reference_helpfulness)}</strong> ({lbl})</span>')
    
    metric_strip = f'<div class="metric-strip">{"".join(strip_parts)}</div>' if strip_parts else ""

    # 4. Critical Warnings Section
    warn_html = ""
    if report.warnings:
        items = "".join(
            f'<div class="warn-row"><span class="warn-ico">{WARN_ICON}</span>{w}</div>'
            for w in report.warnings
        )
        warn_html = f'<div class="warns-box">{items}</div>'

    # 5. Diagnostic Evidence Collapsible
    ev_id = "ev_" + re.sub(r"[^a-zA-Z0-9]", "_", model)
    ev_groups = []
    for tt, res in report.test_results.items():
        if not res.evidence:
            continue
        items_html = []
        for ev in res.evidence[:6]:
            clean, css = clean_evidence(ev)
            if clean:
                items_html.append(f'<div class="ev-item {css}">{clean}</div>')
        if items_html:
            ev_groups.append(
                f'<div class="ev-group">'
                f'  <div class="ev-group-label">{tt.value}</div>'
                f'  {"".join(items_html)}'
                f'</div>'
            )

    evidence_html = ""
    if ev_groups:
        toggle_js = (
            f"var c=document.getElementById('{ev_id}');"
            f"c.classList.toggle('open');"
            f"this.querySelector('.ev-arr').textContent=c.classList.contains('open')?'▲':'▼';"
        )
        evidence_html = f"""
        <div class="evidence-box">
          <div class="ev-toggle" onclick="{toggle_js}">
            <span class="ev-arr">▼</span> Show detailed evidence
          </div>
          <div id="{ev_id}" class="ev-content">{"".join(ev_groups)}</div>
        </div>"""

    return f"""
    <div class="model-card">
      <div class="card-head">
        <div class="card-model">{model}</div>
        <div class="grade-box g-{grade}">{grade}</div>
      </div>
      {score_row}
      {dims_html}
      {metric_strip}
      {warn_html}
      {evidence_html}
    </div>"""

def build_results_html(reports: dict[str, AuditReport], errors: dict[str, str]) -> str:
    """Combines all individual model cards and errors into the primary results grid view."""
    if not reports and not errors:
        return EMPTY_STATE_HTML
        
    n = len(reports) + len(errors)
    cards = [build_model_card(m, r) for m, r in reports.items()]
    
    for model, err in errors.items():
        safe_err = err.replace("<", "&lt;").replace(">", "&gt;")
        cards.append(
            f'<div class="err-card">'
            f'  <div class="err-model">{model}</div>'
            f'  <div class="err-msg">{WARN_ICON} {safe_err}</div>'
            f'</div>'
        )
        
    label = "model" if n == 1 else "models"
    return (
        f'<div class="results-wrap">'
        f'  <div class="results-header">'
        f'    <div class="results-title">Audit Results</div>'
        f'    <div class="results-badge">{n} {label}</div>'
        f'  </div>'
        f'  <div class="results-grid">{"".join(cards)}</div>'
        f'</div>'
    )
