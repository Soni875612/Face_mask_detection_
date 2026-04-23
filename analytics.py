"""
analytics.py — Detection Analytics, Statistics & Report Generator
Author   : [Your Name]
Project  : AI-Based Face Mask Detection System
Purpose  : Aggregates detection results across sessions, computes compliance
           statistics, generates visual dashboards, and exports professional
           PDF-style HTML reports — suitable for institutional submission.
"""

import os
import json
import time
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────
@dataclass
class SessionStats:
    session_id       : str
    start_time       : str
    end_time         : str
    total_frames     : int = 0
    total_faces      : int = 0
    with_mask        : int = 0
    without_mask     : int = 0
    incorrectly_worn : int = 0
    uncertain        : int = 0
    alerts_triggered : int = 0
    avg_conf_masked  : float = 0.0
    avg_conf_unmasked: float = 0.0
    compliance_pct   : float = 0.0

    def compute(self):
        detected = self.with_mask + self.without_mask + self.incorrectly_worn
        if detected > 0:
            self.compliance_pct = round(self.with_mask / detected * 100, 2)


# ─────────────────────────────────────────────
# Analytics Engine
# ─────────────────────────────────────────────
class AnalyticsEngine:
    """
    Tracks real-time detection events, computes statistics,
    and generates visual + textual reports.

    Usage:
        engine = AnalyticsEngine(output_dir="analytics_out")
        engine.start_session()
        for each frame result:
            engine.record(frame_result)
        engine.end_session()
        engine.generate_report()
    """

    def __init__(self, output_dir: str = "analytics_output"):
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session    : Optional[SessionStats] = None
        self._history    : List[dict] = []          # Per-frame history
        self._conf_masked  : List[float] = []
        self._conf_unmasked: List[float] = []
        self._timeline   : List[Dict]   = []
        self._all_sessions: List[SessionStats] = []

        # Load previous sessions if they exist
        sess_file = self.output_dir / "sessions.json"
        if sess_file.exists():
            with open(sess_file) as f:
                raw = json.load(f)
            self._all_sessions = [SessionStats(**s) for s in raw]

    # ── Session Lifecycle ─────────────────────
    def start_session(self, session_id: Optional[str] = None):
        sid = session_id or datetime.now().strftime("SES_%Y%m%d_%H%M%S")
        self._session       = SessionStats(
            session_id  = sid,
            start_time  = datetime.now().isoformat(),
            end_time    = "",
        )
        self._history.clear()
        self._conf_masked.clear()
        self._conf_unmasked.clear()
        self._timeline.clear()
        log.info(f"Session started: {sid}")

    def end_session(self):
        if not self._session:
            return
        self._session.end_time = datetime.now().isoformat()
        if self._conf_masked:
            self._session.avg_conf_masked   = float(np.mean(self._conf_masked))
        if self._conf_unmasked:
            self._session.avg_conf_unmasked = float(np.mean(self._conf_unmasked))
        self._session.compute()
        self._all_sessions.append(self._session)
        self._save_sessions()
        log.info(f"Session ended: {self._session.session_id} | "
                 f"Compliance: {self._session.compliance_pct:.1f}%")

    def _save_sessions(self):
        path = self.output_dir / "sessions.json"
        with open(path, "w") as f:
            json.dump([asdict(s) for s in self._all_sessions], f, indent=2)

    # ── Real-Time Recording ───────────────────
    def record(self, frame_result):
        """
        Accept a FrameResult object from detector.py and update statistics.
        Compatible with the FrameResult dataclass defined in detector.py.
        """
        if not self._session:
            log.warning("record() called before start_session()")
            return

        self._session.total_frames += 1
        self._session.total_faces  += frame_result.total_faces

        if frame_result.alert_triggered:
            self._session.alerts_triggered += 1

        snap = {
            "frame_id"    : frame_result.frame_id,
            "timestamp"   : time.time(),
            "total_faces" : frame_result.total_faces,
            "unmasked"    : frame_result.unmasked_count,
            "alert"       : frame_result.alert_triggered,
        }

        for det in frame_result.detections:
            label = det.label
            conf  = det.confidence
            if label == "WithMask":
                self._session.with_mask += 1
                self._conf_masked.append(conf)
            elif label == "WithoutMask":
                self._session.without_mask += 1
                self._conf_unmasked.append(conf)
            elif label == "MaskWornIncorrectly":
                self._session.incorrectly_worn += 1
            else:
                self._session.uncertain += 1

        self._history.append(snap)
        self._timeline.append(snap)

    # ── Statistics ────────────────────────────
    def current_stats(self) -> dict:
        if not self._session:
            return {}
        s = self._session
        detected = s.with_mask + s.without_mask + s.incorrectly_worn
        return {
            "Session ID"          : s.session_id,
            "Frames Processed"    : s.total_frames,
            "Total Faces"         : s.total_faces,
            "With Mask"           : s.with_mask,
            "Without Mask"        : s.without_mask,
            "Incorrectly Worn"    : s.incorrectly_worn,
            "Uncertain"           : s.uncertain,
            "Alerts Triggered"    : s.alerts_triggered,
            "Compliance %"        : f"{s.with_mask / detected * 100:.1f}%" if detected else "N/A",
            "Avg Conf (Masked)"   : f"{np.mean(self._conf_masked):.3f}" if self._conf_masked else "N/A",
            "Avg Conf (Unmasked)" : f"{np.mean(self._conf_unmasked):.3f}" if self._conf_unmasked else "N/A",
        }

    def multi_session_trend(self) -> dict:
        """Aggregated compliance across all recorded sessions."""
        return {
            s.session_id: s.compliance_pct
            for s in self._all_sessions
        }

    # ── Visualisation Dashboard ───────────────
    def generate_dashboard(self, title: str = "Face Mask Detection — Analytics Dashboard"):
        """
        Generates a multi-panel matplotlib figure with:
          1. Pie chart: label distribution
          2. Timeline: unmasked count per frame
          3. Alert frequency
          4. Compliance trend across sessions
          5. Confidence distribution histograms
        """
        if not self._session:
            log.warning("No active session to visualise.")
            return

        s    = self._session
        Path(self.output_dir / "plots").mkdir(exist_ok=True)

        fig  = plt.figure(figsize=(20, 14), facecolor="#0f1117")
        fig.suptitle(title, fontsize=18, color="white", fontweight="bold", y=0.98)
        gs   = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        label_color = "#e0e0e0"
        grid_color  = "#2a2d3a"

        # ── 1. Pie: Label Distribution ─────────
        ax1 = fig.add_subplot(gs[0, 0])
        counts = [s.with_mask, s.without_mask, s.incorrectly_worn, s.uncertain]
        labels = ["With Mask", "Without Mask", "Incorrectly\nWorn", "Uncertain"]
        colors = ["#00c864", "#e03030", "#ff9900", "#888888"]
        non_zero = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
        if non_zero:
            c_, l_, col_ = zip(*non_zero)
            ax1.pie(c_, labels=l_, colors=col_, autopct="%1.1f%%",
                    pctdistance=0.82, textprops={"color": label_color, "fontsize": 9},
                    wedgeprops={"linewidth": 1.5, "edgecolor": "#0f1117"})
        ax1.set_title("Label Distribution", color=label_color, fontweight="bold")
        ax1.set_facecolor("#1a1d26")

        # ── 2. Compliance Gauge ────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        detected = s.with_mask + s.without_mask + s.incorrectly_worn
        comp = s.with_mask / detected if detected else 0
        ax2.set_facecolor("#1a1d26")
        theta  = np.linspace(0, np.pi, 200)
        ax2.plot(np.cos(theta), np.sin(theta), color="#2a2d3a", linewidth=18)
        fill_theta = np.linspace(0, np.pi * comp, 200)
        bar_color  = "#00c864" if comp > 0.8 else "#ff9900" if comp > 0.5 else "#e03030"
        ax2.plot(np.cos(fill_theta), np.sin(fill_theta), color=bar_color, linewidth=18)
        ax2.text(0, 0.05, f"{comp * 100:.1f}%", ha="center", va="center",
                 fontsize=22, fontweight="bold", color=bar_color)
        ax2.text(0, -0.35, "Compliance Rate", ha="center", color=label_color, fontsize=10)
        ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-0.6, 1.2)
        ax2.axis("off"); ax2.set_title("Compliance Gauge", color=label_color, fontweight="bold")

        # ── 3. Timeline: Unmasked per Frame ────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor("#1a1d26")
        if self._timeline:
            frames = [t["frame_id"]  for t in self._timeline]
            unmasked = [t["unmasked"] for t in self._timeline]
            ax3.fill_between(frames, unmasked, alpha=0.35, color="#e03030")
            ax3.plot(frames, unmasked, color="#e03030", linewidth=1.5)
            ax3.axhline(np.mean(unmasked), color="#ff9900", linewidth=1,
                        linestyle="--", label=f"Mean: {np.mean(unmasked):.2f}")
            ax3.legend(facecolor="#1a1d26", labelcolor=label_color, fontsize=8)
        ax3.set_title("Unmasked Count / Frame", color=label_color, fontweight="bold")
        ax3.set_xlabel("Frame", color=label_color, fontsize=8)
        ax3.set_ylabel("Unmasked Faces", color=label_color, fontsize=8)
        ax3.tick_params(colors=label_color)
        ax3.grid(True, color=grid_color)

        # ── 4. Confidence — Masked ─────────────
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor("#1a1d26")
        if self._conf_masked:
            ax4.hist(self._conf_masked, bins=20, color="#00c864", alpha=0.85, edgecolor="#0f1117")
            ax4.axvline(np.mean(self._conf_masked), color="white", linestyle="--", linewidth=1.5,
                        label=f"μ={np.mean(self._conf_masked):.2f}")
            ax4.legend(facecolor="#1a1d26", labelcolor=label_color, fontsize=8)
        ax4.set_title("Confidence — With Mask", color=label_color, fontweight="bold")
        ax4.tick_params(colors=label_color)
        ax4.grid(True, color=grid_color)

        # ── 5. Confidence — Unmasked ───────────
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor("#1a1d26")
        if self._conf_unmasked:
            ax5.hist(self._conf_unmasked, bins=20, color="#e03030", alpha=0.85, edgecolor="#0f1117")
            ax5.axvline(np.mean(self._conf_unmasked), color="white", linestyle="--", linewidth=1.5,
                        label=f"μ={np.mean(self._conf_unmasked):.2f}")
            ax5.legend(facecolor="#1a1d26", labelcolor=label_color, fontsize=8)
        ax5.set_title("Confidence — Without Mask", color=label_color, fontweight="bold")
        ax5.tick_params(colors=label_color)
        ax5.grid(True, color=grid_color)

        # ── 6. Alerts per Frame Interval ───────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor("#1a1d26")
        if self._timeline:
            alerts = [1 if t["alert"] else 0 for t in self._timeline]
            window = 30
            rolling = np.convolve(alerts, np.ones(window) / window, mode="valid")
            ax6.plot(rolling, color="#ff9900", linewidth=2)
            ax6.fill_between(range(len(rolling)), rolling, alpha=0.3, color="#ff9900")
        ax6.set_title(f"Alert Rate (rolling {window}fr)", color=label_color, fontweight="bold")
        ax6.tick_params(colors=label_color); ax6.grid(True, color=grid_color)

        # ── 7. Multi-Session Trend ─────────────
        ax7 = fig.add_subplot(gs[2, :])
        ax7.set_facecolor("#1a1d26")
        if len(self._all_sessions) > 1:
            sess_ids = [s.session_id for s in self._all_sessions]
            comps    = [s.compliance_pct for s in self._all_sessions]
            bar_cols = ["#00c864" if c > 80 else "#ff9900" if c > 50 else "#e03030" for c in comps]
            ax7.bar(sess_ids, comps, color=bar_cols, edgecolor="#0f1117", width=0.6)
            ax7.axhline(80, color="#00c864", linestyle="--", linewidth=1, label="80% target")
            ax7.set_ylabel("Compliance %", color=label_color)
            ax7.legend(facecolor="#1a1d26", labelcolor=label_color)
        else:
            ax7.text(0.5, 0.5, "Run multiple sessions to see cross-session compliance trend",
                     ha="center", va="center", color="#888", fontsize=12)
        ax7.set_title("Cross-Session Compliance Trend", color=label_color, fontweight="bold")
        ax7.tick_params(colors=label_color, axis="both")
        ax7.grid(True, color=grid_color, axis="y")
        ax7.set_facecolor("#1a1d26")

        out_path = self.output_dir / "plots" / f"dashboard_{s.session_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Dashboard saved → {out_path}")
        return str(out_path)

    # ── HTML Report ───────────────────────────
    def generate_html_report(self) -> str:
        """
        Produces a self-contained professional HTML report with:
        - Summary statistics table
        - Embedded dashboard image
        - Session history table
        """
        s = self._session
        stats = self.current_stats()
        dashboard_path = self.generate_dashboard()

        # Encode image as base64 for self-contained HTML
        import base64
        img_b64 = ""
        if dashboard_path and os.path.exists(dashboard_path):
            with open(dashboard_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

        rows = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in stats.items()
        )
        session_rows = "".join(
            f"<tr><td>{s.session_id}</td><td>{s.total_faces}</td>"
            f"<td>{s.with_mask}</td><td>{s.without_mask}</td>"
            f"<td>{s.incorrectly_worn}</td>"
            f"<td class='{'ok' if s.compliance_pct >= 80 else 'warn' if s.compliance_pct >= 50 else 'bad'}'>"
            f"{s.compliance_pct:.1f}%</td></tr>"
            for s in self._all_sessions
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Face Mask Detection — Analytics Report</title>
<style>
  body  {{ font-family: 'Segoe UI', Arial, sans-serif; background:#0f1117; color:#e0e0e0; margin:0; padding:24px; }}
  h1    {{ color:#00c864; border-bottom:2px solid #00c864; padding-bottom:8px; }}
  h2    {{ color:#aaa; margin-top:32px; }}
  table {{ border-collapse:collapse; width:100%; margin-top:12px; }}
  th    {{ background:#1e2130; color:#00c864; padding:10px 14px; text-align:left; }}
  td    {{ padding:8px 14px; border-bottom:1px solid #2a2d3a; }}
  tr:hover td {{ background:#1a1d26; }}
  .ok   {{ color:#00c864; font-weight:bold; }}
  .warn {{ color:#ff9900; font-weight:bold; }}
  .bad  {{ color:#e03030; font-weight:bold; }}
  .badge{{ display:inline-block; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:bold; }}
  .card {{ background:#1a1d26; border-radius:8px; padding:20px; margin:16px 0; border-left:4px solid #00c864; }}
  img   {{ max-width:100%; border-radius:8px; margin-top:16px; }}
  footer{{ color:#555; font-size:12px; margin-top:40px; border-top:1px solid #2a2d3a; padding-top:12px; }}
</style>
</head>
<body>
<h1>🛡 Face Mask Detection — Analytics Report</h1>
<div class="card">
  <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  <p><strong>Session ID:</strong> {s.session_id if s else 'N/A'}</p>
</div>

<h2>📊 Session Statistics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {rows}
</table>

<h2>📈 Analytics Dashboard</h2>
{"<img src='data:image/png;base64," + img_b64 + "' alt='Dashboard'/>" if img_b64 else "<p>Dashboard image not available.</p>"}

<h2>🗂 All Sessions</h2>
<table>
  <tr>
    <th>Session ID</th><th>Total Faces</th><th>With Mask</th>
    <th>Without Mask</th><th>Incorrectly Worn</th><th>Compliance</th>
  </tr>
  {session_rows if session_rows else "<tr><td colspan='6' style='text-align:center;color:#555'>No sessions recorded yet.</td></tr>"}
</table>

<footer>
  Face Mask Detection System &nbsp;|&nbsp; Powered by MobileNetV2 + OpenCV + TensorFlow
  &nbsp;|&nbsp; Report auto-generated &nbsp;|&nbsp; {datetime.now().year}
</footer>
</body>
</html>"""

        out_path = self.output_dir / f"report_{s.session_id if s else 'default'}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        log.info(f"HTML report saved → {out_path}")
        return str(out_path)


# ─────────────────────────────────────────────
# Entry Point (standalone test)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone demo: inject mock data and generate a full report.
    """
    import random
    from dataclasses import dataclass as dc, field as fld

    @dc
    class MockDet:
        label: str; confidence: float; risk_level: str; bbox: tuple = (0,0,0,0); frame_id: int = 0

    @dc
    class MockFrame:
        frame_id: int; detections: list; total_faces: int; unmasked_count: int; alert_triggered: bool

    engine = AnalyticsEngine("demo_analytics")
    engine.start_session("DEMO_SESSION")

    for fid in range(200):
        n = random.randint(1, 4)
        dets = []
        unmasked = 0
        for _ in range(n):
            lbl = random.choices(["WithMask", "WithoutMask", "MaskWornIncorrectly"],
                                  weights=[0.65, 0.25, 0.10])[0]
            conf = random.uniform(0.72, 0.99)
            risk = {"WithMask": "LOW", "WithoutMask": "HIGH", "MaskWornIncorrectly": "MEDIUM"}[lbl]
            dets.append(MockDet(label=lbl, confidence=conf, risk_level=risk))
            if lbl == "WithoutMask":
                unmasked += 1
        fr = MockFrame(frame_id=fid, detections=dets, total_faces=n,
                       unmasked_count=unmasked, alert_triggered=unmasked > 0)
        engine.record(fr)

    engine.end_session()
    print("\n=== Current Stats ===")
    for k, v in engine.current_stats().items():
        print(f"  {k:<28} {v}")
    engine.generate_html_report()
    print("\nDemo complete. Check demo_analytics/ folder.")
