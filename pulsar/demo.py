from html import escape
import json
from pathlib import Path
from statistics import median

from .algorithm import DetectorConfig, StepCounter
from .denoise import preprocess
from .energy import estimate_calories
from .evaluation import evaluate
from .io import save_simulation, write_json
from .models import Gait, Placement
from .simulator import SimulationConfig, simulate


GAIT_NAMES = {Gait.CRUTCH: "拄拐", Gait.KNEE: "膝关节离断", Gait.HIP: "髋关节离断"}


def signal_svg(sim, result):
    data = preprocess(sim.recording)
    width, height = 1040, 260
    duration = sim.recording.duration_s
    origin = sim.recording.samples[0].t
    x = lambda t: 40 + (t - origin) / duration * 980
    y = lambda v: 132 - max(-3.0, min(3.0, v)) * 30
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="原始和降噪加速度，真实运动周期与检测事件">']
    for tick in range(0, int(duration) + 1, 5):
        px = x(origin + tick)
        parts.append(f'<path d="M{px:.1f} 20V210" stroke="#28374c"/><text x="{px:.1f}" y="242" fill="#9eafc8" font-size="12">{tick}s</text>')
    for seg in data.segments:
        center = median(seg.raw)
        stride = max(1, len(seg.times) // 1200)
        for cls, values, color in (("raw", [v - center for v in seg.raw], "#607894"),
                                    ("filtered", seg.filtered, "#53d9cb")):
            points = " ".join(f"{x(seg.times[i]):.1f},{y(values[i]):.1f}" for i in range(0, len(values), stride))
            parts.append(f'<polyline class="{cls}" points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    for e in sim.events:
        if e.kind == "motion_cycle":
            parts.append(f'<circle cx="{x(e.time_s):.1f}" cy="204" r="3" fill="#91e6a5"/>')
    for t in result.event_times_s:
        parts.append(f'<path d="M{x(t):.1f} 183V196" stroke="#ffbd66" stroke-width="2"/>')
    parts.append('</svg>')
    return "".join(parts)


def run_demo(directory: str | Path, *, seed: int = 42, weight_kg: float = 70,
             met: float = 3) -> dict:
    estimate_calories(weight_kg=weight_kg, duration_s=0, met=met)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for gait in (Gait.CRUTCH, Gait.KNEE, Gait.HIP):
        for placement in (Placement.HAND, Placement.THIGH):
            cases.append((f"{gait}_{placement}", f"{GAIT_NAMES[gait]} · {'手持' if placement == Placement.HAND else '大腿'}",
                          SimulationConfig(gait=gait, placement=placement, seed=seed,
                                           pauses_s=((13, 16),), noise_std=0.1, spike_probability=0.003)))
    cases.extend([
        ("stationary", "静止负样本", SimulationConfig(activity="stationary", seed=seed, noise_std=0.12)),
        ("hand_motion", "摆手干扰（真实脚步为 0）", SimulationConfig(activity="hand_motion", seed=seed)),
        ("gap", "髋关节离断 · 手持 · 长丢包", SimulationConfig(gait=Gait.HIP, seed=seed, gap_s=(10, 13))),
    ])
    summaries, panels, rows = [], [], []
    for index, (case_id, title, cfg) in enumerate(cases):
        sim = simulate(cfg)
        save_simulation(sim, root / case_id)
        # 推理只接受传感器；标签仅传给评估。 / Keep truth labels outside inference.
        result = StepCounter(DetectorConfig(gait=cfg.gait, placement=cfg.placement)).analyze(sim.recording)
        metrics = evaluate(result, sim.events)
        energy = estimate_calories(weight_kg=weight_kg, duration_s=result.active_duration_s,
                                   met=met, met_range=(min(2, met), max(5, met)),
                                   met_source="demo_assumption_not_gait_calibrated",
                                   duration_source="detected_bouts")
        entry = {"id": case_id, "title": title, "result": result.to_dict(),
                 "metrics": metrics, "calories": energy.to_dict()}
        write_json(root / case_id / "analysis.json", entry)
        summaries.append(entry)
        panels.append(f'<section class="case" id="case-{index}" {"hidden" if index else ""}><h2>{escape(title)}</h2>{signal_svg(sim, result)}</section>')
        f1 = metrics["event_f1"]
        rows.append(f'<tr><td>{escape(title)}</td><td>{metrics["truth_steps"]}</td><td>{result.step_count}</td><td>{metrics["absolute_count_error"]}</td><td>{f"{f1:.3f}" if f1 is not None else "—"}</td><td>{result.diagnostics["repaired_spikes"]}</td></tr>')
    report = {"seed": seed, "validation": "synthetic_only", "cases": summaries,
              "energy_model": "MET * kg * hours; range is a scenario range, not a confidence interval"}
    write_json(root / "report.json", report)
    html = _HTML.replace("__OPTIONS__", "".join(f'<option value="{i}">{escape(e["title"])}</option>' for i, e in enumerate(summaries)))
    html = html.replace("__PANELS__", "".join(panels)).replace("__ROWS__", "".join(rows))
    html = html.replace("__DATA__", json.dumps(summaries, ensure_ascii=True).replace("<", "\\u003c"))
    html = html.replace("__WEIGHT__", str(weight_kg)).replace("__MET__", str(met))
    (root / "index.html").write_text(html, encoding="utf-8")
    return report


_HTML = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pulsar · 步态与能耗实验</title><style>
:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0b1320;color:#e7edf8}
*{box-sizing:border-box}body{max-width:1200px;margin:auto;padding:36px 24px}h1{font-size:38px;margin:10px 0}h2{font-size:21px}
p{color:#aebed4;line-height:1.7}.tag{color:#53d9cb;letter-spacing:3px}.notice{padding:14px 18px;border-left:3px solid #ffbd66;background:#1c2432;border-radius:6px;margin:24px 0}
.card{background:#111f31;border:1px solid #29384d;border-radius:16px;padding:24px;margin:20px 0}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}label{display:flex;gap:10px;align-items:center}
select,input{background:#0c1726;color:#e7edf8;border:1px solid #41536f;border-radius:8px;padding:10px;font:inherit}input[type=number]{width:100px}select{max-width:100%}
svg{width:100%;height:auto}.legend{display:flex;gap:22px;flex-wrap:wrap}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.stat strong{display:block;font-size:30px;color:#53d9cb;margin-top:8px}.stat span{color:#afbed4}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}td,th{text-align:left;padding:13px 14px;border-bottom:1px solid #2b3a50}th{color:#aebed4;font-weight:500}
.energy{display:flex;gap:36px;flex-wrap:wrap;margin-top:22px}.energy strong{font-size:32px;color:#ffbd66}.bad{color:#ffbd66}
a{color:#53d9cb}@media(max-width:700px){body{padding:20px 12px}h1{font-size:29px}.card{padding:16px}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="tag">PULSAR / RESEARCH DEMO</div><h1>让更多步态被看见</h1>
<p>手机手持与大腿部计步 · 拄拐 / 膝关节离断 / 髋关节离断 · Python 算法与合成数据</p>
<div class="notice">合成信号演示，尚无真实受试者验证。大腿默认按一跨两步估算；摆手和拐杖触地仍可能造成误计数。</div>
<div class="card"><div class="controls"><label>场景 <select id="scenario">__OPTIONS__</select></label>
<label><input type="checkbox" id="showRaw" checked> 原始信号</label><label><input type="checkbox" id="showClean" checked> 降噪信号</label></div>
__PANELS__<div class="legend"><span style="color:#607894">原始模长（减去中位数）</span><span style="color:#53d9cb">降噪与去基线</span><span style="color:#91e6a5">● 真实运动周期</span><span style="color:#ffbd66">| 检测事件</span></div>
<p>Hampel 尖峰抑制 → 居中滑动平均 → 重力基线去除；长丢包断开处理。显示范围为 ±3 m/s²。</p></div>
<div class="card stats"><div class="stat"><span>估算步数 / 真实步数</span><strong id="count"></strong></div><div class="stat"><span>步频 / 步·分钟⁻¹</span><strong id="cadence"></strong></div><div class="stat"><span>估算活动 / 秒</span><strong id="duration"></strong></div><div class="stat"><span>修复尖峰 / 个</span><strong id="spikes"></strong></div></div>
<div class="card"><h2>卡路里估算</h2><p>MET 为可调整的演示假设，未按截肢类型校准。活动时间来自所选场景的检测结果。</p>
<div class="controls"><label>体重 kg <input id="weight" type="number" min="1" value="__WEIGHT__"></label><label>活动 MET <input id="met" type="number" min="0.1" step="0.1" value="__MET__"></label></div>
<div class="energy"><div>总能耗<br><strong id="gross"></strong> kcal</div><div>高于静息的活动能耗<br><strong id="active"></strong> kcal</div></div><p id="energyNote"></p>
<p>总能耗 ≈ MET × 体重 × 小时；活动能耗扣除 1 MET 静息基线。<a href="https://pacompendium.com/">MET 定义来源</a></p></div>
<div class="card"><h2>完整场景结果</h2><p>事件 F1 按真实运动周期匹配，容差 ±0.2 秒；大腿周期与脚步数量分开评估。摆手负样本会暴露这版基线的误报。</p><div class="table-wrap"><table><thead><tr><th>场景</th><th>真实步数</th><th>估算步数</th><th>绝对误差</th><th>事件 F1</th><th>修复尖峰</th></tr></thead><tbody>__ROWS__</tbody></table></div></div>
<script>
const cases=__DATA__;
const el=id=>document.getElementById(id);
function update(){const index=Number(el('scenario').value),c=cases[index],r=c.result;
document.querySelectorAll('.case').forEach((p,i)=>p.hidden=i!==index);
document.querySelectorAll('.raw').forEach(p=>p.style.display=el('showRaw').checked?'':'none');
document.querySelectorAll('.filtered').forEach(p=>p.style.display=el('showClean').checked?'':'none');
el('count').textContent=r.step_count+' / '+c.metrics.truth_steps;
el('cadence').textContent=r.cadence_spm===null?'—':r.cadence_spm.toFixed(1);
el('duration').textContent=r.active_duration_s.toFixed(1);el('spikes').textContent=r.diagnostics.repaired_spikes;
const w=Number(el('weight').value),m=Number(el('met').value),ok=Number.isFinite(w)&&Number.isFinite(m)&&w>0&&m>0;
const factor=w*r.active_duration_s/3600;
el('gross').textContent=ok?(m*factor).toFixed(2):'—';el('active').textContent=ok?(Math.max(m-1,0)*factor).toFixed(2):'—';
el('energyNote').textContent=ok?'估算活动时间 '+r.active_duration_s.toFixed(1)+' 秒；计步误报会同时影响能耗估算。':'请输入正数体重与 MET。';}
['scenario','showRaw','showClean','weight','met'].forEach(id=>el(id).addEventListener('input',update));update();
</script></body></html>'''
