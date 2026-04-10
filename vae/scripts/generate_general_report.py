"""
Generate a simplified Chinese HTML report that keeps only the relatively
general-model stages and removes task-specific "precision fixes" from the
mainline narrative.
"""

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIST_JSON = ROOT / "outputs/comparisons/distributional_metrics_general.json"
VIZ_JSON = ROOT / "outputs/competitive_model_viz_cn_general/summary.json"
SUMMARY_PLOT = ROOT / "outputs/comparisons/distributional_metrics_general_summary_cn.png"
OUT_HTML = ROOT / "outputs/reports/hand_prior_experiment_report_cn.html"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def img(path: Path, alt: str) -> str:
    return f'<img src="{html.escape(str(path.resolve()))}" alt="{html.escape(alt)}" />'


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def model_intro(key: str) -> str:
    mapping = {
        "vae_step": "最早的一步预测基线。优点是结构最中性；缺点是明显均值化，很多 rollout 到 100 步仍不触发合手。",
        "chunk_h12_b1e3": "把一步预测改成整段 chunk 预测后，长滚动更稳，但会把“open-like 条件下的多峰未来”压成几乎必关。",
        "flow_post_best": "在更通用的家族里，局部分布拟合和闭合幅度校准都更强，但当前实现仍有明显 reopen 问题。",
        "state_space_rw_best": "第一次把 hidden state 和 2D stochastic latent 分开。在 open-like 条件分布上最接近当前数据，但闭合尾段偏大。",
        "traj_hold": "保留了 episode 起点语义，reset 长滚动很像数据，但对任意 open-like chunk 几乎不给触发概率，泛化性不足。",
    }
    return mapping.get(key, "")


def stage_name(key: str) -> str:
    mapping = {
        "vae_step": "阶段一：一步 VAE",
        "chunk_h12_b1e3": "阶段二：Chunk-VAE",
        "flow_post_best": "阶段三：Flow + posterior consistency",
        "state_space_rw_best": "阶段四：随机窗口 State-space",
        "traj_hold": "阶段五：Trajectory-start",
    }
    return mapping.get(key, key)


def build_metric_table(data):
    keep_keys = {"vae_step", "chunk_h12_b1e3", "flow_post_best", "state_space_rw_best", "traj_hold"}
    rows = []
    for m in data["models"]:
        if m["key"] not in keep_keys:
            continue
        met = m["metrics"]
        rows.append(
            f"""
            <tr>
              <td><code>{html.escape(m['key'])}</code></td>
              <td>{html.escape(m['label'])}</td>
              <td>{fmt(met['open_like_close_prob'])}</td>
              <td>{fmt(met['open_like_delay_tv'])}</td>
              <td>{fmt(met['open_like_hazard_l1'])}</td>
              <td>{fmt(met['reset_survival_l1'])}</td>
              <td>{fmt(met['reset_reopen_rate'])}</td>
              <td>{fmt(met['closed_tail_progress_bias'])}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_model_sections(dist_data, viz_data):
    viz_map = {m["key"]: m for m in viz_data["models"]}
    sections = []
    for m in dist_data["models"]:
        if m["key"] not in viz_map:
            continue
        viz = viz_map[m["key"]]
        met = m["metrics"]
        cards = "\n".join(
            f"""
            <figure>
              {img(ROOT / path, f"{m['key']} {Path(path).name}")}
              <figcaption><code>{Path(path).name}</code></figcaption>
            </figure>
            """
            for path in viz["curve_files"]
        )
        section = f"""
        <section class="model-section">
          <h2>{html.escape(stage_name(m['key']))}</h2>
          <p class="intro">{html.escape(model_intro(m['key']))}</p>
          <ul>
            <li><strong>open-like 条件 12 步触发概率：</strong>{fmt(met['open_like_close_prob'])}</li>
            <li><strong>open-like 延迟分布 TV：</strong>{fmt(met['open_like_delay_tv'])}</li>
            <li><strong>reset survival L1：</strong>{fmt(met['reset_survival_l1'])}</li>
            <li><strong>reopen 违例率：</strong>{fmt(met['reset_reopen_rate'])}</li>
            <li><strong>闭合尾段均值偏差：</strong>{fmt(met['closed_tail_progress_bias'])}</li>
          </ul>
          <div class="gallery">
            {cards}
          </div>
        </section>
        """
        sections.append(section)
    return "\n".join(sections)


def main():
    dist_data = load_json(DIST_JSON)
    viz_data = load_json(VIZ_JSON)
    selected = viz_data["selected_trajectories"]
    gt_open_close = dist_data["references"]["open_like_test"]["close_prob"]
    gt_tail = dist_data["references"]["reset_test"]["closed_tail_progress_mean"]
    selected_text = "、".join(
        f"{row['traj_id']}（onset={row['onset']}）" for row in selected
    )

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>手状态先验实验报告（通用版）</title>
  <style>
    body {{
      font-family: "PingFang SC","Hiragino Sans GB","Songti SC","Microsoft YaHei",sans-serif;
      margin: 0;
      background: #f6f7fb;
      color: #1f2937;
      line-height: 1.7;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 36px 24px 64px;
    }}
    h1, h2 {{
      color: #111827;
    }}
    .lead, .note {{
      background: white;
      border-radius: 18px;
      padding: 20px 24px;
      box-shadow: 0 10px 30px rgba(17,24,39,0.06);
      margin-bottom: 22px;
    }}
    .lead strong {{
      color: #b42318;
    }}
    .hero-img img, .summary-img img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid #dbe1ea;
      background: white;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(17,24,39,0.06);
      margin: 18px 0 28px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #eef2ff;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .gallery figure, .latent figure {{
      margin: 0;
      background: white;
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 8px 24px rgba(17,24,39,0.06);
    }}
    .gallery img, .latent img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
    }}
    .model-section {{
      margin: 34px 0 44px;
    }}
    code {{
      background: #f3f4f6;
      padding: 1px 6px;
      border-radius: 6px;
      font-size: 0.95em;
    }}
    .small {{
      font-size: 13px;
      color: #4b5563;
    }}
    @media (max-width: 900px) {{
      .gallery {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>手状态先验实验报告（通用版）</h1>

    <div class="lead">
      <p><strong>这版报告已把“精准动刀”的模型从主线里移除。</strong> 当前不再把 <code>plateau</code> / <code>m1</code> / <code>anti-drop</code> 当作通用方案的一部分，因为它们对“单次张开→合拢”任务引入了过强的定向偏置。</p>
      <p>保留下来的主线只包含 5 个相对通用的阶段：<code>vae_step</code>、<code>chunk_h12_b1e3</code>、<code>flow_post_best</code>、<code>state_space_rw_best</code>、<code>traj_hold</code>。</p>
      <p class="small">本次重新评估的关键参考值：open-like 条件下 12 步内触发概率 = <strong>{fmt(gt_open_close)}</strong>；GT 闭合尾段 progress 均值 = <strong>{fmt(gt_tail)}</strong>。</p>
    </div>

    <div class="note">
      <h2>这版文档看什么</h2>
      <ul>
        <li>先看新的分布型总览图：它直接对应“给定当前 state chunk，未来分布像不像 GT”。</li>
        <li>再看每个模型的 5 张“GT + 50 次 AR”曲线图：被选中的 5 条轨迹是 {html.escape(selected_text)}。</li>
        <li>如果一个模型只会在 reset 时看起来像样，但对任意 open-like chunk 给不出合理触发分布，它就不算通用先验。</li>
      </ul>
    </div>

    <section class="summary-img">
      <h2>新的分布型评估总览</h2>
      {img(SUMMARY_PLOT, "分布型评估总览")}
    </section>

    <section>
      <h2>保留模型的关键指标</h2>
      <table>
        <thead>
          <tr>
            <th>模型</th>
            <th>标签</th>
            <th>open-like 触发概率</th>
            <th>延迟分布 TV</th>
            <th>hazard L1</th>
            <th>reset survival L1</th>
            <th>reopen 率</th>
            <th>尾段偏差</th>
          </tr>
        </thead>
        <tbody>
          {build_metric_table(dist_data)}
        </tbody>
      </table>
      <p class="small">解释：越接近 GT 越好；其中 <code>reopen 率</code> 对当前数据应尽量接近 0，因为 GT 不存在 close→open。</p>
    </section>

    {build_model_sections(dist_data, viz_data)}
  </div>
</body>
</html>
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(OUT_HTML)


if __name__ == "__main__":
    main()
