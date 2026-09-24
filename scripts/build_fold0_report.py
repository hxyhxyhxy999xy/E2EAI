from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"D:\study\研究生\实习\中金\E2EAI")
RESULT_ROOT = ROOT / "output" / "csi500_fold0_2018_formula"
METRICS_PATH = RESULT_ROOT / "test_metrics.json"
FIGURE_PATH = RESULT_ROOT / "loss_curves.png"
SPLIT_PATH = RESULT_ROOT / "split_metadata.json"
TRAINING_PATH = RESULT_ROOT / "training.jsonl"
DAILY_PATH = RESULT_ROOT / "daily_backtest" / "daily_backtest_metrics.json"
OUTPUT_PATH = ROOT / "output" / "CSI500_Fold0_2018_experiment_report.docx"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color: str = "D9D9D9", size: str = "6") -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + m))
        if node is None:
            node = OxmlElement("w:" + m)
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(run, name="Aptos", size=10.5, bold=False, color="000000") -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def style_paragraph(paragraph, space_before=0, space_after=6, line_spacing=1.15):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(space_before)
    fmt.space_after = Pt(space_after)
    fmt.line_spacing = line_spacing


def add_text(paragraph, text, bold=False, size=10.5, color="000000", name="Aptos"):
    run = paragraph.add_run(str(text))
    set_run_font(run, name=name, size=size, bold=bold, color=color)
    return run


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    add_text(p, text, bold=True, size=14 if level == 1 else 11.5, name="Aptos")
    return p


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="Body Text")
    style_paragraph(p, space_after=6)
    if bold_lead and text.startswith(bold_lead):
        add_text(p, bold_lead, bold=True)
        add_text(p, text[len(bold_lead):])
    else:
        add_text(p, text)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        style_paragraph(p, space_after=2)
        add_text(p, item)


def add_table(doc, headers, rows, widths=None, font_size=8.7):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header = table.rows[0]
    set_repeat_table_header(header)
    for i, head in enumerate(headers):
        cell = header.cells[i]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "1F4E78")
        set_cell_border(cell)
        set_cell_margins(cell)
        if widths:
            cell.width = Inches(widths[i])
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_paragraph(p, space_after=0, line_spacing=1.0)
        add_text(p, head, bold=True, size=font_size, color="FFFFFF")
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cell)
            set_cell_margins(cell)
            if widths:
                cell.width = Inches(widths[i])
            if ridx % 2 == 1:
                set_cell_shading(cell, "F2F6FA")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            style_paragraph(p, space_after=0, line_spacing=1.05)
            add_text(p, value, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def pct(x):
    return f"{x * 100:.2f}%"


def fmt(x):
    return f"{x:.4f}"


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    for name, size in (("Title", 20), ("Heading 1", 14), ("Heading 2", 11.5)):
        style = styles[name]
        style.font.name = "Aptos Display" if name == "Title" else "Aptos"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
    styles["Body Text"].font.name = "Aptos"
    styles["Body Text"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Body Text"].font.size = Pt(10.5)
    styles["List Bullet"].font.name = "Aptos"
    styles["List Bullet"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["List Bullet"].font.size = Pt(10.5)


def add_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_paragraph(p, space_after=0, line_spacing=1.0)
    add_text(p, "CSI500 Fold 0 2018 起点实验报告", size=8.5, color="666666")


def main():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    model = metrics["by_horizon"]
    baseline = metrics["factor_equal_weight"]["by_horizon"]
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    validation_rows = []
    if TRAINING_PATH.exists():
        for line in TRAINING_PATH.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "validation/validation_loss" in row:
                validation_rows.append(row)
    stopped_epoch = len(validation_rows) - 1
    best_epoch = min(
        range(len(validation_rows)),
        key=lambda index: validation_rows[index]["validation/validation_loss"],
    )
    best_validation_loss = validation_rows[best_epoch]["validation/validation_loss"]
    daily = json.loads(DAILY_PATH.read_text(encoding="utf-8")) if DAILY_PATH.exists() else None

    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    add_footer(section)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    style_paragraph(title, space_after=4, line_spacing=1.0)
    add_text(title, "CSI500 Fold 0 2018 起点实验报告", bold=True, size=20, name="Aptos Display")
    subtitle = doc.add_paragraph()
    style_paragraph(subtitle, space_after=16, line_spacing=1.0)
    add_text(subtitle, "模型结构 数据输入 损失函数与收敛诊断", size=11, color="4F4F4F")

    add_heading(doc, "一 实验结论", 1)
    effective_values = [model[str(h)]["average_effective_holdings"] for h in [3, 5, 10, 15, 20]]
    selected_values = [model[str(h)]["average_selected_stocks"] for h in [3, 5, 10, 15, 20]]
    max_single = max(model[str(h)]["maximum_single_weight"] for h in [3, 5, 10, 15, 20])
    add_body(doc, f"本报告记录 CSI500 Fold 0 的 2018 起点修正版实验。原始日期窗为 2018-01-02 至 2020-12-31 训练、2021 年验证和 2022-01-04 至 2022-12-30 样本外。按 20 日标签的实际退出日期清除边界交叠后，训练期实际截至 {split['used_date_ranges']['train'][1]}，验证期实际截至 {split['used_date_ranges']['validation'][1]}。训练最大轮数为 100，早停 patience 为 20，实际记录 epoch 0 至 {stopped_epoch}。")
    add_body(doc, f"当前 long_only_softmax 配置没有执行单股仓位上限。样本外单股权重极值为 {max_single:.2%}，真正的有效持仓约为 {min(effective_values):.2f} 至 {max(effective_values):.2f} 只，阈值筛选的平均入选股票数约为 {min(selected_values):.1f} 至 {max(selected_values):.1f} 只。这些结果用于诊断修正版数据链路与现有模型配置，不能代表满足 0.5% 业务约束的策略结果。")
    add_bullets(doc, [
        "模型的早停依据为 validation_loss，即四项损失加权后的验证集总损失。",
        "输入因子来自新的 64 因子 H5 面板，行业信息来自 H5 对齐后的一级行业面板。",
        "五个期限的主标签严格使用论文公式 VWAP[t+h] / VWAP[t+1] - 1；t+h+1 版本作为 legacy 敏感性定义保留。",
        "因子等权基准为有效单因子横截面 softmax 组合的等权平均，不使用学习得到的因子注意力。",
    ])

    add_heading(doc, "二 实验区间与运行设置", 1)
    add_table(doc, ["项目", "设置"], [
        ("股票池", "CSI500 历史成员；按交易日精确匹配，不使用未来成员回填"),
        ("训练期", f"原始 2018-01-02 至 2020-12-31；清除后截至 {split['used_date_ranges']['train'][1]}"),
        ("验证期", f"原始 2021-01-04 至 2021-12-31；清除后截至 {split['used_date_ranges']['validation'][1]}"),
        ("样本外期", "2022-01-04 至 2022-12-30"),
        ("预测期限", "3、5、10、15、20 个交易日"),
        ("最大 epoch", "100"),
        ("实际停止", f"记录 epoch 0 至 {stopped_epoch}，共 {len(validation_rows)} 轮"),
        ("最佳验证轮", f"epoch {best_epoch}；validation_loss={best_validation_loss:.6f}"),
        ("早停 patience", "20"),
        ("设备", "CUDA GPU"),
        ("批次", "每批 8 个交易日"),
        ("随机种子", "42"),
    ], widths=[1.65, 5.55])

    add_heading(doc, "三 输入数据细节", 1)
    add_table(doc, ["数据项", "来源与处理"], [
        ("因子面板", "旧全样本面板已因前视偏差被移除；该报告仅保留历史诊断意义"),
        ("因子数量", "64 个新 H5 选出因子；不是原始高频因子集合"),
        ("市场与成员数据", "/cloud/hdf5/historical/china_astock_2018.h5 对齐生成 CSI500 历史成员"),
        ("行业数据", "/cloud/E2EAI/data/index_universes/industry_level1_h5_2018_2026.npz"),
        ("行业层级", "Citic 一级行业；与交易日和股票轴对齐"),
        ("因子输入形状", "[B, N, 64]，B 为交易日批次，N 为当日有效股票数"),
        ("成员筛选", "历史成员精确日期匹配；不做未来日期回填"),
        ("股票数截断", "当前 index 配置不做市场规模截断；max_assets_per_date=null"),
        ("切分样本数", f"训练 {split['used_counts']['train']}、验证 {split['used_counts']['validation']}、样本外 {split['used_counts']['test']} 个交易日"),
        ("收益标签", "论文公式版：VWAP[t+h] / VWAP[t+1] - 1"),
        ("基准收益", "决策日指数权重调整 VWAP 收益代理；缺失标签重新归一化"),
    ], widths=[1.65, 5.55], font_size=8.5)
    add_body(doc, "本次修正版采用论文公式版，因此 h=3、5、10、15、20 对应的退出位置分别是 t+3、t+5、t+10、t+15、t+20；原 t+h+1 版本保留为 legacy 敏感性测试。训练集和验证集均按最大退出偏移清除跨边界标签，样本外评价仍采用 overlapping label 模式，年化使用 252/h。")

    add_heading(doc, "四 模型网络结构", 1)
    add_body(doc, "模型输入为每个交易日的 64 因子横截面和两张关系图。主路径依次经过动态因子选择、股票上下文编码、行业 GAT 中性化、股票池 GAT 中性化、五个独立深度因子头和五个独立组合分配头。")
    add_table(doc, ["模块", "输入", "网络层", "输出与关键设置"], [
        ("动态因子选择", "[B,N,64]", "横截面均值 → Linear 64→32 → LeakyReLU(0.2) → Linear 32→64 → Softmax", "[B,N,64]；gamma_f=0.02；训练 STE hard，测试 hard"),
        ("上下文编码", "选中因子 [B,N,64]", "逐日期逐因子横截面标准化 → Linear 64→64 → LeakyReLU(0.2) → Dropout 0.10 → Linear 64→64", "stock_context [B,N,64]"),
        ("行业中性化", "上下文与行业图", "单头 GAT 64→64；注意力 dropout 0.10；self-loop；ELU；残差相减", "industry_neutral [B,N,64]"),
        ("股票池中性化", "行业中性结果与股票池图", "结构同上，输入为 industry_neutral，不重新使用原始上下文", "universe_neutral [B,N,64]"),
        ("深度因子头", "3 个 64 维表示拼接", "5 个独立 Linear 192→1 → LeakyReLU(0.2)", "deep_factors [B,5,N]；不共享期限头"),
        ("因子解释头", "选中因子 [B,N,64]", "5 个独立 Linear 64→64 → LeakyReLU(0.2) → 因子维 softmax", "factor attention 与 deep factor approximation"),
        ("组合分配头", "上下文 64 维 + signed deep factor 1 维", "5 个独立 Linear 65→1 → LeakyReLU(0.2) → 股票维 masked softmax", "portfolio_weights [B,5,N]"),
    ], widths=[1.25, 1.50, 3.15, 1.30], font_size=7.9)
    add_body(doc, "当前组合分配模式为 long_only_softmax：权重非负、每个日期和期限的权重和为 1，gamma_p=0.001，至少保留 1 只股票。theta=0.10 仅是旧 capped-simplex 模式的兼容字段，本次 no-cap 运行不生效。")
    add_body(doc, f"需要审查的集中度事实：当前运行没有启用 0.5% 单股权重上限，模型样本外单股权重极值为 {max_single:.2%}；有效持仓和阈值入选数分别列示，不能混用。")

    add_heading(doc, "五 损失函数", 1)
    add_body(doc, "当前 long_only_softmax 路径的总损失为：")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_paragraph(p, space_after=8, line_spacing=1.0)
    add_text(p, "L_total = L_portfolio + 0.10 L_stability + 0.10 L_factor + 0.10 L_approximation", bold=True, size=11)
    add_table(doc, ["损失项", "定义与作用", "权重"], [
        ("L_portfolio", "组合收益损失，等于 -mean(portfolio return)。缺失收益标签按有效权重重新归一化。", "1.00"),
        ("L_stability", "方向加权 ICIR 的负值；通过横截面 IC 和时间序列 ICIR 约束深度因子方向稳定性。", "0.10"),
        ("L_factor", "局部横截面回归得到的 psi 与历史 deep direction 的方向收益损失，目标是最大化方向一致的因子收益。", "0.10"),
        ("L_approximation", "deep_factors 与因子注意力方向近似 deep_factor_approx 的 masked MSE。", "0.10"),
        ("上限损失", "long_only_softmax 路径不加入 upper_bound_loss；不会在本次训练中惩罚单股权重超过 theta。", "不启用"),
    ], widths=[1.35, 5.00, 0.85], font_size=8.5)
    add_body(doc, "验证集早停监控的是上述 L_total 的 validation_loss，而不是单独的收益损失、ICIR 或因子近似损失。当前实验使用 AdamW，学习率 0.001，weight decay 0.0001，无学习率调度，梯度裁剪范数 5.0。")

    add_heading(doc, "六 样本外结果", 1)
    add_body(doc, "下表中的换手率是每个期限单独计算的目标权重换手，不是带漂移、交易成本和实际成交路径的执行换手。")
    add_heading(doc, "六点一 模型组合", 2)
    add_table(doc, ["期限 h", "年化收益", "年化超额", "最大回撤", "单次换手", "有效持仓", "平均入选"], [
        (str(h), pct(model[str(h)]["annualized_return"]), pct(model[str(h)]["annualized_excess_return"]), pct(model[str(h)]["max_drawdown"]), pct(model[str(h)]["one_way_turnover"]), f"{model[str(h)]['average_holdings']:.2f}", f"{model[str(h)]['average_selected_stocks']:.1f}")
        for h in [3, 5, 10, 15, 20]
    ], widths=[0.55, 1.05, 1.05, 1.05, 1.05, 1.05, 1.10], font_size=8.0)
    add_heading(doc, "六点二 因子等权基准", 2)
    add_table(doc, ["期限 h", "年化收益", "年化超额", "最大回撤", "单次换手", "有效持仓", "平均入选"], [
        (str(h), pct(baseline[str(h)]["annualized_return"]), pct(baseline[str(h)]["annualized_excess_return"]), pct(baseline[str(h)]["max_drawdown"]), pct(baseline[str(h)]["one_way_turnover"]), f"{baseline[str(h)]['average_holdings']:.2f}", f"{baseline[str(h)]['average_selected_stocks']:.1f}")
        for h in [3, 5, 10, 15, 20]
    ], widths=[0.55, 1.05, 1.05, 1.05, 1.05, 1.05, 1.10], font_size=8.0)
    add_body(doc, "因子等权基准的有效持仓约 422 只，平均入选股票约 496 只，最大单股权重约 2.11%，而模型组合由于没有仓位上限而呈现明显集中。该差异是解释模型与基准表现差异时必须保留的配置因素。")

    if daily is not None:
        add_heading(doc, "七 逐日执行回测", 1)
        add_body(doc, "逐日回测将决策日 t 的目标权重在下一交易日 t+1 的复权 VWAP 执行，持仓在相邻交易日之间按价格漂移，再依据新目标权重调仓。下表使用实际每日净值；本次交易成本为 0 bps，换手率为漂移后权重到目标权重的单边换手。")
        rows = []
        for h in [3, 5, 10, 15, 20]:
            for key, label in (("model", "模型"), ("factor_equal_weight", "因子等权")):
                item = daily["by_horizon"][str(h)][key]
                rows.append((str(h), label, pct(item["annualized_return"]), pct(item["annualized_excess_return"]), pct(item["max_drawdown"]), pct(item["max_active_drawdown"]), pct(item["average_one_way_turnover"])))
        add_table(doc, ["h", "组合", "年化收益", "年化超额", "最大回撤", "主动回撤", "日均单边换手"], rows,
                  widths=[0.42, 0.80, 1.08, 1.08, 1.08, 1.08, 1.26], font_size=7.7)

    add_heading(doc, "八 损失函数收敛曲线", 1)
    add_body(doc, "下图为本次最终运行生成的训练目标曲线。训练在最大 100 轮之前因 validation_loss 连续 20 轮没有改善而早停。曲线显示总损失和验证损失仍存在明显波动，因此本次实验说明模型完成了既定早停流程，但不能将其表述为已经达到全局收敛。")
    if FIGURE_PATH.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(FIGURE_PATH), width=Inches(6.45))
        p.paragraph_format.space_after = Pt(2)
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_paragraph(cap, space_after=8, line_spacing=1.0)
        add_text(cap, "图 1 本次 CSI500 Fold 0 训练目标与验证损失曲线", size=9, color="555555")

    add_heading(doc, "九 审查重点", 1)
    add_bullets(doc, [
        "确认评估是否应严格启用此前提出的 0.5% 单股权重上限；本次结果没有启用该约束。",
        "确认 no-cap long-only-softmax 是否属于论文目标配置，还是仅用于网络联通性和数据接口测试。",
        "重点检查 validation_loss 在后半程的波动、早停轮数与学习率之间的关系。",
        "区分模型表现与组合约束差异：因子等权基准高度分散，而当前模型组合高度集中。",
        "如需论文正式结果，应在确定仓位约束、交易成本和非重叠回测口径后重新运行。",
        "论文表 1 汇总 2015 年至 2022 年中多组时间切分，本报告仅为 2022 年单折，不能直接做数值复现比较。",
    ])

    doc.core_properties.title = "CSI500 Fold 0 2018 起点实验报告"
    doc.core_properties.subject = "E2EAI model audit report"
    doc.core_properties.author = "Codex"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
