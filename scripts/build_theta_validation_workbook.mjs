import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultRoot = path.join(project, "output", "theta_cap_gamma0p5_lambda0", "csi500", "fold_0");
const workbookPath = path.join(project, "output", "CSI500_theta_gamma0p5_lambda0_validation.xlsx");
const previewRoot = path.join(project, "output", "theta_cap_gamma0p5_lambda0", "previews");
const names = ["theta_0p02", "theta_0p015", "theta_0p01", "theta_0p005"];
const horizons = [3, 5, 10, 15, 20];
const data = [];
for (const name of names) {
  const source = path.join(resultRoot, name, "validation_metrics.json");
  const metric = JSON.parse(await fs.readFile(source, "utf8"));
  if (metric.experiment.oos_evaluated !== false) throw new Error(`OOS flag invalid: ${name}`);
  for (const h of horizons) {
    if (metric.by_horizon[String(h)].cap_violation_count !== 0) {
      throw new Error(`Cap violation: ${name} horizon ${h}`);
    }
  }
  data.push(metric);
}
const hashes = new Set(data.map(x => x.experiment.initial_state_sha256));
if (hashes.size !== 1) throw new Error("Initial model state differs between experiments");

const wb = Workbook.create();
const summary = wb.worksheets.add("汇总");
const details = new Map(horizons.map(h => [h, wb.worksheets.add(`H${h}`)]));
const methods = wb.worksheets.add("指标口径");
const font = "Arial";
const navy = "#16304B";
const blue = "#244D72";
const pale = "#EAF1F7";
const ink = "#202B36";

function baseStyle(sheet, width = "A1:R10") {
  sheet.showGridLines = false;
  sheet.getRange(width).format.font = { name: font, size: 10, color: ink };
  sheet.getRange(width).format.verticalAlignment = "center";
  sheet.getRange(width).format.rowHeight = 23;
}
function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: blue,
    font: { name: font, size: 10, color: "#FFFFFF", bold: true },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 34;
}
function title(sheet, cell, value) {
  sheet.getRange(cell).values = [[value]];
  sheet.getRange(cell).format.font = { name: font, size: 14, color: navy, bold: true };
  sheet.getRange(cell).format.rowHeight = 32;
}

baseStyle(summary, "A1:K17");
summary.tabColor = navy;
title(summary, "A2", "CSI500 Fold 0 单股上限验证实验");
summary.getRange("A3").values = [["固定 γ=0.5/N、λs=0；2018–2020训练，2021验证；最佳模型按验证IR选择"]];
summary.getRange("A3").format.font = { name: font, size: 10, color: "#5F7080", italic: true };
summary.getRange("A5:K5").values = [[
  "θ 上限", "γ 系数", "λs", "最佳验证IR", "最佳轮次(0起)", "完成轮数",
  "五期限平均超额", "平均有效持仓", "实际最大单股权重", "平均单边换手率", "权重违规数",
]];
header(summary, "A5:K5");
for (let i = 0; i < data.length; i++) {
  const row = 6 + i;
  const metric = data[i];
  const e = metric.experiment;
  summary.getRange(`A${row}:F${row}`).values = [[
    e.theta, e.gamma_p, e.lambda_s, e.best_validation_ir,
    e.best_epoch, e.epochs_completed,
  ]];
  const detailRow = 5 + i;
  const ex = horizons.map(h => `'H${h}'!E${detailRow}`).join(",");
  const eff = horizons.map(h => `'H${h}'!L${detailRow}`).join(",");
  const max = horizons.map(h => `'H${h}'!M${detailRow}`).join(",");
  const turn = horizons.map(h => `'H${h}'!H${detailRow}`).join(",");
  const cap = horizons.map(h => `'H${h}'!R${detailRow}`).join(",");
  summary.getRange(`G${row}:K${row}`).formulas = [[
    `=AVERAGE(${ex})`, `=AVERAGE(${eff})`, `=MAX(${max})`,
    `=AVERAGE(${turn})`, `=SUM(${cap})`,
  ]];
}
summary.getRange("A6:A9").setNumberFormat("0.0%");
summary.getRange("B6:D9").setNumberFormat("0.0000");
summary.getRange("E6:F9").setNumberFormat("0");
summary.getRange("G6:G9").setNumberFormat("0.00%");
summary.getRange("H6:H9").setNumberFormat("0.0");
summary.getRange("I6:J9").setNumberFormat("0.00%");
summary.getRange("K6:K9").setNumberFormat("0");
summary.getRange("A6:K9").format.rowHeight = 27;
summary.getRange("A6:K9").format.borders = { insideHorizontal: { style: "thin", color: "#DCE5EC" } };
summary.getRange("A6:K6").format.fill = pale;
summary.getRange("A12").values = [["五期限平均超额为标签收益的算术平均，不是年化收益或真实交易净值。"]];
summary.getRange("A13").values = [["换手率是相邻决策日目标权重的单边变化，不含持仓漂移、交易成本和成交约束。"]];
summary.getRange("A14").values = [["全部指标只来自2021验证集；2022样本外未评估。"]];
summary.getRange("A12:A14").format.font = { name: font, size: 10, color: "#5F7080" };
summary.getRange("A:A").format.columnWidth = 18;
summary.getRange("B:C").format.columnWidth = 13;
summary.getRange("D:K").format.columnWidth = 18;

const detailHeaders = [
  "θ 上限", "多头平均收益", "多头年化", "多头夏普",
  "超额平均收益", "超额年化", "超额夏普(IR)", "单边换手率",
  "双边换手率", "多头最大回撤", "主动最大回撤", "平均有效持仓",
  "实际最大单股权重", "平均每日最大权重", "平均IC", "ICIR",
  "平均入选股票数", "权重违规数",
];
for (const h of horizons) {
  const sheet = details.get(h);
  baseStyle(sheet, "A1:R12");
  title(sheet, "A2", `未来${h}日验证指标`);
  sheet.getRange("A3").values = [[`VWAP[t+${h}]/VWAP[t+1]−1；年化乘数252/${h}；最大回撤用每隔${h}日决策样本`]];
  sheet.getRange("A3").format.font = { name: font, size: 10, color: "#5F7080", italic: true };
  sheet.getRange("A4:R4").values = [detailHeaders];
  header(sheet, "A4:R4");
  const rows = data.map(metric => {
    const x = metric.by_horizon[String(h)];
    return [
      metric.experiment.theta, x.mean_return, x.annualized_return, x.sharpe,
      x.mean_excess_return, x.annualized_excess_return, x.excess_sharpe,
      x.one_way_turnover, x.gross_turnover, x.max_drawdown,
      x.max_active_drawdown, x.average_effective_holdings,
      x.maximum_single_weight, x.average_max_weight, x.mean_ic, x.icir,
      x.average_final_selected_count, x.cap_violation_count,
    ];
  });
  sheet.getRange("A5:R8").values = rows;
  sheet.getRange("A5:R8").format.rowHeight = 26;
  sheet.getRange("A5:R8").format.borders = { insideHorizontal: { style: "thin", color: "#DCE5EC" } };
  sheet.getRange("A5:R5").format.fill = pale;
  for (const col of ["A","B","C","E","F","H","I","J","K","M","N"]) {
    sheet.getRange(`${col}5:${col}8`).setNumberFormat("0.00%");
  }
  sheet.getRange("D5:D8").setNumberFormat("0.00");
  sheet.getRange("G5:G8").setNumberFormat("0.00");
  sheet.getRange("L5:L8").setNumberFormat("0.0");
  sheet.getRange("O5:P8").setNumberFormat("0.0000");
  sheet.getRange("Q5:Q8").setNumberFormat("0.0");
  sheet.getRange("R5:R8").setNumberFormat("0");
  sheet.getRange("A:A").format.columnWidth = 13;
  sheet.getRange("B:R").format.columnWidth = 18;
  sheet.freezePanes.freezeRows(4);
}

baseStyle(methods, "A1:B15");
methods.tabColor = "#8EA4B7";
title(methods, "A2", "指标口径");
methods.getRange("A4:B12").values = [
  ["数据区间", "Fold 0：2018–2020训练（边界清除后到2020-12-03），2021验证（到2021-12-03）；2022样本外未评估。"],
  ["收益标签", "未来h日收益 = VWAP[t+h]/VWAP[t+1] − 1；期限h为3、5、10、15、20。"],
  ["年化收益", "重叠h日标签的平均收益 × 252/h，为算术代理口径。"],
  ["多头夏普", "sqrt(252/h) × 多头收益均值 / 多头收益标准差（总体标准差）。"],
  ["超额夏普", "信息比率：sqrt(252/h) × 超额收益均值 / 超额收益标准差。"],
  ["最大回撤", "从验证集第一个决策日开始，每隔h个决策日取一个收益标签，复利得到峰谷回撤。主动回撤对同一子序列的多头减基准收益计算。"],
  ["换手率", "每个期限相邻决策日，以稳定asset_id对齐后的目标权重差；单边=0.5×Σ|w_t−w_{t−1}|，双边=Σ|w_t−w_{t−1}|。首日不计。"],
  ["有效持仓", "逐日 1/Σw_i²，再对验证决策日取平均。"],
  ["限制", "标签重叠、未模拟持仓漂移、实际成交、费用或交易成本；此表不是实际交易净值回测。"],
];
methods.getRange("A4:A12").format.font = { name: font, size: 10, color: navy, bold: true };
methods.getRange("A4:B12").format.rowHeight = 29;
methods.getRange("A:A").format.columnWidth = 18;
methods.getRange("B:B").format.columnWidth = 115;
methods.getRange("B4:B12").format.wrapText = true;

wb.recalculate();
const check = await wb.inspect({
  kind: "table", range: "汇总!A5:K9", include: "values,formulas",
  tableMaxRows: 6, tableMaxCols: 11,
});
console.log(check.ndjson);
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
await fs.mkdir(previewRoot, { recursive: true });
for (const sheetName of ["汇总", ...horizons.map(h => `H${h}`), "指标口径"]) {
  const range = sheetName === "汇总" ? "A1:K14" : sheetName === "指标口径" ? "A1:B12" : "A1:R8";
  const preview = await wb.render({ sheetName, range, scale: 1.5, format: "png" });
  await fs.writeFile(
    path.join(previewRoot, `${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(workbookPath);
const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const savedCheck = await saved.inspect({
  kind: "table", range: "汇总!A5:K9", include: "values,formulas",
  tableMaxRows: 6, tableMaxCols: 11,
});
console.log(savedCheck.ndjson);
const savedErrors = await saved.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "saved workbook formula error scan",
});
console.log(savedErrors.ndjson);
console.log(JSON.stringify({ workbookPath, sheets: ["汇总", ...horizons.map(h => `H${h}`), "指标口径"] }));
