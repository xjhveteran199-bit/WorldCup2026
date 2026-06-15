/**
 * apply-ratings.js — 把 core/ratings.js（训练同源·赛前 Elo/form）写回 core/data.js 的 48 队。
 *
 * 目的：让端内推理用的 elo/form 与离线训练 MLP 时**同源同标度**，
 * 消除「线上 eloratings 标度 ≠ 训练 Elo 标度」导致的特征标准化错位。
 *
 * 只改 data.js 里每队的 elo: 与 form: 两个数值（保留全部编辑字段、赛程、results）。
 * data.js 仍是 results（赛果）的唯一编辑源；本脚本只在重训后手动运行一次。
 *
 *   form 映射：data.js form(0-100) = 40 + 55 × 近期状态率(0~1)，
 *   使 engine.formRatio(form) ≈ 训练时的 form_h/form_a，喂给模型即真实赛前状态。
 *
 * 运行：node apply-ratings.js
 */
const fs = require("fs");
const path = require("path");

const data = require("./core/data.js");
const { ratings, meta } = require("./core/ratings.js");

// data.js 的 en 名 → ratings.js（martj42 训练队名）别名映射
const ALIAS = {
  "Czechia": "Czech Republic",
  "Türkiye": "Turkey",
  "Côte d’Ivoire": "Ivory Coast",
  "Côte d'Ivoire": "Ivory Coast",
  "Bosnia & Herzegovina": "Bosnia and Herzegovina",
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const DATA_PATH = path.join(__dirname, "core", "data.js");
let src = fs.readFileSync(DATA_PATH, "utf8");

const rows = [];
const missing = [];
for (const code of Object.keys(data.teams)) {
  const t = data.teams[code];
  const key = ALIAS[t.en] || t.en;
  const r = ratings[key];
  if (!r) { missing.push(`${code}(${t.en})`); continue; }
  const newElo = Math.round(r.elo);
  const newForm = clamp(Math.round(40 + 55 * r.form), 40, 98);

  // 限定在该队对象块内替换首个 elo:/form: 数值（code 唯一，非贪婪到第一个键）
  const eloRe = new RegExp("(" + code + ": \\{[\\s\\S]*?\\belo: )\\d+");
  const formRe = new RegExp("(" + code + ": \\{[\\s\\S]*?\\bform: )\\d+");
  if (!eloRe.test(src) || !formRe.test(src)) { missing.push(`${code}(块未匹配)`); continue; }
  src = src.replace(eloRe, "$1" + newElo).replace(formRe, "$1" + newForm);
  rows.push({ code, en: t.en, elo: `${t.elo}->${newElo}`, form: `${t.form}->${newForm}`, ratio: r.form });
}

if (missing.length) {
  console.error("❌ 以下队未写回，请检查别名映射或 ratings.js：", missing.join(", "));
  process.exit(1);
}

// 更新头部基线日期标注（elo/form 已改为模型同源）
src = src.replace(/eloDate: "[^"]*"/, `eloDate: "${meta.asof}"`);
src = src.replace(
  /\* 数据基线：Elo = [^\n]*\n/,
  `* 数据基线：Elo/form = 训练同源（约5万场历史比赛时序评分，赛前截至 ${meta.asof}）| FIFA排名 = 2026-06-11\n`
);

fs.writeFileSync(DATA_PATH, src, "utf8");
console.log(`✅ 已写回 ${rows.length} 队到 core/data.js（赛前截至 ${meta.asof}，源 ${meta.source_rows} 场）`);
console.log(rows.sort((a, b) => b.ratio - a.ratio).slice(0, 6).map(r => `${r.code} elo ${r.elo} form ${r.form}`).join("\n"));
