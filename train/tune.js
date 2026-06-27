/* 离线调参 + 现状分析（不进生产）。运行：node train/tune.js */
const path = require("path");
const D = require(path.join(__dirname, "..", "core", "data.js"));
const E = require(path.join(__dirname, "..", "core", "engine.js"));

// ---------- 1. 小组积分榜（基于已录入赛果） ----------
function standings() {
  const S = {};
  Object.keys(D.teams).forEach(c => S[c] = { code: c, g: D.teams[c].group, pts: 0, gf: 0, ga: 0, pl: 0 });
  D.fixtures.forEach(fx => {
    const r = D.results[fx.n]; if (!r) return;
    const h = S[fx.h], a = S[fx.a];
    h.pl++; a.pl++; h.gf += r[0]; h.ga += r[1]; a.gf += r[1]; a.ga += r[0];
    if (r[0] > r[1]) h.pts += 3; else if (r[0] < r[1]) a.pts += 3; else { h.pts++; a.pts++; }
  });
  const groups = {};
  Object.values(S).forEach(t => (groups[t.g] = groups[t.g] || []).push(t));
  const cmp = (a, b) => b.pts - a.pts || (b.gf - b.ga) - (a.gf - a.ga) || b.gf - a.gf;
  console.log("\n===== 小组积分榜（已赛 " + Object.keys(D.results).length + " 场）=====");
  Object.keys(groups).sort().forEach(g => {
    groups[g].sort(cmp);
    const done = groups[g].every(t => t.pl === 3);
    console.log(`\n[${g}组]${done ? "" : "  ⏳未踢完"}`);
    groups[g].forEach((t, i) => {
      const mark = i === 0 ? "①" : i === 1 ? "②" : i === 2 ? "③" : "  ";
      console.log(`  ${mark} ${t.code.padEnd(4)} ${t.pl}场 ${t.pts}分  净${(t.gf - t.ga >= 0 ? "+" : "") + (t.gf - t.ga)}  (${t.gf}:${t.ga})`);
    });
  });
}

// 给每场标注「该两队较少一方的赛前已踢场次」(=轮次-1)。md>=1 即两队都至少踢过1场 → 淘汰赛代理子集
function annotateMatchday(rounds) {
  const played = {};
  const byN = {};
  D.fixtures.forEach(fx => { if (D.results[fx.n]) byN[fx.n] = fx; });
  // 按赛程顺序推进
  D.fixtures.filter(f => f.n <= 72 && D.results[f.n]).forEach(fx => {
    const ph = played[fx.h] || 0, pa = played[fx.a] || 0;
    const rec = rounds.find(r => r.n === fx.n);
    if (rec) rec.prior = Math.min(ph, pa); // 两队中较少的赛前场次
    played[fx.h] = ph + 1; played[fx.a] = pa + 1;
  });
}
// 子集指标
function metric(rounds, filt) {
  const r = rounds.filter(filt);
  if (!r.length) return null;
  const hit = r.filter(x => x.hit).length;
  const ll = r.reduce((s, x) => s + x.ll, 0) / r.length;
  return { n: r.length, hit, hitRate: hit / r.length, ll };
}

// ---------- 2. 网格搜索 ELO_K × FORM_BLEND，目标=淘汰赛代理子集(两队均≥1场) logloss ----------
function grid() {
  const Ks = [15, 20, 25, 30, 35, 40, 50, 60];
  const Bs = [0.3, 0.4, 0.5, 0.6];
  console.log("\n===== 网格搜索（目标：淘汰赛代理子集 = 两队均≥1场后的比赛，越低越好 logloss / 越高越好命中）=====");
  console.log("（淘汰赛各队已踢满3场，最贴近 prior≥1 的子集；首轮 prior=0 的冷启动噪声不纳入调参目标）");
  console.log("ELO_K \\ BLEND   " + Bs.map(b => "b=" + b).join("       "));
  let best = null;
  Ks.forEach(k => {
    let line = "K=" + String(k).padEnd(4) + "  ";
    Bs.forEach(b => {
      E.configure({ ELO_K: k, FORM_BLEND: b });
      const bt = E.backtestWC(D);
      annotateMatchday(bt.rounds);
      const ko = metric(bt.rounds, x => x.prior >= 1);     // 淘汰赛代理
      const cell = ko.ll.toFixed(3) + "/" + (ko.hitRate * 100).toFixed(0) + "%";
      line += cell.padEnd(12);
      if (!best || ko.ll < best.ll) best = { k, b, ll: ko.ll, hit: ko.hitRate, n: ko.n };
    });
    console.log(line);
  });
  console.log(`\n>>> 淘汰赛子集最优(按logloss): ELO_K=${best.k} FORM_BLEND=${best.b}  logloss=${best.ll.toFixed(4)} 命中=${(best.hit*100).toFixed(1)}% (n=${best.n})`);

  // 对比现行默认 30/0.5（全集 + 淘汰赛子集）
  E.configure({ ELO_K: 30, FORM_BLEND: 0.5 });
  const base = E.backtestWC(D); annotateMatchday(base.rounds);
  const baseAll = metric(base.rounds, () => true), baseKo = metric(base.rounds, x => x.prior >= 1);
  console.log(`    现行默认 30/0.5  全集 logloss=${baseAll.ll.toFixed(4)} 命中=${(baseAll.hitRate*100).toFixed(1)}%  |  淘汰赛子集 logloss=${baseKo.ll.toFixed(4)} 命中=${(baseKo.hitRate*100).toFixed(1)}%`);
  console.log("    注：KO_FACTOR(0.90)与点球模型只作用于 knockout:true，无WC淘汰赛赛果可回测，保持合理先验不动。");
  return best;
}

// ---------- 3. 阶段拆分命中率 ----------
function byStage() {
  E.configure({ ELO_K: 60, FORM_BLEND: 0.5 });
  const bt = E.backtestWC(D);
  // 1-48 = 前两轮, 49-72 = 第三轮（小组收官）
  const seg = (lo, hi) => {
    const r = bt.rounds.filter(x => x.n >= lo && x.n <= hi);
    const hit = r.filter(x => x.hit).length, sc = r.filter(x => x.scoreHit).length;
    const ll = r.reduce((s, x) => s + x.ll, 0) / (r.length || 1);
    return `${r.length}场 命中${hit}(${(hit/r.length*100).toFixed(0)}%) 比分${sc} logloss${ll.toFixed(3)}`;
  };
  console.log("\n===== 阶段拆分 =====");
  console.log("  第1轮(1-32):  " + seg(1, 32));
  console.log("  第2轮(33-48): " + seg(33, 48));
  console.log("  第3轮(49-72): " + seg(49, 72));
  console.log("  全部:         " + seg(1, 72));
}

standings();
byStage();
grid();
