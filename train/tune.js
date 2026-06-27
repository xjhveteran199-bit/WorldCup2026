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

// ---------- 2. 网格搜索 ELO_K × FORM_BLEND（walk-forward logloss/命中） ----------
function grid() {
  const Ks = [40, 50, 60, 70, 80, 100];
  const Bs = [0.3, 0.4, 0.5, 0.6, 0.7];
  console.log("\n===== 网格搜索（66场 walk-forward，越低越好的 logloss / 越高越好的命中）=====");
  console.log("ELO_K \\ BLEND   " + Bs.map(b => "b=" + b).join("        "));
  let best = null;
  Ks.forEach(k => {
    let line = "K=" + String(k).padEnd(4) + "  ";
    Bs.forEach(b => {
      E.configure({ ELO_K: k, FORM_BLEND: b });
      const bt = E.backtestWC(D);
      const cell = bt.logloss.toFixed(3) + "/" + (bt.topHitRate * 100).toFixed(0) + "%";
      line += cell.padEnd(13);
      const score = bt.logloss; // 主目标：logloss
      if (!best || score < best.ll) best = { k, b, ll: bt.logloss, hit: bt.topHitRate, sc: bt.scoreHit };
    });
    console.log(line);
  });
  console.log(`\n>>> 最优(按logloss): ELO_K=${best.k} FORM_BLEND=${best.b}  logloss=${best.ll.toFixed(4)} 命中=${(best.hit*100).toFixed(1)}% 比分=${best.sc}/66`);

  // 对比：现行默认 60/0.5
  E.configure({ ELO_K: 60, FORM_BLEND: 0.5 });
  const base = E.backtestWC(D);
  console.log(`    现行默认 60/0.5     logloss=${base.logloss.toFixed(4)} 命中=${(base.topHitRate*100).toFixed(1)}% 比分=${base.scoreHit}/66`);
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
