/**
 * 硅基章鱼 · 统计预测引擎（纯函数，零平台依赖，网页与小程序共享）
 *
 * 方法论：
 * 1. Elo 胜负期望  We = 1 / (1 + 10^(-Δ/400))，Δ = 有效Elo差
 *    有效Elo = 基础Elo + 东道主加成(+100) + 状态修正((form-70)×1.5, 封顶±50)
 * 2. 双泊松进球模型  λ = 1.30 × e^(0.0011×Δ)，淘汰赛 ×0.90（保守系数）
 * 3. Dixon-Coles 低比分修正  ρ = -0.13（修正 0-0/1-0/0-1/1-1 的相关性）
 * 4. 比分矩阵 0-8 × 0-8 → 胜/平/负概率、最可能比分、淘汰赛点球模型
 */
(function (root) {
  "use strict";

  var HOST_BONUS = 100;        // 东道主 Elo 加成
  var FORM_NEUTRAL = 70;       // 状态中性值
  var FORM_WEIGHT = 1.5;       // 状态权重
  var FORM_CAP = 50;           // 状态修正封顶
  var BASE_LAMBDA = 1.30;      // Δ=0 时单队期望进球
  var K_LAMBDA = 0.0011;       // Elo差→进球的弹性
  var KO_FACTOR = 0.90;        // 淘汰赛进球压缩
  var DC_RHO = -0.13;          // Dixon-Coles 相关系数
  var MAX_GOALS = 8;           // 比分矩阵上限

  var factCache = [1];
  function fact(n) {
    if (factCache[n] === undefined) factCache[n] = n * fact(n - 1);
    return factCache[n];
  }
  function poisson(k, lambda) {
    return Math.pow(lambda, k) * Math.exp(-lambda) / fact(k);
  }
  // Dixon-Coles 修正系数
  function dcTau(x, y, la, lb, rho) {
    if (x === 0 && y === 0) return 1 - la * lb * rho;
    if (x === 0 && y === 1) return 1 + la * rho;
    if (x === 1 && y === 0) return 1 + lb * rho;
    if (x === 1 && y === 1) return 1 - rho;
    return 1;
  }

  /**
   * 计算有效 Elo 与修正明细
   * t: 球队对象 { elo, form }; opts: { isHost, formOverride }
   */
  function effectiveElo(t, opts) {
    opts = opts || {};
    var form = (opts.formOverride !== undefined && opts.formOverride !== null) ? opts.formOverride : t.form;
    var formAdj = Math.max(-FORM_CAP, Math.min(FORM_CAP, (form - FORM_NEUTRAL) * FORM_WEIGHT));
    var hostAdj = opts.isHost ? HOST_BONUS : 0;
    return { base: t.elo, formAdj: Math.round(formAdj), hostAdj: hostAdj, total: t.elo + formAdj + hostAdj };
  }

  /**
   * 核心：预测一场比赛
   * teamA/teamB: 球队对象（含 elo, form）
   * opts: { knockout: bool, hostA: bool, hostB: bool, formA, formB }
   * 返回完整预测对象（概率、比分矩阵、明细）
   */
  function predictMatch(teamA, teamB, opts) {
    opts = opts || {};
    var ea = effectiveElo(teamA, { isHost: opts.hostA, formOverride: opts.formA });
    var eb = effectiveElo(teamB, { isHost: opts.hostB, formOverride: opts.formB });
    var dr = ea.total - eb.total;
    var we = 1 / (1 + Math.pow(10, -dr / 400));

    var koMul = opts.knockout ? KO_FACTOR : 1;
    var la = clamp(BASE_LAMBDA * Math.exp(K_LAMBDA * dr) * koMul, 0.15, 4.5);
    var lb = clamp(BASE_LAMBDA * Math.exp(-K_LAMBDA * dr) * koMul, 0.15, 4.5);

    // 比分矩阵
    var matrix = [], pWin = 0, pDraw = 0, pLoss = 0, total = 0;
    for (var x = 0; x <= MAX_GOALS; x++) {
      matrix[x] = [];
      for (var y = 0; y <= MAX_GOALS; y++) {
        var p = poisson(x, la) * poisson(y, lb) * dcTau(x, y, la, lb, DC_RHO);
        matrix[x][y] = p;
        total += p;
      }
    }
    // 归一化（截断+DC修正后概率和略偏离1）
    var scores = [];
    for (x = 0; x <= MAX_GOALS; x++) {
      for (y = 0; y <= MAX_GOALS; y++) {
        matrix[x][y] /= total;
        if (x > y) pWin += matrix[x][y];
        else if (x === y) pDraw += matrix[x][y];
        else pLoss += matrix[x][y];
        scores.push({ a: x, b: y, p: matrix[x][y] });
      }
    }
    scores.sort(function (m, n) { return n.p - m.p; });

    // 淘汰赛：平局进入加时+点球。点球胜率以 Elo 微调（大赛点球大体随机）
    var pPenA = clamp(0.5 + dr / 4000, 0.35, 0.65);
    var advanceA = opts.knockout ? pWin + pDraw * pPenA : null;

    return {
      eloA: ea, eloB: eb, drift: dr, winExpectancy: we,
      lambdaA: la, lambdaB: lb,
      pWin: pWin, pDraw: pDraw, pLoss: pLoss,
      matrix: matrix, topScores: scores.slice(0, 5),
      knockout: !!opts.knockout, pPenaltyA: pPenA, pAdvanceA: advanceA
    };
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /** 从比分矩阵采样一个比分 [x, y] */
  function sampleScore(matrix, rng) {
    var r = (rng || Math.random)(), acc = 0;
    for (var x = 0; x <= MAX_GOALS; x++) {
      for (var y = 0; y <= MAX_GOALS; y++) {
        acc += matrix[x][y];
        if (r < acc) return [x, y];
      }
    }
    return [0, 0];
  }

  /**
   * 蒙特卡洛模拟整届世界杯
   * data: WC_DATA；nSims: 模拟次数
   * useResults: 是否锁定已赛结果（默认 true）
   * onProgress(done, total): 进度回调（可选）
   * 返回 { champion: {code: count}, final: {}, semi: {}, nSims }
   */
  function simulateTournament(data, nSims, useResults, onProgress) {
    nSims = nSims || 10000;
    useResults = useResults !== false;
    var teams = data.teams;
    var champion = {}, finalApp = {}, semiApp = {};
    var matrixCache = {};

    function getMatrix(hCode, aCode, knockout) {
      var key = hCode + "|" + aCode + "|" + (knockout ? 1 : 0);
      if (!matrixCache[key]) {
        matrixCache[key] = predictMatch(teams[hCode], teams[aCode], {
          knockout: knockout,
          hostA: data.hosts.indexOf(hCode) >= 0,
          hostB: data.hosts.indexOf(aCode) >= 0
        });
      }
      return matrixCache[key];
    }

    for (var sim = 0; sim < nSims; sim++) {
      // ---- 小组赛 ----
      var standings = {}; // code -> {pts, gf, ga, code, group}
      Object.keys(teams).forEach(function (c) {
        standings[c] = { code: c, group: teams[c].group, pts: 0, gf: 0, ga: 0 };
      });
      data.fixtures.forEach(function (fx) {
        var score;
        if (useResults && data.results[fx.n]) score = data.results[fx.n];
        else score = sampleScore(getMatrix(fx.h, fx.a, false).matrix);
        applyResult(standings[fx.h], standings[fx.a], score[0], score[1]);
      });

      // ---- 小组排名 ----
      var groups = {};
      Object.keys(standings).forEach(function (c) {
        var g = standings[c].group;
        (groups[g] = groups[g] || []).push(standings[c]);
      });
      var firsts = {}, seconds = {}, thirds = [];
      Object.keys(groups).forEach(function (g) {
        groups[g].sort(rankCmp);
        firsts[g] = groups[g][0].code;
        seconds[g] = groups[g][1].code;
        thirds.push(groups[g][2]);
      });
      // 最佳 8 个第三名（简化：不做 FIFA 官方落位表，按排名顺序填入 3RD 槽位）
      thirds.sort(rankCmp);
      var bestThirds = thirds.slice(0, 8).map(function (t) { return t.code; });

      // ---- 淘汰赛 ----
      var winners = {}; // matchNo -> code
      var thirdIdx = 0;
      function resolve(slot) {
        if (slot === "3RD") return bestThirds[thirdIdx++];
        var type = slot[0], rest = slot.slice(1);
        if (type === "1") return firsts[rest];
        if (type === "2") return seconds[rest];
        if (type === "W") return winners[parseInt(rest, 10)];
        return null;
      }
      function playKO(m) {
        var h = resolve(m.h), a = resolve(m.a);
        var pred = getMatrix(h, a, true);
        var sc = sampleScore(pred.matrix);
        var w;
        if (sc[0] > sc[1]) w = h;
        else if (sc[1] > sc[0]) w = a;
        else w = Math.random() < pred.pPenaltyA ? h : a; // 点球
        winners[m.n] = w;
        return w;
      }
      data.knockout.r32.forEach(playKO);
      data.knockout.r16.forEach(playKO);
      data.knockout.qf.forEach(playKO);
      var sfTeams = [];
      data.knockout.sf.forEach(function (m) {
        sfTeams.push(resolve(m.h), resolve(m.a));
        playKO(m);
      });
      sfTeams.forEach(function (c) { semiApp[c] = (semiApp[c] || 0) + 1; });
      var fh = resolve(data.knockout.final.h), fa = resolve(data.knockout.final.a);
      finalApp[fh] = (finalApp[fh] || 0) + 1;
      finalApp[fa] = (finalApp[fa] || 0) + 1;
      var champ = playKO(data.knockout.final);
      champion[champ] = (champion[champ] || 0) + 1;

      if (onProgress && sim % 500 === 0) onProgress(sim, nSims);
    }
    return { champion: champion, final: finalApp, semi: semiApp, nSims: nSims };
  }

  function applyResult(sh, sa, gh, ga) {
    sh.gf += gh; sh.ga += ga; sa.gf += ga; sa.ga += gh;
    if (gh > ga) sh.pts += 3;
    else if (gh < ga) sa.pts += 3;
    else { sh.pts += 1; sa.pts += 1; }
  }
  function rankCmp(a, b) {
    if (b.pts !== a.pts) return b.pts - a.pts;
    var gdA = a.gf - a.ga, gdB = b.gf - b.ga;
    if (gdB !== gdA) return gdB - gdA;
    if (b.gf !== a.gf) return b.gf - a.gf;
    return Math.random() - 0.5; // 同分同净胜同进球：随机（模拟用途足够）
  }

  /** 六维雷达数据（0-100）：进攻/防守/状态/大赛经验/教练/阵容深度 */
  function radarData(t, data) {
    var eloNorm = clamp((t.elo - 1400) / (2160 - 1400) * 100, 5, 100);
    var fifaNorm = clamp((90 - t.fifa) / 89 * 100, 5, 100);
    var host = data.hosts.indexOf(codeOf(t, data)) >= 0 ? 8 : 0;
    return {
      攻击力: Math.round(clamp(eloNorm * 0.7 + t.form * 0.3, 10, 99)),
      防守力: Math.round(clamp(eloNorm * 0.6 + fifaNorm * 0.4, 10, 99)),
      近期状态: Math.round(t.form),
      大赛经验: Math.round(clamp(fifaNorm * 0.8 + 20 - (t.fifa <= 20 ? 0 : 10), 10, 99)),
      教练博弈: Math.round(clamp(eloNorm * 0.4 + fifaNorm * 0.4 + 15, 10, 99)),
      综合实力: Math.round(clamp(eloNorm * 0.85 + host, 10, 99))
    };
  }
  function codeOf(t, data) {
    var codes = Object.keys(data.teams);
    for (var i = 0; i < codes.length; i++) if (data.teams[codes[i]] === t) return codes[i];
    return null;
  }

  /** 自检：在 node 中运行 `node core/engine.js` */
  function selfTest(data) {
    var t1 = { elo: 2000, form: 80 }, t2 = { elo: 1700, form: 60 };
    var p = predictMatch(t1, t2, {});
    var sum = p.pWin + p.pDraw + p.pLoss;
    console.assert(Math.abs(sum - 1) < 1e-9, "概率和应为1, 实际 " + sum);
    console.assert(p.pWin > p.pLoss, "强队胜率应更高");
    var q = predictMatch(t2, t1, {});
    console.assert(Math.abs(q.pLoss - p.pWin) < 1e-9, "对称性: 互换后胜负概率应互换");
    var ko = predictMatch(t1, t2, { knockout: true });
    console.assert(Math.abs(ko.pAdvanceA - (ko.pWin + ko.pDraw * ko.pPenaltyA)) < 1e-9, "晋级概率公式");
    if (data) {
      var r = simulateTournament(data, 200);
      var tot = 0; Object.keys(r.champion).forEach(function (k) { tot += r.champion[k]; });
      console.assert(tot === 200, "模拟次数守恒, 实际 " + tot);
    }
    console.log("✅ engine self-test passed");
  }

  var Engine = {
    predictMatch: predictMatch,
    simulateTournament: simulateTournament,
    radarData: radarData,
    sampleScore: sampleScore,
    selfTest: selfTest,
    CONST: { HOST_BONUS: HOST_BONUS, BASE_LAMBDA: BASE_LAMBDA, K_LAMBDA: K_LAMBDA, KO_FACTOR: KO_FACTOR, DC_RHO: DC_RHO }
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = Engine;
    if (require.main === module) {
      var path = require("path");
      var WC = require(path.join(__dirname, "data.js"));
      selfTest(WC);
    }
  } else {
    root.Engine = Engine;
  }
})(typeof window !== "undefined" ? window : this);
