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
  var WC_HOME_ADV = 65;        // 数据模型的主场 Elo 加成（与训练一致）
  var WC_IMP = 2;              // 世界杯=大赛重要度

  // 载入数据模型权重（离线训练，端内推理；缺失则回退手调公式）
  var MODEL = null;
  if (typeof WC_MODEL !== "undefined") MODEL = WC_MODEL;
  else if (typeof module !== "undefined" && module.exports) { try { MODEL = require("./model.js"); } catch (e) {} }
  else if (root && root.WC_MODEL) MODEL = root.WC_MODEL;

  // MLP 前向推理：特征向量 → [logλ_home, logλ_away] → [λ_home, λ_away]
  function modelLambdas(feat) {
    var x = [], i, j;
    for (i = 0; i < feat.length; i++) x[i] = (feat[i] - MODEL.mean[i]) / MODEL.std[i];
    var a = x;
    for (var L = 0; L < MODEL.layers.length; L++) {
      var ly = MODEL.layers[L], W = ly.W, b = ly.b, out = [];
      for (j = 0; j < W.length; j++) {
        var s = b[j], wj = W[j];
        for (i = 0; i < wj.length; i++) s += wj[i] * a[i];
        out[j] = ly.act === "relu" ? Math.max(0, s) : s;
      }
      a = out;
    }
    var lo = MODEL.clip[0], hi = MODEL.clip[1];
    return [Math.exp(Math.max(lo, Math.min(hi, a[0]))), Math.exp(Math.max(lo, Math.min(hi, a[1])))];
  }

  // 由球队对象组特征 → 数据模型期望进球；返回 null 表示无模型
  function lambdasFromModel(teamA, teamB, opts) {
    if (!MODEL) return null;
    var eh = teamA.elo, ea = teamB.elo;
    var haH = opts.hostA ? WC_HOME_ADV : 0, haA = opts.hostB ? WC_HOME_ADV : 0;
    var diff = (eh + haH) - (ea + haA);
    var neutral = (opts.hostA || opts.hostB) ? 0 : 1;
    var fh = (teamA._formRatio != null) ? teamA._formRatio : formRatio(teamA.form);
    var fa = (teamB._formRatio != null) ? teamB._formRatio : formRatio(teamB.form);
    // 特征顺序须与训练一致：elo_h, elo_a, diff, neutral, imp, form_h, form_a
    return modelLambdas([eh, ea, diff, neutral, WC_IMP, fh, fa]);
  }
  // 引擎 form(0-100) → 训练用的近期积分率(0~1) 的近似
  function formRatio(form) { return clamp((form - 40) / 55, 0, 1); }

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
    // 优先用数据模型（离线训练）的期望进球；无模型则回退手调公式
    var ml = lambdasFromModel(teamA, teamB, opts), la, lb;
    if (ml) {
      la = clamp(ml[0] * koMul, 0.15, 4.5);
      lb = clamp(ml[1] * koMul, 0.15, 4.5);
    } else {
      la = clamp(BASE_LAMBDA * Math.exp(K_LAMBDA * dr) * koMul, 0.15, 4.5);
      lb = clamp(BASE_LAMBDA * Math.exp(-K_LAMBDA * dr) * koMul, 0.15, 4.5);
    }

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

  var ELO_K = 60;              // 世界杯赛事权重（FIFA/eloratings 体系）
  var FORM_BLEND = 0.5;        // 实时状态 = 0.5×赛前评估 + 0.5×实战表现

  /**
   * 根据 data.results 里已结束的比赛，动态更新每支球队的实时 Elo 与近期状态。
   * 每场比赛后按 Elo 公式 + 净胜球权重更新评分，并由实战积分/净胜球重算 form。
   * 返回 { [code]: { elo, baseElo, eloDelta, form, baseForm, played, w, d, l, gf, ga, recent } }
   */
  function computeLiveRatings(data) {
    var R = {};
    Object.keys(data.teams).forEach(function (c) {
      var t = data.teams[c];
      R[c] = { elo: t.elo, baseElo: t.elo, baseForm: t.form,
               played: 0, w: 0, d: 0, l: 0, gf: 0, ga: 0, recent: [] };
    });
    // 按赛程顺序逐场套用真实结果
    data.fixtures.forEach(function (fx) {
      var res = data.results && data.results[fx.n];
      if (!res) return;
      var h = R[fx.h], a = R[fx.a];
      if (!h || !a) return;
      var gh = res[0], ga = res[1];
      var dr = h.elo - a.elo;
      var we = 1 / (1 + Math.pow(10, -dr / 400));
      var S = gh > ga ? 1 : (gh < ga ? 0 : 0.5);
      // 净胜球权重（大胜更能体现实力差）
      var gd = Math.abs(gh - ga);
      var gdMul = gd <= 1 ? 1 : (gd === 2 ? 1.5 : (11 + gd) / 8);
      var delta = ELO_K * gdMul * (S - we);
      h.elo += delta; a.elo -= delta;
      h.played++; a.played++;
      h.gf += gh; h.ga += ga; a.gf += ga; a.ga += gh;
      if (gh > ga) { h.w++; a.l++; } else if (gh < ga) { h.l++; a.w++; } else { h.d++; a.d++; }
      h.recent.push({ pts: gh > ga ? 3 : (gh === ga ? 1 : 0), gd: gh - ga });
      a.recent.push({ pts: ga > gh ? 3 : (gh === ga ? 1 : 0), gd: ga - gh });
    });
    // 由实战表现重算 form，与赛前评估融合
    Object.keys(R).forEach(function (c) {
      var r = R[c];
      if (r.played === 0) {
        r.form = r.baseForm; r.eloDelta = 0; r.elo = Math.round(r.elo);
        r.formRatio = 0.5; return;  // 近期积分率默认 0.5（与训练一致）
      }
      var ppg = r.recent.reduce(function (s, m) { return s + m.pts; }, 0) / r.played; // 0~3
      var gdAvg = r.recent.reduce(function (s, m) { return s + m.gd; }, 0) / r.played;
      var perf = clamp(55 + ppg * 12 + gdAvg * 4, 40, 98);
      r.form = Math.round(FORM_BLEND * r.baseForm + (1 - FORM_BLEND) * perf);
      r.eloDelta = Math.round(r.elo - r.baseElo);
      r.elo = Math.round(r.elo);
      r.formRatio = ppg / 3;  // 0~1，喂给数据模型的近期状态特征
    });
    return R;
  }

  // 数据模型派生的「进攻/防守指数」（0-99）：该队 vs 平均对手(Elo≈1600)的期望进/失球
  var AVG_OPP = { elo: 1600, form: 70, _formRatio: 0.5 };
  function teamIndices(team) {
    if (!MODEL) return null;
    var l = lambdasFromModel(team, AVG_OPP, {});  // 中立场，该队进攻 / 对手进攻=该队失球
    return {
      attackIdx: Math.round(clamp(50 + (l[0] - 1.3) * 30, 1, 99)),
      defenseIdx: Math.round(clamp(50 - (l[1] - 1.3) * 30, 1, 99))
    };
  }

  /** 构造一个套用实时评分的球队对象（用于 predictMatch） */
  function liveTeam(team, liveRow) {
    if (!liveRow) return team;
    var o = {};
    for (var k in team) if (team.hasOwnProperty(k)) o[k] = team[k];
    o.elo = liveRow.elo; o.form = liveRow.form;
    o._eloDelta = liveRow.eloDelta; o._played = liveRow.played;
    o._formRatio = (liveRow.formRatio != null) ? liveRow.formRatio : 0.5;
    o._record = liveRow.w + '胜' + liveRow.d + '平' + liveRow.l + '负';
    var idx = teamIndices(o);
    if (idx) { o.attackIdx = idx.attackIdx; o.defenseIdx = idx.defenseIdx; }
    return o;
  }

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
    // 用真实结果动态修正后的实时评分，作为未赛场次的预测基础
    var live = computeLiveRatings(data);

    function getMatrix(hCode, aCode, knockout) {
      var key = hCode + "|" + aCode + "|" + (knockout ? 1 : 0);
      if (!matrixCache[key]) {
        matrixCache[key] = predictMatch(liveTeam(teams[hCode], live[hCode]), liveTeam(teams[aCode], live[aCode]), {
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
    // 攻防优先用数据模型派生指数（liveTeam 已附加），否则回退 Elo 估算
    var att = (t.attackIdx != null) ? t.attackIdx : Math.round(clamp(eloNorm * 0.7 + t.form * 0.3, 10, 99));
    var def = (t.defenseIdx != null) ? t.defenseIdx : Math.round(clamp(eloNorm * 0.6 + fifaNorm * 0.4, 10, 99));
    return {
      攻击力: att,
      防守力: def,
      近期状态: Math.round(t.form),
      大赛经验: Math.round(clamp(fifaNorm * 0.8 + 20 - (t.fifa <= 20 ? 0 : 10), 10, 99)),
      教练博弈: Math.round(clamp(eloNorm * 0.4 + fifaNorm * 0.4 + 15, 10, 99)),
      综合实力: Math.round(clamp(eloNorm * 0.85 + host, 10, 99))
    };
  }
  function codeOf(t, data) {
    var codes = Object.keys(data.teams);
    // zh 队名唯一，兼容实时评分生成的球队副本
    for (var i = 0; i < codes.length; i++) {
      var dt = data.teams[codes[i]];
      if (dt === t || dt.zh === t.zh) return codes[i];
    }
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
    // 数据模型有微弱主场偏置（真实学到），不再严格对称，放宽容差
    console.assert(Math.abs(q.pLoss - p.pWin) < 0.08, "近似对称: 互换后胜负概率应大致互换, 差 " + Math.abs(q.pLoss - p.pWin).toFixed(3));
    var ko = predictMatch(t1, t2, { knockout: true });
    console.assert(Math.abs(ko.pAdvanceA - (ko.pWin + ko.pDraw * ko.pPenaltyA)) < 1e-9, "晋级概率公式");
    if (data) {
      var r = simulateTournament(data, 200);
      var tot = 0; Object.keys(r.champion).forEach(function (k) { tot += r.champion[k]; });
      console.assert(tot === 200, "模拟次数守恒, 实际 " + tot);
      // 实时评分自检：胜者 Elo 应升、负者应降，未赛球队保持基准
      var live = computeLiveRatings(data);
      var anyPlayed = false;
      Object.keys(live).forEach(function (c) {
        var L = live[c];
        if (L.played === 0) { console.assert(L.eloDelta === 0, c + " 未赛却变动了 Elo"); }
        else { anyPlayed = true; console.assert(L.form >= 40 && L.form <= 98, c + " form 越界"); }
      });
      // 已赛场次的胜者 eloDelta 应 > 负者
      if (data.results) {
        var firstN = Object.keys(data.results)[0];
        if (firstN) {
          var fx = null; data.fixtures.forEach(function (f) { if (f.n == firstN) fx = f; });
          if (fx) {
            var rr = data.results[firstN];
            if (rr[0] > rr[1]) console.assert(live[fx.h].eloDelta > live[fx.a].eloDelta, "胜者实时Elo应更高");
          }
        }
      }
    }
    // 数据模型：进攻/防守指数应区分强弱队
    if (MODEL) {
      var strong = teamIndices({ elo: 2050, form: 85, _formRatio: 0.8 });
      var weak = teamIndices({ elo: 1480, form: 55, _formRatio: 0.3 });
      console.assert(strong.attackIdx > weak.attackIdx, "强队进攻指数应更高");
      console.assert(strong.defenseIdx > weak.defenseIdx, "强队防守指数应更高");
      console.log("  数据模型已载入 · 回测logloss " + (MODEL.meta ? MODEL.meta.test_logloss + " vs 基线 " + MODEL.meta.baseline_logloss : "?"));
    } else {
      console.log("  ⚠️ 未载入数据模型，使用回退公式");
    }
    console.log("✅ engine self-test passed");
  }

  var Engine = {
    predictMatch: predictMatch,
    teamIndices: teamIndices,
    simulateTournament: simulateTournament,
    computeLiveRatings: computeLiveRatings,
    liveTeam: liveTeam,
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
