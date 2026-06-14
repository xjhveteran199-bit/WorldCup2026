const WC = require('../../utils/wc.js').current();
const E = require('../../utils/engine.js');

const WATERMARK = '小红书 @碳化硅SiC';

function pct(x) { return (x * 100).toFixed(1) + '%'; }

// 构建分组球队选择列表
function buildTeamList() {
  const groups = {};
  Object.keys(WC.teams).forEach(c => {
    const g = WC.teams[c].group;
    (groups[g] = groups[g] || []).push(c);
  });
  const list = [];
  Object.keys(groups).sort().forEach(g => {
    groups[g].forEach(c => {
      list.push({ code: c, label: g + '组 · ' + WC.teams[c].zh });
    });
  });
  return list;
}

Page({
  data: {
    teamList: [],
    idxA: 0,
    idxB: 0,
    aZh: '', bZh: '', aFlag: '', bFlag: '', aMeta: '', bMeta: '',
    knockout: false,
    result: null,        // 渲染就绪的结果对象
    sharePath: '',
    dataDate: WC.updated,
    resultsCount: Object.keys(WC.results || {}).length
  },

  onLoad(opts) {
    const teamList = buildTeamList();
    // 默认阿根廷 vs 法国
    const idxA = teamList.findIndex(t => t.code === 'ARG');
    const idxB = teamList.findIndex(t => t.code === 'FRA');
    this.setData({ teamList, idxA: idxA < 0 ? 0 : idxA, idxB: idxB < 0 ? 1 : idxB });
    // 支持从赛程页跳转带入对阵
    if (opts.h && opts.a) {
      const ia = teamList.findIndex(t => t.code === opts.h);
      const ib = teamList.findIndex(t => t.code === opts.a);
      if (ia >= 0 && ib >= 0) this.setData({ idxA: ia, idxB: ib, knockout: false });
    }
    this.refreshMeta();
    if (opts.h && opts.a) this.predict();
  },

  refreshMeta() {
    const ca = this.data.teamList[this.data.idxA].code;
    const cb = this.data.teamList[this.data.idxB].code;
    const a = WC.teams[ca], b = WC.teams[cb];
    const live = E.computeLiveRatings(WC);
    this.setData({
      aZh: a.zh, bZh: b.zh, aFlag: a.flag, bFlag: b.flag,
      aMeta: liveMeta(a, live[ca]),
      bMeta: liveMeta(b, live[cb])
    });
  },

  onPickA(e) { this.setData({ idxA: +e.detail.value }, () => this.refreshMeta()); },
  onPickB(e) { this.setData({ idxB: +e.detail.value }, () => this.refreshMeta()); },
  setMode(e) { this.setData({ knockout: e.currentTarget.dataset.ko === '1' }); },

  predict() {
    const ca = this.data.teamList[this.data.idxA].code;
    const cb = this.data.teamList[this.data.idxB].code;
    if (ca === cb) { wx.showToast({ title: '请选择两支不同球队', icon: 'none' }); return; }
    const live = E.computeLiveRatings(WC);
    const a = E.liveTeam(WC.teams[ca], live[ca]), b = E.liveTeam(WC.teams[cb], live[cb]);
    const pred = E.predictMatch(a, b, {
      knockout: this.data.knockout,
      hostA: WC.hosts.indexOf(ca) >= 0, hostB: WC.hosts.indexOf(cb) >= 0
    });
    pred.liveA = live[ca]; pred.liveB = live[cb];
    this._pred = pred; this._ca = ca; this._cb = cb;

    // 热力图 0-5（带颜色）
    let maxP = 0;
    for (let x = 0; x <= 5; x++) for (let y = 0; y <= 5; y++) maxP = Math.max(maxP, pred.matrix[x][y]);
    const heat = [];
    for (let x = 0; x <= 5; x++) {
      const row = [];
      for (let y = 0; y <= 5; y++) {
        const v = pred.matrix[x][y], r = v / maxP;
        let col;
        if (x > y) col = `rgba(255,61,46,${(r * 0.9).toFixed(2)})`;
        else if (x < y) col = `rgba(61,165,255,${(r * 0.85).toFixed(2)})`;
        else col = `rgba(255,211,77,${(r * 0.7).toFixed(2)})`;
        row.push({ v: (v * 100).toFixed(1), col });
      }
      heat.push(row);
    }

    // 六维对比
    const da = E.radarData(a, WC), db = E.radarData(b, WC);
    const dims = Object.keys(da).map(k => ({ name: k, a: da[k], b: db[k] }));

    // 因子
    const la = live[ca], lb = live[cb];
    const hasResult = la.played || lb.played;
    const eloVal = a.elo + (la.eloDelta ? '(' + fmt(la.eloDelta) + ')' : '') +
      ' vs ' + b.elo + (lb.eloDelta ? '(' + fmt(lb.eloDelta) + ')' : '');
    const factors = [
      { k: hasResult ? '实时Elo(含赛果)' : '基础 Elo', v: eloVal, cls: pred.eloA.base - pred.eloB.base > 0 ? 'pos' : 'neg' }
    ];
    if (hasResult) factors.push({ k: '已结合赛果', v: a.zh + ' ' + la.played + '场 / ' + b.zh + ' ' + lb.played + '场', cls: 'neu' });
    factors.push(
      { k: '状态修正', v: fmt(pred.eloA.formAdj) + ' / ' + fmt(pred.eloB.formAdj), cls: 'neu' },
      { k: '东道主加成', v: fmt(pred.eloA.hostAdj) + ' / ' + fmt(pred.eloB.hostAdj), cls: pred.eloA.hostAdj ? 'pos' : 'neu' },
      { k: '有效Elo差 Δ', v: (pred.drift > 0 ? '+' : '') + Math.round(pred.drift), cls: pred.drift > 0 ? 'pos' : 'neg' },
      { k: '期望进球 λ', v: pred.lambdaA.toFixed(2) + ' vs ' + pred.lambdaB.toFixed(2), cls: 'neu' },
      { k: '赛制', v: pred.knockout ? '淘汰赛(含点球)' : '小组赛', cls: 'neu' }
    );

    const top = pred.topScores[0];
    this.setData({
      result: {
        pW: pct(pred.pWin), pD: pct(pred.pDraw), pL: pct(pred.pLoss),
        wWidth: (pred.pWin * 100).toFixed(1), dWidth: (pred.pDraw * 100).toFixed(1), lWidth: (pred.pLoss * 100).toFixed(1),
        bigScore: top.a + ' : ' + top.b,
        topScores: pred.topScores.map(s => ({ s: s.a + '-' + s.b, p: pct(s.p) })),
        heat, dims, factors,
        koLine: pred.knockout
          ? `晋级概率：${a.zh} ${pct(pred.pAdvanceA)} / ${b.zh} ${pct(1 - pred.pAdvanceA)}`
          : ''
      },
      sharePath: ''
    });
  },

  // ====== 分享卡片（Canvas 2D 离屏）======
  makeCard() {
    if (!this._pred) return;
    wx.showLoading({ title: '生成中…' });
    const a = WC.teams[this._ca], b = WC.teams[this._cb];
    const pred = this._pred;
    const W = 1080, H = 1440;
    const canvas = wx.createOffscreenCanvas({ type: '2d', width: W, height: H });
    const ctx = canvas.getContext('2d');
    drawCard(ctx, W, H, a, b, pred);
    wx.canvasToTempFilePath({
      canvas,
      success: res => {
        wx.hideLoading();
        this.setData({ sharePath: res.tempFilePath });
      },
      fail: () => { wx.hideLoading(); wx.showToast({ title: '生成失败', icon: 'none' }); }
    });
  },

  saveCard() {
    if (!this.data.sharePath) return;
    wx.saveImageToPhotosAlbum({
      filePath: this.data.sharePath,
      success: () => wx.showToast({ title: '已保存到相册', icon: 'success' }),
      fail: err => {
        if (/auth|deny/i.test(err.errMsg)) {
          wx.showModal({
            title: '需要相册权限', content: '请在设置中允许保存到相册',
            confirmText: '去设置', success: r => { if (r.confirm) wx.openSetting(); }
          });
        }
      }
    });
  },

  // 转发引流
  onShareAppMessage() {
    const r = this.data.result;
    const title = r
      ? `🐙 ${this.data.aZh} vs ${this.data.bZh} 数据预测 ${r.bigScore}`
      : '🐙 硅基看球 · 2026世界杯数据预测';
    return { title, path: '/pages/predict/predict' };
  },
  onShareTimeline() {
    return { title: '🐙 硅基看球 · 数据模型算遍2026世界杯每场比赛' };
  }
});

function fmt(v) { return (v > 0 ? '+' : '') + v; }
function liveMeta(t, L) {
  const elo = (L && L.played > 0) ? 'Elo ' + L.elo + '(' + fmt(L.eloDelta) + ')' : 'Elo ' + t.elo;
  const rec = (L && L.played > 0) ? ' · ' + L.w + '胜' + L.d + '平' + L.l + '负' : '';
  return elo + ' · FIFA#' + t.fifa + ' · ' + t.formation + rec;
}

// Canvas 2D 绘制分享卡片（火焰风）
function drawCard(ctx, W, H, a, b, pred) {
  const g = ctx.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, '#1c0608'); g.addColorStop(.55, '#2a0d0a'); g.addColorStop(1, '#341103');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  let rg = ctx.createRadialGradient(W * .8, H * .1, 0, W * .8, H * .1, 460);
  rg.addColorStop(0, 'rgba(255,61,46,.40)'); rg.addColorStop(1, 'transparent');
  ctx.fillStyle = rg; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,77,46,.07)'; ctx.lineWidth = 22;
  for (let i = -4; i < 14; i++) { ctx.beginPath(); ctx.moveTo(i * 110, 0); ctx.lineTo(i * 110 + 300, H); ctx.stroke(); }

  ctx.textAlign = 'center';
  ctx.fillStyle = '#fff'; ctx.font = 'italic 900 58px sans-serif';
  ctx.fillText('🐙 硅基看球', W / 2, 110);
  ctx.fillStyle = '#d8a48f'; ctx.font = 'italic 32px sans-serif';
  ctx.fillText('2026 美加墨世界杯 · 数据预测卡', W / 2, 168);

  ctx.font = '120px sans-serif';
  ctx.fillText(a.flag, W * .27, 400); ctx.fillText(b.flag, W * .73, 400);
  ctx.fillStyle = '#fff'; ctx.font = '900 52px sans-serif';
  ctx.fillText(a.zh, W * .27, 488); ctx.fillText(b.zh, W * .73, 488);
  ctx.fillStyle = '#ff7a1f'; ctx.font = 'italic 900 64px sans-serif'; ctx.fillText('VS', W / 2, 420);
  ctx.fillStyle = '#c49b8b'; ctx.font = '26px sans-serif';
  ctx.fillText('Elo ' + a.elo + ' · ' + a.formation, W * .27, 530);
  ctx.fillText('Elo ' + b.elo + ' · ' + b.formation, W * .73, 530);

  const top = pred.topScores[0];
  const grad = ctx.createLinearGradient(W * .3, 0, W * .7, 0);
  grad.addColorStop(0, '#ff2d2e'); grad.addColorStop(1, '#ffae00');
  ctx.fillStyle = grad; ctx.font = 'italic 900 175px sans-serif';
  ctx.fillText(top.a + ' : ' + top.b, W / 2, 740);
  ctx.fillStyle = '#c49b8b'; ctx.font = '30px sans-serif';
  ctx.fillText('模型最可能比分（' + pct(top.p) + '）', W / 2, 800);

  const pw = pred.pWin, pd = pred.pDraw, pl = pred.pLoss;
  const bx = 90, bw = W - 180, by = 880, bh = 64;
  const wg = ctx.createLinearGradient(bx, 0, bx + bw * pw, 0);
  wg.addColorStop(0, '#ff2d2e'); wg.addColorStop(1, '#ff7a1f');
  ctx.fillStyle = wg; ctx.fillRect(bx, by, bw * pw, bh);
  ctx.fillStyle = '#7a5c50'; ctx.fillRect(bx + bw * pw, by, bw * pd, bh);
  ctx.fillStyle = '#3da5ff'; ctx.fillRect(bx + bw * (pw + pd), by, bw * pl, bh);
  ctx.fillStyle = '#fff'; ctx.font = '700 30px sans-serif';
  if (pw > .12) ctx.fillText(pct(pw), bx + bw * pw / 2, by + 42);
  if (pd > .12) ctx.fillText(pct(pd), bx + bw * (pw + pd / 2), by + 42);
  if (pl > .12) ctx.fillText(pct(pl), bx + bw * (pw + pd + pl / 2), by + 42);
  ctx.fillStyle = '#c6cee6'; ctx.font = '30px sans-serif';
  ctx.textAlign = 'left'; ctx.fillText(a.zh + ' 胜', bx, by + 110);
  ctx.textAlign = 'right'; ctx.fillText(b.zh + ' 胜', bx + bw, by + 110);
  ctx.textAlign = 'center'; ctx.fillText('平', bx + bw * (pw + pd / 2), by + 110);

  ctx.fillStyle = '#c49b8b'; ctx.font = '28px sans-serif';
  ctx.fillText('Elo × 双泊松 × Dixon-Coles 统计模型', W / 2, H - 130);
  ctx.fillStyle = '#ffae00'; ctx.font = 'italic 700 34px sans-serif';
  ctx.fillText(WATERMARK, W / 2, H - 78);
  ctx.fillStyle = '#8a625a'; ctx.font = '22px sans-serif';
  ctx.fillText('仅供娱乐 · 理性看球', W / 2, H - 38);
}
