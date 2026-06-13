const WC = require('../../utils/wc.js').current();
const E = require('../../utils/engine.js');

function pct(x) { return (x * 100).toFixed(1) + '%'; }

Page({
  data: {
    running: false, progress: 0, done: false, ranks: [], meta: '', sharePath: ''
  },

  runSim() {
    if (this.data.running) return;
    this.setData({ running: true, progress: 0, done: false, sharePath: '' });
    const N = 10000, CHUNK = 1000;
    let doneN = 0;
    const agg = { champion: {}, final: {}, semi: {} };
    const step = () => {
      const r = E.simulateTournament(WC, CHUNK, true);
      ['champion', 'final', 'semi'].forEach(k => {
        Object.keys(r[k]).forEach(c => { agg[k][c] = (agg[k][c] || 0) + r[k][c]; });
      });
      doneN += CHUNK;
      this.setData({ progress: Math.round(doneN / N * 100) });
      if (doneN < N) setTimeout(step, 0);
      else this.finish(agg, N);
    };
    setTimeout(step, 30);
  },

  finish(agg, N) {
    const rows = Object.keys(agg.champion).map(c => ({ c, p: agg.champion[c] / N }))
      .sort((a, b) => b.p - a.p).slice(0, 12);
    const maxP = rows[0].p;
    const ranks = rows.map((r, i) => {
      const t = WC.teams[r.c];
      return { rank: i + 1, flag: t.flag, zh: t.zh, pct: (r.p * 100).toFixed(1), width: (r.p / maxP * 100).toFixed(1), top3: i < 3 };
    });
    this._rows = rows;
    this.setData({ running: false, done: true, ranks, meta: '已模拟 ' + N + ' 届 · 已赛比分锁定' });
  },

  makeCard() {
    if (!this._rows) return;
    wx.showLoading({ title: '生成中…' });
    const W = 1080, H = 1440;
    const canvas = wx.createOffscreenCanvas({ type: '2d', width: W, height: H });
    const ctx = canvas.getContext('2d');
    drawSimCard(ctx, W, H, this._rows);
    wx.canvasToTempFilePath({
      canvas,
      success: res => { wx.hideLoading(); this.setData({ sharePath: res.tempFilePath }); },
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
          wx.showModal({ title: '需要相册权限', content: '请在设置中允许保存到相册', confirmText: '去设置',
            success: r => { if (r.confirm) wx.openSetting(); } });
        }
      }
    });
  },

  onShareAppMessage() {
    const top = this.data.ranks[0];
    return { title: top ? `🐙 AI 模拟一万届世界杯：${top.zh} 夺冠概率 ${top.pct}%` : '🐙 AI 模拟一万届世界杯', path: '/pages/sim/sim' };
  },
  onShareTimeline() { return { title: '🐙 AI 模拟一万届世界杯，看看谁夺冠' }; }
});

function drawSimCard(ctx, W, H, rows) {
  const g = ctx.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, '#1c0608'); g.addColorStop(.55, '#2a0d0a'); g.addColorStop(1, '#341103');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  const rg = ctx.createRadialGradient(W * .8, H * .1, 0, W * .8, H * .1, 460);
  rg.addColorStop(0, 'rgba(255,61,46,.40)'); rg.addColorStop(1, 'transparent');
  ctx.fillStyle = rg; ctx.fillRect(0, 0, W, H);
  ctx.textAlign = 'center';
  ctx.fillStyle = '#fff'; ctx.font = 'italic 900 58px sans-serif';
  ctx.fillText('🐙 硅基看球', W / 2, 110);
  ctx.fillStyle = '#d8a48f'; ctx.font = 'italic 32px sans-serif';
  ctx.fillText('AI 模拟了 10000 届世界杯', W / 2, 168);
  ctx.fillStyle = '#ffd34d'; ctx.font = 'italic 900 64px sans-serif';
  ctx.fillText('👑 夺冠概率排行榜', W / 2, 280);

  const top10 = rows.slice(0, 10), maxP = top10[0].p, y0 = 380, rh = 78;
  top10.forEach((r, i) => {
    const t = WC.teams[r.c], y = y0 + i * rh;
    ctx.textAlign = 'left';
    ctx.fillStyle = i < 3 ? '#ffd34d' : '#c49b8b'; ctx.font = '700 34px sans-serif';
    ctx.fillText(String(i + 1).padStart(2, '0'), 90, y + 40);
    ctx.font = '44px sans-serif'; ctx.fillText(t.flag, 160, y + 44);
    ctx.fillStyle = '#fdeee4'; ctx.font = '700 36px sans-serif'; ctx.fillText(t.zh, 232, y + 42);
    const bx = 460, bw = 420;
    ctx.fillStyle = 'rgba(255,255,255,.08)'; roundRect(ctx, bx, y + 8, bw, 40, 12); ctx.fill();
    const gg = ctx.createLinearGradient(bx, 0, bx + bw, 0);
    gg.addColorStop(0, '#ff2d2e'); gg.addColorStop(1, '#ffae00');
    ctx.fillStyle = gg; roundRect(ctx, bx, y + 8, Math.max(24, bw * r.p / maxP), 40, 12); ctx.fill();
    ctx.fillStyle = '#ffae00'; ctx.font = '700 34px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText((r.p * 100).toFixed(1) + '%', W - 80, y + 42);
    ctx.textAlign = 'center';
  });
  ctx.fillStyle = '#c49b8b'; ctx.font = '28px sans-serif';
  ctx.fillText('蒙特卡洛 × Elo × 双泊松 · 已赛比分锁定', W / 2, H - 130);
  ctx.fillStyle = '#ffae00'; ctx.font = 'italic 700 34px sans-serif';
  ctx.fillText('小红书 @碳化硅SiC', W / 2, H - 78);
  ctx.fillStyle = '#8a625a'; ctx.font = '22px sans-serif';
  ctx.fillText('仅供娱乐 · 理性看球', W / 2, H - 38);
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
