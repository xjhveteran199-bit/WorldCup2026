const WC = require('../../utils/wc.js').current();
const E = require('../../utils/engine.js');
let META = {};
try { META = require('../../utils/model.js').meta || {}; } catch (e) {}

const TR = { W: '主胜', D: '平局', L: '客胜' };
function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

function buildTrack() {
  if (!E.backtestWC) return null;
  var bt = E.backtestWC(WC);
  if (!bt || !bt.n) return null;
  var rows = bt.rounds.slice(-16).reverse().map(function (r) {
    var h = WC.teams[r.h], a = WC.teams[r.a];
    return {
      key: r.n,
      txt: h.flag + h.zh + ' ' + r.gh + '-' + r.ga + ' ' + a.zh + a.flag,
      top: '看好' + TR[r.top] + ' ' + Math.round(r.topP * 100) + '%',
      mark: (r.hit ? '✓' : '✗') + (r.scoreHit ? ' 🎯' : ''),
      hit: r.hit
    };
  });
  return {
    n: bt.n,
    topHit: Math.round(bt.topHitRate * 100),
    scoreHit: bt.scoreHit,
    logloss: bt.logloss.toFixed(3),
    coin: bt.coinLogloss.toFixed(2),
    brier: bt.brier.toFixed(3),
    beat: bt.logloss < bt.coinLogloss,
    rows: rows
  };
}

Page({
  data: {
    eloDate: WC.eloDate,
    dataUpdated: WC.updated,
    track: buildTrack(),
    mdSamples: fmtInt(META.samples || 17861),
    mdLL: META.test_logloss || '0.887',
    mdBase: META.baseline_logloss || '0.899'
  },
  onShareAppMessage() { return { title: '🐙 硅基看球 · 看懂世界杯数据预测背后的模型', path: '/pages/predict/predict' }; }
});
