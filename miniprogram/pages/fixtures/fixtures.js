const WC = require('../../utils/wc.js').current();

Page({
  data: { days: [] },
  onLoad() {
    const byDay = {};
    WC.fixtures.forEach(fx => {
      const day = fx.d.slice(0, 5);
      const a = WC.teams[fx.h], b = WC.teams[fx.a];
      const res = WC.results[fx.n];
      (byDay[day] = byDay[day] || []).push({
        n: fx.n, h: fx.h, a: fx.a,
        hZh: a.zh, aZh: b.zh, hFlag: a.flag, aFlag: b.flag,
        g: fx.g, v: fx.v,
        time: fx.d.length > 5 ? fx.d.slice(6) : '已赛',
        res: res ? res[0] + ' : ' + res[1] : '',
        mid: res ? res[0] + ' : ' + res[1] : 'VS',
        played: !!res
      });
    });
    const today = new Date();
    const todayStr = String(today.getMonth() + 1).padStart(2, '0') + '-' + String(today.getDate()).padStart(2, '0');
    const days = Object.keys(byDay).map(day => ({
      day, label: day.replace('-', ' 月 ') + ' 日' + (day === todayStr ? ' · 今天' : ''),
      isToday: day === todayStr, matches: byDay[day]
    }));
    this.setData({ days });
  },
  toPredict(e) {
    const { h, a } = e.currentTarget.dataset;
    // switchTab 不能带参；reLaunch 可跳转 tab 页并携带参数
    wx.reLaunch({ url: `/pages/predict/predict?h=${h}&a=${a}` });
  }
});
