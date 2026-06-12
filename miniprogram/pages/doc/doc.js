Page({
  data: { eloDate: '2026-06-10' },
  goSettings() { wx.navigateTo({ url: '/pages/settings/settings' }); },
  onShareAppMessage() { return { title: '🐙 硅基看球 · 看懂世界杯AI预测背后的科学', path: '/pages/predict/predict' }; }
});
