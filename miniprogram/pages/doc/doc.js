const WC = require('../../utils/wc.js').current();
Page({
  data: { eloDate: WC.eloDate, dataUpdated: WC.updated },
  onShareAppMessage() { return { title: '🐙 硅基看球 · 看懂世界杯数据预测背后的模型', path: '/pages/predict/predict' }; }
});
