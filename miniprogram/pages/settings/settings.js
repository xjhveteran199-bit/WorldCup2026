const L = require('../../utils/llm.js');
const app = getApp();

Page({
  data: {
    providers: [],
    provIdx: 0,
    models: [],
    modelIdx: 0,
    key: '',
    status: '',
    statusType: '',
    keyUrl: ''
  },

  onLoad() {
    const providers = L.PROVIDERS.map(p => p.name);
    this.setData({ providers });
    const cfg = app.globalData.llm || wx.getStorageSync('sjzy_llm');
    if (cfg) {
      const pi = L.PROVIDERS.findIndex(p => p.id === cfg.providerId);
      this.setData({ provIdx: pi < 0 ? 0 : pi, key: cfg.key || '' });
      this.refreshModels(() => {
        const mi = this.data.models.indexOf(cfg.model);
        if (mi >= 0) this.setData({ modelIdx: mi });
      });
    } else {
      this.refreshModels();
    }
  },

  refreshModels(cb) {
    const p = L.PROVIDERS[this.data.provIdx];
    this.setData({ models: p.models, modelIdx: 0, keyUrl: p.keyUrl }, cb);
  },

  onProv(e) { this.setData({ provIdx: +e.detail.value }, () => this.refreshModels()); },
  onModel(e) { this.setData({ modelIdx: +e.detail.value }); },
  onKey(e) { this.setData({ key: e.detail.value }); },

  save() {
    const key = this.data.key.trim();
    if (!key) { this.setData({ status: '请先粘贴 Key', statusType: 'err' }); return; }
    const p = L.PROVIDERS[this.data.provIdx];
    const model = this.data.models[this.data.modelIdx];
    this.setData({ status: '验证中…', statusType: 'wait' });
    L.verifyKey(p.id, key).then(() => {
      const cfg = { providerId: p.id, model, key };
      wx.setStorageSync('sjzy_llm', cfg);
      app.globalData.llm = cfg;
      this.setData({ status: '✅ 验证成功，已保存（仅存本机）', statusType: 'ok' });
      setTimeout(() => wx.navigateBack(), 900);
    }).catch(e => {
      this.setData({ status: '❌ ' + String(e.message || e).slice(0, 100), statusType: 'err' });
    });
  },

  clear() {
    wx.removeStorageSync('sjzy_llm');
    app.globalData.llm = null;
    this.setData({ key: '', status: '已清除', statusType: '' });
  },

  copyUrl() {
    wx.setClipboardData({ data: 'https://' + this.data.keyUrl, success: () => wx.showToast({ title: '申请地址已复制', icon: 'none' }) });
  }
});
