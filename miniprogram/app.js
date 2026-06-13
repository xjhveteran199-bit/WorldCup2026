// 硅基看球 · 2026美加墨世界杯AI预测
const wc = require('./utils/wc.js');

// 远程数据端点（网页版每天自动更新 → 小程序无需重新提审即可获取新比分）
const DATA_URL = 'https://worldcup2026-blond.vercel.app/data.json';

App({
  globalData: {
    llm: null,       // AI 配置（BYOK：用户自带 API Key）
    dataUpdated: ''  // 当前数据更新日期
  },

  onLaunch() {
    try {
      const cfg = wx.getStorageSync('sjzy_llm');
      if (cfg) this.globalData.llm = cfg;
    } catch (e) {}
    this.globalData.dataUpdated = wc.current().updated;
    this.fetchRemoteData();
  },

  // 拉取最新数据，缓存供下次冷启动使用（拉取失败则继续用现有数据，不影响体验）
  fetchRemoteData() {
    wx.request({
      url: DATA_URL,
      method: 'GET',
      timeout: 8000,
      success: res => {
        const d = res.data;
        if (res.statusCode === 200 && wc.valid(d) &&
            String(d.updated) >= String(wc.bundled.updated)) {
          try { wx.setStorageSync(wc.CACHE_KEY, d); } catch (e) {}
        }
      },
      fail: () => {}  // 静默失败，用打包/上次缓存数据兜底
    });
  }
});
