// 硅基看球 · 2026美加墨世界杯AI预测
App({
  globalData: {
    // AI 配置（BYOK：用户自带 API Key），存于本地
    llm: null
  },
  onLaunch() {
    try {
      const cfg = wx.getStorageSync('sjzy_llm');
      if (cfg) this.globalData.llm = cfg;
    } catch (e) {}
  }
});
