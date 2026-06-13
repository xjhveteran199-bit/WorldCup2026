/**
 * 数据源选择：优先用上次缓存的远程数据，否则用打包的兜底数据。
 * 远程数据由 app.js onLaunch 异步拉取并写入缓存，下次冷启动即生效（stale-while-revalidate）。
 */
const bundled = require('./data.js');
const CACHE_KEY = 'wc_data_cache';

function valid(d) {
  return d && d.teams && d.fixtures && d.fixtures.length && d.updated;
}

/** 当前应使用的数据：缓存的远程数据(更新日期不早于打包数据) 或 打包兜底 */
function current() {
  try {
    const c = wx.getStorageSync(CACHE_KEY);
    if (valid(c) && String(c.updated) >= String(bundled.updated)) return c;
  } catch (e) {}
  return bundled;
}

module.exports = { bundled, current, CACHE_KEY, valid };
