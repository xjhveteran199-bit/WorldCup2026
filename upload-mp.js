/**
 * 一键上传小程序代码（无需微信开发者工具）
 * 用法：node upload-mp.js [版本号] [备注]
 * 前置：把 mp.weixin.qq.com 下载的 private.wxaea84e45a0e5975e.key 放在本目录
 */
const ci = require('miniprogram-ci');
const path = require('path');
const fs = require('fs');

const APPID = 'wxaea84e45a0e5975e';
const KEY = path.join(__dirname, `private.${APPID}.key`);
const VERSION = process.argv[2] || '1.0.0';
const DESC = process.argv[3] || '硅基看球 — 2026世界杯统计模型与数据分析工具';

if (!fs.existsSync(KEY)) {
  console.error(`❌ 未找到密钥文件: ${KEY}
请到 mp.weixin.qq.com → 开发管理 → 开发设置 → 小程序代码上传 → 生成密钥并下载，
将 private.${APPID}.key 放到本目录后重试。`);
  process.exit(1);
}

(async () => {
  const project = new ci.Project({
    appid: APPID,
    type: 'miniProgram',
    projectPath: path.join(__dirname, 'miniprogram'),
    privateKeyPath: KEY,
    ignores: ['node_modules/**/*']
  });
  console.log(`⬆️  上传版本 ${VERSION} …`);
  const result = await ci.upload({
    project,
    version: VERSION,
    desc: DESC,
    setting: { es6: true, minify: true },
    onProgressUpdate: m => { if (m && m._status === 'done') process.stdout.write('.'); }
  });
  console.log('\n✅ 上传成功！', JSON.stringify(result.subPackageInfo || result, null, 2));
  console.log('下一步：到 mp.weixin.qq.com → 管理 → 版本管理，把这个开发版本「提交审核」。');
})().catch(e => {
  console.error('\n❌ 上传失败:', e.message || e);
  if (/whitelist|白名单|invalid ip/i.test(String(e.message))) {
    console.error('→ 这是 IP 白名单拦截：到 开发设置→小程序代码上传 里关闭 IP 白名单，或把当前 IP 加入白名单。');
  }
  process.exit(1);
});
