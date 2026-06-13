/**
 * 生成预览二维码（管理员/开发者本人微信扫码即可真机体验，无需审核）
 * 用法：node preview-mp.js
 */
const ci = require('miniprogram-ci');
const path = require('path');
const fs = require('fs');

const APPID = 'wxaea84e45a0e5975e';
const KEY = path.join(__dirname, `private.${APPID}.key`);
const OUT = path.join(__dirname, 'preview-qr.jpg');

if (!fs.existsSync(KEY)) {
  console.error(`❌ 未找到密钥文件: ${KEY}`);
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
  console.log('🔄 生成预览二维码 …');
  await ci.preview({
    project,
    desc: '管理员体验版预览',
    setting: { es6: true, minify: true },
    qrcodeFormat: 'image',
    qrcodeOutputDest: OUT,
    pagePath: 'pages/predict/predict',
    onProgressUpdate: () => {}
  });
  console.log('✅ 二维码已生成:', OUT);
  console.log('用微信「扫一扫」扫描即可在手机上体验（需用绑定为管理员/开发者/体验成员的微信号）。');
})().catch(e => {
  console.error('❌ 失败:', e.message || e);
  if (/whitelist|白名单|invalid ip/i.test(String(e.message))) {
    console.error('→ IP 白名单拦截：到 开发设置→小程序代码上传 关闭 IP 白名单。');
  }
  process.exit(1);
});
