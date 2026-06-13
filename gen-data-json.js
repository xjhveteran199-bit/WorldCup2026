/**
 * 从 core/data.js 生成纯 JSON 数据端点 data.json
 * 供小程序远程拉取（Vercel 静态托管 → https://<domain>/data.json）
 * 每次更新比分后运行：node gen-data-json.js
 */
const fs = require('fs');
const path = require('path');
const WC = require('./core/data.js');
const out = path.join(__dirname, 'data.json');
fs.writeFileSync(out, JSON.stringify(WC));
console.log('✅ 已生成 data.json （' + Object.keys(WC.teams).length + ' 队 / ' +
  WC.fixtures.length + ' 场 / 已录入 ' + Object.keys(WC.results || {}).length + ' 场赛果）');
