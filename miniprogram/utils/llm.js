/**
 * 硅基看球 · 小程序版 LLM 接入层（wx.request，非流式一次性返回）
 * 仅含 ICP 已备案、可加入 request 合法域名白名单的国产渠道（Gemini 不进小程序）
 * 纯逻辑函数（buildPrompt/parseAdjustment/applyAdjustment）与网页版一致
 */
const PROVIDERS = [
  { id: "deepseek", name: "DeepSeek 深度求索", base: "https://api.deepseek.com/chat/completions",
    models: ["deepseek-v4-flash", "deepseek-v4-pro"], keyUrl: "platform.deepseek.com" },
  { id: "kimi", name: "Kimi（月之暗面）", base: "https://api.moonshot.cn/v1/chat/completions",
    models: ["kimi-k2-0905-preview", "moonshot-v1-8k"], keyUrl: "platform.moonshot.cn" },
  { id: "glm", name: "智谱 GLM", base: "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    models: ["glm-4.6", "glm-4-flash"], keyUrl: "open.bigmodel.cn" },
  { id: "qwen", name: "通义千问", base: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    models: ["qwen-plus", "qwen-flash"], keyUrl: "bailian.console.aliyun.com" },
  { id: "minimax", name: "MiniMax", base: "https://api.minimaxi.com/v1/text/chatcompletion_v2",
    models: ["MiniMax-M2", "MiniMax-Text-01"], keyUrl: "platform.minimaxi.com" }
];

function getProvider(id) {
  return PROVIDERS.find(p => p.id === id) || null;
}

function pct(x) { return (x * 100).toFixed(1) + "%"; }

function buildPrompt(a, b, stat, extra, opts) {
  opts = opts || {};
  const lines = [
    "你是一名顶级足球战术分析师。请基于以下数据，对 2026 美加墨世界杯的这场比赛给出专业战术分析。",
    "",
    `【对阵】${a.zh}（${a.en}） vs ${b.zh}（${b.en}）${opts.knockout ? "（淘汰赛）" : "（小组赛）"}`,
    "",
    `【${a.zh}】Elo:${a.elo} FIFA排名:${a.fifa} 阵型:${a.formation}`,
    `风格:${a.style.join("/")} | 教练:${a.coach}（${a.coachStyle}）`,
    `核心球员:${a.stars.join("、")} | 近期状态:${a.form}/100 | 历史:${a.wcBest}`,
    "",
    `【${b.zh}】Elo:${b.elo} FIFA排名:${b.fifa} 阵型:${b.formation}`,
    `风格:${b.style.join("/")} | 教练:${b.coach}（${b.coachStyle}）`,
    `核心球员:${b.stars.join("、")} | 近期状态:${b.form}/100 | 历史:${b.wcBest}`,
    "",
    "【统计模型基线】（Elo + 双泊松 + Dixon-Coles）",
    `${a.zh}胜:${pct(stat.pWin)} 平:${pct(stat.pDraw)} ${b.zh}胜:${pct(stat.pLoss)}`,
    `期望进球 λ: ${stat.lambdaA.toFixed(2)} vs ${stat.lambdaB.toFixed(2)}`,
    `最可能比分: ${stat.topScores.slice(0, 3).map(s => s.a + "-" + s.b).join("、")}`
  ];
  if (extra && extra.trim()) lines.push("", "【最新情报（用户提供）】", extra.trim());
  lines.push(
    "",
    "请按以下结构输出（用中文，markdown 格式，每节 2-4 句，犀利专业有梗，适合社交媒体传播）：",
    `## 阵型相克：${a.formation} vs ${b.formation} 的体系博弈`,
    `## 教练斗法：${a.coach} vs ${b.coach}`,
    "## 三组关键对位（具体到球员）",
    "## X 因素（伤病/天气/心理/裁判尺度等变数）",
    "## 章鱼结论（一句话金句预测）",
    "",
    "最后，必须输出一个 JSON 代码块（```json 包裹），格式严格如下：",
    '```json',
    '{"adjustWin": 0, "adjustDraw": 0, "adjustLoss": 0, "reason": "调整理由（20字内）", "predictedScore": "2-1", "confidence": 75, "quote": "章鱼金句（25字内）"}',
    '```',
    "其中 adjustWin/adjustDraw/adjustLoss 是你对统计模型三项概率的修正（单位:百分点，每项限 -8 到 +8，三项之和必须为 0）；confidence 是信心（50-95）。"
  );
  return lines.join("\n");
}

function parseAdjustment(text) {
  const m = text.match(/```json\s*([\s\S]*?)```\s*$/) || text.match(/```json\s*([\s\S]*?)```/);
  if (!m) return null;
  try {
    const j = JSON.parse(m[1]);
    const clampPt = v => Math.max(-8, Math.min(8, Number(v) || 0));
    let aw = clampPt(j.adjustWin), ad = clampPt(j.adjustDraw), al = clampPt(j.adjustLoss);
    const sum = aw + ad + al;
    aw -= sum / 3; ad -= sum / 3; al -= sum / 3;
    return {
      adjustWin: aw, adjustDraw: ad, adjustLoss: al,
      reason: String(j.reason || "").slice(0, 40),
      predictedScore: String(j.predictedScore || ""),
      confidence: Math.max(50, Math.min(95, Number(j.confidence) || 70)),
      quote: String(j.quote || "").slice(0, 50)
    };
  } catch (e) { return null; }
}

function applyAdjustment(stat, adj) {
  if (!adj) return { pWin: stat.pWin, pDraw: stat.pDraw, pLoss: stat.pLoss, adjusted: false };
  const w = Math.max(0.01, stat.pWin + adj.adjustWin / 100);
  const d = Math.max(0.01, stat.pDraw + adj.adjustDraw / 100);
  const l = Math.max(0.01, stat.pLoss + adj.adjustLoss / 100);
  const s = w + d + l;
  return { pWin: w / s, pDraw: d / s, pLoss: l / s, adjusted: true };
}

/** wx.request 调用（非流式）。返回 Promise<string> 完整文本 */
function chat(opts) {
  const p = getProvider(opts.providerId);
  if (!p) return Promise.reject(new Error("未知渠道"));
  return new Promise((resolve, reject) => {
    wx.request({
      url: p.base,
      method: "POST",
      header: { "Content-Type": "application/json", "Authorization": "Bearer " + opts.key },
      data: {
        model: opts.model || p.models[0],
        messages: [{ role: "user", content: opts.prompt }],
        temperature: opts.temperature !== undefined ? opts.temperature : 0.8,
        stream: false,
        max_tokens: opts.maxTokens || 2048
      },
      timeout: 60000,
      success(res) {
        const j = res.data;
        if (res.statusCode !== 200 || (j && j.error)) {
          reject(new Error((j && j.error && j.error.message) || ("HTTP " + res.statusCode)));
          return;
        }
        try { resolve(j.choices[0].message.content); }
        catch (e) { reject(new Error("响应解析失败")); }
      },
      fail(err) { reject(new Error(err.errMsg || "网络请求失败")); }
    });
  });
}

function verifyKey(providerId, key) {
  return chat({ providerId, key, prompt: "回复:OK", maxTokens: 8, temperature: 0 })
    .then(() => getProvider(providerId).models[0]);
}

module.exports = {
  PROVIDERS, getProvider, buildPrompt, parseAdjustment, applyAdjustment, chat, verifyKey
};
