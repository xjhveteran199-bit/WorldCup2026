/**
 * 硅基章鱼 · 多渠道 LLM 接入层（OpenAI-compatible 统一调用）
 * 平台无关：网页直接用 fetch；小程序注入 wx.request 适配器
 */
(function (root) {
  "use strict";

  var PROVIDERS = [
    { id: "deepseek", name: "DeepSeek 深度求索", base: "https://api.deepseek.com/chat/completions",
      models: ["deepseek-v4-flash", "deepseek-v4-pro"], keyUrl: "https://platform.deepseek.com/api_keys",
      keyPrefix: "sk-", search: false },
    { id: "kimi", name: "Kimi（月之暗面）", base: "https://api.moonshot.cn/v1/chat/completions",
      models: ["kimi-k2-0905-preview", "moonshot-v1-8k"], keyUrl: "https://platform.moonshot.cn/console/api-keys",
      keyPrefix: "sk-", search: false },
    { id: "glm", name: "智谱 GLM", base: "https://open.bigmodel.cn/api/paas/v4/chat/completions",
      models: ["glm-4.6", "glm-4-flash"], keyUrl: "https://open.bigmodel.cn/usercenter/apikeys",
      keyPrefix: "", search: true },
    { id: "qwen", name: "通义千问", base: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
      models: ["qwen-plus", "qwen-flash"], keyUrl: "https://bailian.console.aliyun.com/?apiKey=1",
      keyPrefix: "sk-", search: false },
    { id: "minimax", name: "MiniMax", base: "https://api.minimaxi.com/v1/text/chatcompletion_v2",
      models: ["MiniMax-M2", "MiniMax-Text-01"], keyUrl: "https://platform.minimaxi.com/user-center/basic-information/interface-key",
      keyPrefix: "", search: false },
    { id: "gemini", name: "Google Gemini（需科学上网）", base: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
      models: ["gemini-2.5-flash", "gemini-2.5-pro"], keyUrl: "https://aistudio.google.com/apikey",
      keyPrefix: "AIza", search: false, webOnly: true }
  ];

  function getProvider(id) {
    for (var i = 0; i < PROVIDERS.length; i++) if (PROVIDERS[i].id === id) return PROVIDERS[i];
    return null;
  }

  /**
   * 构造战术分析 Prompt
   * a/b: 球队对象, codeA/codeB, stat: Engine.predictMatch 输出, extra: 用户粘贴的最新情报
   */
  function buildPrompt(a, b, stat, extra, opts) {
    opts = opts || {};
    var pct = function (x) { return (x * 100).toFixed(1) + "%"; };
    var lines = [
      "你是一名顶级足球战术分析师。请基于以下数据，对 2026 美加墨世界杯的这场比赛给出专业战术分析。",
      "",
      "【对阵】" + a.zh + "（" + a.en + "） vs " + b.zh + "（" + b.en + "）" + (opts.knockout ? "（淘汰赛）" : "（小组赛）"),
      "",
      "【" + a.zh + "】Elo:" + a.elo + " FIFA排名:" + a.fifa + " 阵型:" + a.formation,
      "风格:" + a.style.join("/") + " | 教练:" + a.coach + "（" + a.coachStyle + "）",
      "核心球员:" + a.stars.join("、") + " | 近期状态:" + a.form + "/100 | 历史:" + a.wcBest,
      "",
      "【" + b.zh + "】Elo:" + b.elo + " FIFA排名:" + b.fifa + " 阵型:" + b.formation,
      "风格:" + b.style.join("/") + " | 教练:" + b.coach + "（" + b.coachStyle + "）",
      "核心球员:" + b.stars.join("、") + " | 近期状态:" + b.form + "/100 | 历史:" + b.wcBest,
      "",
      "【统计模型基线】（Elo + 双泊松 + Dixon-Coles）",
      a.zh + "胜:" + pct(stat.pWin) + " 平:" + pct(stat.pDraw) + " " + b.zh + "胜:" + pct(stat.pLoss),
      "期望进球 λ: " + stat.lambdaA.toFixed(2) + " vs " + stat.lambdaB.toFixed(2),
      "最可能比分: " + stat.topScores.slice(0, 3).map(function (s) { return s.a + "-" + s.b; }).join("、")
    ];
    if (extra && extra.trim()) {
      lines.push("", "【最新情报（用户提供）】", extra.trim());
    }
    lines.push(
      "",
      "请按以下结构输出（用中文，markdown 格式，每节 2-4 句，犀利专业有梗，适合社交媒体传播）：",
      "## 阵型相克：" + a.formation + " vs " + b.formation + " 的体系博弈",
      "## 教练斗法：" + a.coach + " vs " + b.coach,
      "## 三组关键对位（具体到球员）",
      "## X 因素（伤病/天气/心理/裁判尺度等变数）",
      "## 章鱼结论（一句话金句预测）",
      "",
      "最后，必须输出一个 JSON 代码块（```json 包裹），格式严格如下：",
      '```json',
      '{"adjustWin": 0, "adjustDraw": 0, "adjustLoss": 0, "reason": "调整理由（20字内）", "predictedScore": "2-1", "confidence": 75, "quote": "章鱼金句（25字内）"}',
      '```',
      "其中 adjustWin/adjustDraw/adjustLoss 是你基于战术分析对统计模型三项概率的修正（单位:百分点，每项限 -8 到 +8，三项之和必须为 0）；confidence 是你对预测的信心（50-95）。"
    );
    return lines.join("\n");
  }

  /** 从 LLM 输出文本中提取末尾 JSON 块并做有界校验 */
  function parseAdjustment(text) {
    var m = text.match(/```json\s*([\s\S]*?)```\s*$/) || text.match(/```json\s*([\s\S]*?)```/);
    if (!m) return null;
    try {
      var j = JSON.parse(m[1]);
      var clampPt = function (v) { return Math.max(-8, Math.min(8, Number(v) || 0)); };
      var aw = clampPt(j.adjustWin), ad = clampPt(j.adjustDraw), al = clampPt(j.adjustLoss);
      var sum = aw + ad + al; // 强制三项归零
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

  /** 应用 LLM 修正到统计概率，返回融合后概率（归一化） */
  function applyAdjustment(stat, adj) {
    if (!adj) return { pWin: stat.pWin, pDraw: stat.pDraw, pLoss: stat.pLoss, adjusted: false };
    var w = Math.max(0.01, stat.pWin + adj.adjustWin / 100);
    var d = Math.max(0.01, stat.pDraw + adj.adjustDraw / 100);
    var l = Math.max(0.01, stat.pLoss + adj.adjustLoss / 100);
    var s = w + d + l;
    return { pWin: w / s, pDraw: d / s, pLoss: l / s, adjusted: true };
  }

  /**
   * 流式调用（SSE）。opts:
   *  { providerId, key, model, prompt, temperature, enableSearch,
   *    onDelta(textChunk), onDone(fullText), onError(err) }
   */
  function chatStream(opts) {
    var p = getProvider(opts.providerId);
    if (!p) { opts.onError(new Error("未知渠道")); return; }
    var body = {
      model: opts.model || p.models[0],
      messages: [{ role: "user", content: opts.prompt }],
      temperature: opts.temperature !== undefined ? opts.temperature : 0.8,
      stream: true
    };
    // 智谱联网搜索工具
    if (opts.enableSearch && p.id === "glm") {
      body.tools = [{ type: "web_search", web_search: { enable: true } }];
    }
    var full = "";
    fetch(p.base, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": "Bearer " + opts.key },
      body: JSON.stringify(body)
    }).then(function (resp) {
      if (!resp.ok) {
        return resp.text().then(function (t) {
          throw new Error("HTTP " + resp.status + ": " + t.slice(0, 200));
        });
      }
      var reader = resp.body.getReader();
      var decoder = new TextDecoder("utf-8");
      var buf = "";
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) { opts.onDone(full); return; }
          buf += decoder.decode(r.value, { stream: true });
          var parts = buf.split("\n");
          buf = parts.pop();
          parts.forEach(function (line) {
            line = line.trim();
            if (!line.startsWith("data:")) return;
            var payload = line.slice(5).trim();
            if (payload === "[DONE]") return;
            try {
              var j = JSON.parse(payload);
              var delta = j.choices && j.choices[0] && j.choices[0].delta;
              if (delta && delta.content) { full += delta.content; opts.onDelta(delta.content); }
            } catch (e) { /* 忽略半截 JSON */ }
          });
          return pump();
        });
      }
      return pump();
    }).catch(function (err) { opts.onError(err); });
  }

  /** 非流式调用（小程序兼容 / 验证 Key 用） */
  function chatOnce(opts) {
    var p = getProvider(opts.providerId);
    if (!p) return Promise.reject(new Error("未知渠道"));
    return fetch(p.base, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": "Bearer " + opts.key },
      body: JSON.stringify({
        model: opts.model || p.models[0],
        messages: [{ role: "user", content: opts.prompt }],
        temperature: 0.3,
        stream: false,
        max_tokens: opts.maxTokens || 2048
      })
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok || j.error) throw new Error((j.error && j.error.message) || ("HTTP " + r.status));
        return j.choices[0].message.content;
      });
    });
  }

  /** 验证 Key 有效性，resolve(modelUsed) / reject(err) */
  function verifyKey(providerId, key) {
    return chatOnce({ providerId: providerId, key: key, prompt: "回复:OK", maxTokens: 8 })
      .then(function () { return getProvider(providerId).models[0]; });
  }

  var LLM = {
    PROVIDERS: PROVIDERS,
    getProvider: getProvider,
    buildPrompt: buildPrompt,
    parseAdjustment: parseAdjustment,
    applyAdjustment: applyAdjustment,
    chatStream: chatStream,
    chatOnce: chatOnce,
    verifyKey: verifyKey
  };

  if (typeof module !== "undefined" && module.exports) module.exports = LLM;
  else root.LLM = LLM;
})(typeof window !== "undefined" ? window : this);
