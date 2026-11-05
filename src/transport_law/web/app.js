"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const ITEM_ID = "road_occupation_excavation";
  const viewNames = { retrieve: "法规检索", ask: "带依据问答", facts: "裁量辅助", status: "服务状态" };
  const kindNames = { recognition: "认定依据", penalty: "处罚依据", discretion: "裁量基准", provision: "法规条文", procedure: "程序依据" };
  const verificationNames = { pending: "适用待核验", verified: "来源已核验", superseded: "已被替代" };
  const fieldNames = {
    region: "案发省份", location: "具体地点", occurred_on: "发生日期", road_type: "道路类型",
    behavior: "具体行为", has_permission: "相关许可", area_m2: "占用 / 挖掘面积",
    facts_verified: "事实核验声明", question: "问题", top_k: "检索条目数",
  };
  const dialog = $("#detail-dialog");
  let dialogController;
  let healthLoading = false;
  let toastTimer;

  // All API and model text is rendered as text, never as executable HTML.
  function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(svg.namespaceURI, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function button(text, className, callback) {
    const result = el("button", className, text);
    result.type = "button";
    if (callback) result.addEventListener("click", callback);
    return result;
  }

  function notice(text, variant = "") {
    const box = el("div", `notice ${variant}`);
    box.append(icon("info"), el("p", "", text));
    return box;
  }

  function toast(text) {
    clearTimeout(toastTimer);
    const box = $("#toast");
    box.textContent = text;
    box.hidden = false;
    toastTimer = setTimeout(() => { box.hidden = true; }, 3600);
  }

  async function request(path, { body, signal } = {}) {
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    signal?.addEventListener("abort", forwardAbort, { once: true });
    const waitMs = path.endsWith("/ask") ? 610000 : path.endsWith("/retrieve") ? 310000 : 20000;
    const timeout = setTimeout(() => controller.abort(), waitMs);
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? { Accept: "application/json" } : {
          Accept: "application/json", "Content-Type": "application/json",
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: "same-origin",
        signal: controller.signal,
      });
      let data;
      try {
        data = await response.json();
      } catch (error) {
        if (controller.signal.aborted) throw error;
        throw new Error("服务返回了无法读取的内容，请确认 Python 服务已正常启动。");
      }
      if (!response.ok) {
        let message = data.error?.message;
        if (!message && Array.isArray(data.detail)) {
          const fields = [...new Set(data.detail.map((entry) => fieldNames[entry.loc?.[1]] || "输入内容"))];
          message = `请检查${fields.join("、")}的格式或范围。`;
        }
        const error = new Error(message || `请求失败（${response.status}），请稍后重试。`);
        error.code = data.error?.code;
        throw error;
      }
      return data;
    } catch (error) {
      if (signal?.aborted) throw new DOMException("已取消请求", "AbortError");
      if (controller.signal.aborted) throw new Error("等待服务响应超时，请稍后重试。");
      if (error instanceof TypeError) throw new Error("无法连接本地服务，请确认服务正在运行，然后重试。");
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", forwardAbort);
    }
  }

  function renderLoading(target, text, cancel) {
    const box = el("div", "loading-state");
    box.append(el("span", "spinner"), el("span", "", text));
    if (cancel) box.append(button("取消", "text-button", cancel));
    target.replaceChildren(box);
    target.setAttribute("aria-busy", "true");
  }

  function renderError(target, error, retry) {
    const box = el("div", "error-panel");
    box.setAttribute("role", "alert");
    box.append(el("h2", "", "暂时无法完成请求"), el("p", "", error.message));
    if (error.code === "llm_not_configured") {
      box.append(el("p", "", "请在项目的 .env 中配置 DEEPSEEK_API_KEY 并重启服务，也可以先使用法规检索查看原文。"));
      const link = el("a", "button secondary", "前往法规检索");
      link.href = "#retrieve";
      box.append(link);
    } else if (retry) {
      box.append(button("重新尝试", "button secondary", retry));
    }
    target.replaceChildren(box);
  }

  function emptyState(title, description, iconName = "search") {
    const box = el("div", "empty-state");
    const emblem = el("span", "empty-icon");
    emblem.append(icon(iconName));
    box.append(emblem, el("h2", "", title), el("p", "", description));
    return box;
  }

  function resultHeading(title, detail = "") {
    const heading = el("div", "results-heading");
    heading.append(el("h2", "", title), el("span", "", detail));
    return heading;
  }

  function sourceButton(chunkId, label = "查看完整条目") {
    const control = button(label, "text-button", () => openArticle(chunkId));
    control.append(icon("arrow"));
    return control;
  }

  function sourceCard(chunk, index) {
    const card = el("article", "card result-card");
    const meta = el("div", "result-meta");
    meta.append(
      el("span", "result-number", String(index + 1).padStart(2, "0")),
      el("span", `badge ${chunk.kind === "penalty" ? "blue" : "green"}`, kindNames[chunk.kind] || chunk.kind),
      el("span", "badge amber", verificationNames[chunk.source.verification_status] || "待核验"),
      el("span", "source-page", chunk.source.source_type === "web" ? `官网 · ${chunk.source.locator || "条文"}` : `PDF 第 ${chunk.source.pdf_page} 页`),
    );
    const footer = el("div", "result-footer");
    footer.append(el("span", "", `${chunk.region} · ${chunk.domain} · ${chunk.id}`), sourceButton(chunk.id));
    card.append(meta, el("h3", "", chunk.article), el("p", "source-text", chunk.text), footer);
    return card;
  }

  function renderRetrieval(target, data) {
    target.replaceChildren(resultHeading("检索结果", `${data.hits.length} 条相关依据`));
    target.append(el("p", "result-query", `本次查询：${data.question}`));
    if (!data.hits.length) {
      target.append(emptyState("当前资料中没有相关依据", "可尝试输入法规名称和条号，或补充具体行为。未命中不代表不存在相关规定。"));
    } else {
      data.hits.forEach((hit, index) => target.append(sourceCard(hit.chunk, index)));
    }
    if (data.coverage.length) {
      const coverage = el("details", "coverage");
      const list = el("ul");
      data.coverage.forEach((text) => list.append(el("li", "", text)));
      coverage.append(el("summary", "", "查看当前检索覆盖范围"), list);
      target.append(coverage);
    }
  }

  function followUp(questions, title = "还需要补充的信息") {
    const box = el("section", "follow-up");
    const list = el("ol");
    questions.forEach((question) => list.append(el("li", "", question)));
    box.append(el("h3", "", title), list);
    return box;
  }

  function renderAnswer(target, data) {
    target.replaceChildren();
    if (data.status === "insufficient_evidence") {
      target.append(emptyState("资料依据不足", data.answer, "book"));
    } else {
      const card = el("section", "card answer-card");
      card.append(el("span", "badge green", "已生成 · 含来源引用"), el("h2", "dialog-section-title", "基于材料的解释"));
      data.claims.forEach((claim) => {
        const paragraph = el("p", "answer-claim");
        paragraph.append(document.createTextNode(claim.text));
        claim.citation_ids.forEach((id) => {
          const index = data.sources.findIndex((source) => source.id === id);
          const citation = button(`[${index + 1}]`, "citation", () => openArticle(id));
          citation.setAttribute("aria-label", `查看引用 ${index + 1}：${data.sources[index]?.article || id}`);
          paragraph.append(citation);
        });
        card.append(paragraph);
      });
      const usage = data.usage ? ` · Token 用量 ${data.usage.total_tokens}` : "";
      card.append(el("p", "answer-meta", `模型：${data.model || "未返回模型名称"}${usage}`));
      card.append(notice(data.notice));
      target.append(card);
    }
    if (data.follow_up_questions.length) target.append(followUp(data.follow_up_questions));
    if (data.sources.length) {
      const heading = resultHeading("引用原文", `${data.sources.length} 条来源`);
      heading.classList.add("source-list-heading");
      target.append(heading);
      data.sources.forEach((source, index) => target.append(sourceCard(source, index)));
    }
  }

  function setFormBusy(form, busy) {
    form.dataset.loading = String(busy);
    $$("button, input, select, textarea", form).forEach((control) => { control.disabled = busy; });
  }

  $$("[data-search]").forEach((form) => {
    const mode = form.dataset.search;
    const input = $("textarea", form);
    const target = $(`#${mode}-results`);
    function updateInput() {
      input.setCustomValidity("");
      $(`#${mode}-count`).textContent = `${input.value.length} / 2000`;
    }
    input.addEventListener("input", updateInput);
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    $$("[data-example]", form).forEach((control) => {
      control.addEventListener("click", () => {
        input.value = control.dataset.example;
        updateInput();
        input.focus();
      });
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.dataset.loading === "true") return;
      const question = input.value.trim();
      if (question.length < 2) {
        input.setCustomValidity("请输入至少两个字符的问题。");
        input.reportValidity();
        return;
      }
      const body = { question, top_k: Number($("select", form).value) };
      const controller = new AbortController();
      setFormBusy(form, true);
      renderLoading(target, mode === "ask" ? "正在检索资料并生成解释，请稍候…" : "正在检索相关依据…", () => controller.abort());
      try {
        const data = await request(`/api/v1/${mode}`, { body, signal: controller.signal });
        if (mode === "retrieve") renderRetrieval(target, data);
        else renderAnswer(target, data);
      } catch (error) {
        if (error.name === "AbortError") target.replaceChildren(emptyState("已取消等待", "可以修改问题后再次提交。已发送的模型请求可能仍在服务端执行。"));
        else renderError(target, error, () => form.requestSubmit());
      } finally {
        setFormBusy(form, false);
        target.setAttribute("aria-busy", "false");
      }
    });
  });

  const factsForm = $("#facts-form");
  const factsTarget = $("#facts-results");
  function resetFactHighlights() {
    $$(".field", factsForm).forEach((field) => field.classList.remove("is-missing"));
  }

  function renderFacts(data) {
    const labels = {
      needs_facts: ["需要补充事实", "点击缺失字段可返回对应输入位置。"],
      needs_review: ["事实已填齐，仍需复核", "资料效力、适用条件和裁量边界尚未完成核验。"],
      out_of_scope: ["超出当前规则范围", "当前规则草稿不能用于检查该地域或道路类型。"],
    };
    const [title, description] = labels[data.status];
    const card = el("section", "card check-result");
    card.append(el("span", "badge amber", "个案适用未核验"), el("h2", "", title), el("p", "", description));
    if (data.missing_fields.length) {
      card.append(el("h3", "", `待补充字段（${data.missing_fields.length}）`));
      const fields = el("div", "missing-fields");
      data.missing_fields.forEach((name) => {
        const input = factsForm.elements.namedItem(name);
        input?.closest(".field")?.classList.add("is-missing");
        fields.append(button(fieldNames[name] || name, "missing-field", () => {
          input?.scrollIntoView({ block: "center" });
          input?.focus({ preventScroll: true });
        }));
      });
      card.append(fields);
    }
    if (data.follow_up_questions.length) card.append(followUp(data.follow_up_questions, "需要核实的问题"));
    if (data.review_notes.length) {
      const notes = el("ul", "review-list");
      data.review_notes.forEach((note) => notes.append(el("li", "", note)));
      card.append(el("h3", "", "复核提醒"), notes);
    }
    if (data.rule_set) {
      const rulesButton = button("查看关联规则草稿", "text-button", openRules);
      rulesButton.style.marginTop = "18px";
      rulesButton.append(icon("arrow"));
      card.append(rulesButton);
    }
    factsTarget.replaceChildren(card);
    factsTarget.dataset.hasResult = "true";
    if (data.sources.length) {
      const sources = el("details", "coverage");
      const links = el("div", "rule-sources");
      data.sources.forEach((source) => links.append(sourceButton(source.id, source.article)));
      sources.append(el("summary", "", `查看 ${data.sources.length} 条关联依据`), links);
      factsTarget.append(sources);
    }
  }

  factsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (factsForm.dataset.loading === "true") return;
    const body = { item_id: ITEM_ID, facts_verified: factsForm.elements.facts_verified.checked };
    for (const name of ["region", "location", "occurred_on", "road_type", "behavior", "area_m2"]) {
      const value = factsForm.elements.namedItem(name).value.trim();
      if (value) body[name] = value;
    }
    const permission = factsForm.elements.has_permission.value;
    if (permission !== "") body.has_permission = permission === "true";
    // Empty values remain absent; in particular "false" is an explicit fact, not missing data.
    resetFactHighlights();
    factsTarget.dataset.hasResult = "false";
    setFormBusy(factsForm, true);
    renderLoading(factsTarget, "正在检查事实与复核事项…");
    try {
      renderFacts(await request("/api/v1/discretion/check", { body }));
    } catch (error) {
      renderError(factsTarget, error, () => factsForm.requestSubmit());
    } finally {
      setFormBusy(factsForm, false);
      factsTarget.setAttribute("aria-busy", "false");
    }
  });

  factsForm.addEventListener("reset", () => {
    resetFactHighlights();
    factsTarget.dataset.hasResult = "false";
    factsTarget.replaceChildren(emptyState("让待核实信息更清晰", "提交后将列出缺失事实、补充问题、边界与适用复核事项。", "clipboard"));
  });
  factsForm.addEventListener("input", () => {
    if (factsTarget.dataset.hasResult === "true" && !$(".stale-notice", factsTarget)) {
      const note = notice("表单已修改，以下结果对应上次提交；请重新检查。");
      note.classList.add("stale-notice");
      factsTarget.prepend(note);
    }
  });
  $("#load-example").addEventListener("click", () => {
    factsForm.reset();
    factsForm.elements.region.value = "湖北省";
    factsForm.elements.area_m2.value = "20";
    factsForm.elements.region.focus();
    toast("已载入缺失事实示例，点击“检查事实”查看结果。");
  });

  function metadata(entries) {
    const list = el("dl", "metadata");
    entries.forEach(([label, value]) => {
      const row = el("div");
      row.append(el("dt", "", label), el("dd", "", value === null || value === undefined || value === "" ? "未记录" : value));
      list.append(row);
    });
    return list;
  }

  async function showDetail(kicker, loader, renderer) {
    dialogController?.abort();
    const controller = new AbortController();
    dialogController = controller;
    $("#dialog-kicker").textContent = kicker;
    const target = $("#dialog-content");
    renderLoading(target, "正在读取完整资料…");
    const title = el("h2", "", "正在读取资料");
    title.id = "dialog-title";
    target.prepend(title);
    dialog.scrollTop = 0;
    if (!dialog.open) dialog.showModal();
    try {
      const data = await loader(controller.signal);
      if (controller.signal.aborted) return;
      target.replaceChildren();
      renderer(target, data);
    } catch (error) {
      if (controller.signal.aborted) return;
      renderError(target, error, () => showDetail(kicker, loader, renderer));
      $("h2", target).id = "dialog-title";
    } finally {
      if (!controller.signal.aborted) target.setAttribute("aria-busy", "false");
    }
  }

  function openArticle(id) {
    return showDetail("原文与来源", (signal) => request(`/api/v1/articles/${encodeURIComponent(id)}`, { signal }), (target, chunk) => {
      const title = el("h2", "", chunk.article);
      title.id = "dialog-title";
      const source = chunk.source;
      target.append(el("span", "badge amber", verificationNames[source.verification_status] || "待核验"), title,
        el("div", "original-text", chunk.text), el("h3", "dialog-section-title", "出处与核验记录"));
      target.append(metadata([
        ["资料名称", source.document_title], ["来源文件", source.file],
        ["原文位置", source.source_type === "web" ? `官方网页 · ${source.locator || "条文"}` : `PDF 第 ${source.pdf_page} 页`], ["印刷页码", source.printed_page],
        ["效力层级", source.legal_level], ["检索整理日期", source.retrieved_on],
        ["条目类型", kindNames[chunk.kind]], ["地域 / 领域", `${chunk.region} / ${chunk.domain}`],
        ["条目编号", chunk.id], ["文档编号", chunk.document_id], ["资料版本", chunk.version_id],
        ["发布机关", source.publisher], ["发布日期", source.published_on],
        ["施行日期", source.effective_on], ["失效日期", source.expires_on],
        ["转录核对日期", source.transcription_checked_on], ["文件 SHA-256", source.sha256],
      ]));
      const references = [
        { official_url: source.official_url, title: "查看官方来源" },
        ...(source.supporting_sources || []).map((ref) => ({ ...ref, title: `合并依据：${ref.title} · ${ref.locator}` })),
      ];
      for (const ref of references) {
        if (!ref.official_url) continue;
        try {
          const url = new URL(ref.official_url);
          if (["https:", "http:"].includes(url.protocol)) {
            const link = el("a", "text-link", ref.title);
            link.href = url.href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.append(icon("external"));
            target.append(link);
          }
        } catch { /* An invalid source URL must not prevent reading the original text. */ }
      }
      target.append(notice(source.verification_note));
    });
  }

  function openRules() {
    return showDetail("裁量规则 · 转录草稿", (signal) => request(`/api/v1/discretion/${ITEM_ID}`, { signal }), (target, rules) => {
      const title = el("h2", "", rules.name);
      title.id = "dialog-title";
      target.append(el("span", "badge amber", "草稿 · 未核验"), title,
        notice("以下为材料中的档次原文，尚未确定个案适用和边界含义；不自动选择档次或计算罚款。"));
      rules.bands.forEach((band, index) => {
        const card = el("section", "rule-band");
        card.append(el("span", "", `原表第 ${index + 1} 档`), el("h3", "", band.condition_text), el("p", "", band.consequence_text));
        target.append(card);
      });
      target.append(el("h3", "dialog-section-title", "需要复核的事项"));
      const notes = el("ul", "review-list");
      rules.review_notes.forEach((note) => notes.append(el("li", "", note)));
      target.append(notes, metadata([["规则版本", rules.version_id], ["待核验边界", `${rules.unresolved_boundaries_m2.join("、")} 平方米`]]));
      const sources = el("div", "rule-sources");
      rules.source_ids.forEach((id) => sources.append(sourceButton(id, id)));
      target.append(el("h3", "dialog-section-title", "规则关联条目"), sources);
    });
  }

  $("#open-rules").addEventListener("click", openRules);
  $("#close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => dialogController?.abort());
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });

  async function refreshHealth() {
    if (healthLoading) return;
    healthLoading = true;
    const control = $("#refresh-health");
    control.disabled = true;
    $("#health-error").hidden = true;
    try {
      const data = await request("/health");
      $$("[data-health]").forEach((element) => { element.textContent = data[element.dataset.health] ?? "—"; });
      $("#sidebar-status").className = "status-dot";
      $("#sidebar-status").setAttribute("aria-label", "服务在线");
      $("#app-version").textContent = `v${data.version}`;
      $("#backend-status").textContent = data.status === "ok" ? "运行正常" : "状态异常";
      $("#health-checked-at").textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
      $("#health-model").textContent = data.llm_model;
      $("#retrieval-mode").textContent = data.retrieval_backend === "chroma"
        ? (data.embedding_configured ? "在线向量检索" : "在线向量检索 · 未配置密钥")
        : "本地 BM25 · 离线可用";
      $("#status-model-badge").textContent = data.llm_configured ? "密钥已配置" : "未配置密钥";
      $("#status-model-badge").className = `badge ${data.llm_configured ? "green" : "amber"}`;
      $("#model-config-label").textContent = data.llm_configured ? "密钥已配置" : "尚未配置密钥";
      $("#model-config-detail").textContent = data.llm_configured
        ? `当前模型为 ${data.llm_model}。配置已读取，实际连通性尚未检测。`
        : "在本地 .env 中填写 DeepSeek API Key 并重启服务后，即可使用命中资料后的模型解释。";
    } catch (error) {
      $("#sidebar-status").className = "status-dot offline";
      $("#sidebar-status").setAttribute("aria-label", "服务连接失败");
      $("#backend-status").textContent = "连接失败";
      $("#health-checked-at").textContent = "无法获取最新状态";
      $$("[data-health]").forEach((element) => { element.textContent = "—"; });
      $("#app-version").textContent = "—";
      $("#health-model").textContent = "—";
      $("#retrieval-mode").textContent = "检索配置不可用";
      $("#status-model-badge").textContent = "状态未知";
      $("#status-model-badge").className = "badge amber";
      $("#model-config-label").textContent = "无法读取配置";
      $("#model-config-detail").textContent = "请确认本地服务已启动，稍后在服务状态中刷新。";
      const errorBox = $("#health-error");
      errorBox.replaceChildren(notice(error.message, "error"));
      errorBox.hidden = false;
    } finally {
      healthLoading = false;
      control.disabled = false;
    }
  }

  function switchView(focus = false) {
    const requested = location.hash.slice(1);
    const name = Object.hasOwn(viewNames, requested) ? requested : "retrieve";
    $$(".view").forEach((view) => { view.hidden = view.id !== `view-${name}`; });
    $$("[data-view]").forEach((link) => {
      const selected = link.dataset.view === name;
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    $("#current-page").textContent = viewNames[name];
    document.title = `${viewNames[name]} · 湖北交通法规助手`;
    if (focus) {
      const heading = $(`#${name}-title`);
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
      window.scrollTo({ top: 0 });
    }
    if (name === "status") refreshHealth();
  }

  $("#refresh-health").addEventListener("click", refreshHealth);
  $(".skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    $("#main-content").focus();
  });
  window.addEventListener("hashchange", () => switchView(true));
  switchView();
  refreshHealth();
})();
