"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const form = $("#ask-form");
  const input = $("#ask-question");
  const results = $("#ask-results");
  const dialog = $("#detail-dialog");
  const coverageNames = {
    full_text: "整部法规",
    selected_sections: "已收录章节",
    excerpts: "已收录节选",
    uploaded_document: "上传文档",
  };
  let questionController;
  let dialogController;
  let healthLoading = false;

  // API responses and model output are always text, never executable markup.
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
    const control = el("button", className, text);
    control.type = "button";
    control.addEventListener("click", callback);
    return control;
  }

  function notice(text) {
    return el("p", "notice", text);
  }

  async function request(path, { body, signal } = {}) {
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    signal?.addEventListener("abort", forwardAbort, { once: true });
    const timeout = setTimeout(() => controller.abort(), body === undefined ? 20000 : 610000);
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
        throw new Error("暂时无法读取服务返回的内容，请稍后重试。");
      }
      if (!response.ok) {
        let message = data.error?.message;
        if (!message && Array.isArray(data.detail)) message = "请检查问题长度，输入 2 至 2000 个字符后重试。";
        const error = new Error(message || `请求失败（${response.status}），请稍后重试。`);
        error.code = data.error?.code;
        throw error;
      }
      return data;
    } catch (error) {
      if (signal?.aborted) throw new DOMException("已取消等待", "AbortError");
      if (controller.signal.aborted) throw new Error("等待服务响应超时，请稍后重新发送。");
      if (error instanceof TypeError) throw new Error("暂时无法连接服务，请检查连接后重试。");
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", forwardAbort);
    }
  }

  function renderLoading(target, text, cancel) {
    const box = el("div", "loading-state");
    const spinner = el("span", "spinner");
    spinner.setAttribute("aria-hidden", "true");
    box.append(spinner, el("span", "", text));
    if (cancel) box.append(button("取消等待", "text-button", cancel));
    target.replaceChildren(box);
    target.setAttribute("aria-busy", "true");
  }

  function renderError(target, error, retry) {
    const box = el("div", "error-panel");
    box.setAttribute("role", "alert");
    box.append(el("h2", "", "暂时没有完成"), el("p", "", error.message));
    if (retry) box.append(button("重新尝试", "button secondary", retry));
    target.replaceChildren(box);
  }

  function officialLink(urlValue, label = "查看官方出处") {
    if (!urlValue) return null;
    try {
      const url = new URL(urlValue);
      if (!["https:", "http:"].includes(url.protocol)) return null;
      const link = el("a", "text-link", label);
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.append(icon("external"));
      return link;
    } catch {
      return null;
    }
  }

  function sourceLocation(source) {
    if (source.source_type === "pdf") {
      return `PDF 第 ${source.pdf_page} 页${source.printed_page ? ` · 印刷页码 ${source.printed_page}` : ""}`;
    }
    return `${source.source_type === "web" ? "官方网页" : "收录文档"}${source.locator ? ` · ${source.locator}` : ""}`;
  }

  function appendSourceLinks(target, source) {
    const official = officialLink(source.official_url);
    if (official) target.append(official);
    for (const reference of source.supporting_sources || []) {
      const link = officialLink(reference.official_url, `补充出处：${reference.title}`);
      if (link) target.append(link);
    }
  }

  function documentButton(chunk) {
    // Coverage belongs to the document endpoint; never infer completeness from a chunk.
    const control = button("查看已收录原文", "text-button", () => openDocument(chunk));
    control.append(icon("book"));
    return control;
  }

  function sourceCard(chunk, index) {
    const card = el("article", "source-card");
    const heading = el("div", "source-heading");
    if (index !== undefined) heading.append(el("span", "source-number", String(index + 1)));
    const title = el("div");
    title.append(el("h3", "", chunk.article), el("p", "source-document", chunk.source.document_title));
    heading.append(title);
    const actions = el("div", "source-actions");
    actions.append(button("条文详情", "text-button", () => openArticle(chunk.id)), documentButton(chunk));
    appendSourceLinks(actions, chunk.source);
    card.append(heading, el("p", "original-text", chunk.text), el("p", "source-location", sourceLocation(chunk.source)), actions);
    return card;
  }

  function renderAnswer(data) {
    const sources = data.sources || [];
    const card = el("section", "answer-card");
    card.append(el("h2", "", data.status === "insufficient_evidence" ? "还需要更多依据" : "关于你的问题"));
    const claims = data.claims || [];
    if (!claims.length) card.append(el("p", "answer-claim", data.answer));
    for (const claim of claims) {
      const paragraph = el("p", "answer-claim", claim.text);
      for (const id of claim.citation_ids || []) {
        const index = sources.findIndex((source) => source.id === id);
        if (index < 0) continue;
        const citation = button(`[${index + 1}]`, "citation", () => openArticle(id));
        citation.setAttribute("aria-label", `查看引用 ${index + 1}：${sources[index].article}`);
        paragraph.append(citation);
      }
      card.append(paragraph);
    }
    if (data.notice) card.append(notice(data.notice));
    results.replaceChildren(card);
    if (data.follow_up_questions?.length) {
      const followUp = el("section", "follow-up");
      const list = el("ol");
      data.follow_up_questions.forEach((question) => list.append(el("li", "", question)));
      followUp.append(el("h2", "", "还需要了解这些情况"), list,
        el("p", "", "请把补充信息写回上方问题，保留原有背景后再次发送。"),
        button("补充情况", "text-button", () => {
          const marker = "\n\n补充：";
          if (!input.value.endsWith(marker) && input.value.length + marker.length <= input.maxLength) input.value += marker;
          updateInput();
          input.focus();
          input.setSelectionRange(input.value.length, input.value.length);
          input.scrollIntoView({ block: "center", behavior: "smooth" });
        }));
      results.append(followUp);
    }
    if (sources.length) {
      const section = el("section", "sources-section");
      const heading = el("div", "results-heading");
      heading.append(el("h2", "", "回答引用的原文"), el("span", "", `${sources.length} 条依据`));
      section.append(heading);
      sources.forEach((source, index) => section.append(sourceCard(source, index)));
      results.append(section);
    }
    const citedIds = new Set(sources.map((source) => source.id));
    const additional = (data.source_documents || []).filter((source) => !citedIds.has(source.id));
    if (additional.length) {
      const details = el("details", "additional-sources");
      details.append(el("summary", "", `另有 ${additional.length} 条相关资料`),
        el("p", "", "以下资料由检索找到，未作为本次回答的引用依据。"));
      additional.forEach((source) => details.append(sourceCard(source)));
      results.append(details);
    }
  }

  function updateInput() {
    input.setCustomValidity("");
    $("#ask-count").textContent = `${input.value.length} / 2000`;
  }

  function setBusy(busy) {
    $$("button, textarea", form).forEach((control) => { control.disabled = busy; });
    $("#send-question span").textContent = busy ? "正在回答" : "发送问题";
    results.setAttribute("aria-busy", String(busy));
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
    if (questionController) return;
    const question = input.value.trim();
    if (question.length < 2 || question.length > 2000) {
      input.setCustomValidity("请输入 2 至 2000 个字符的问题。");
      input.reportValidity();
      return;
    }
    const controller = new AbortController();
    questionController = controller;
    setBusy(true);
    renderLoading(results, "正在查找依据并在线回答…", () => controller.abort());
    try {
      const data = await request("/api/v1/ask", { body: { question }, signal: controller.signal });
      if (!controller.signal.aborted) renderAnswer(data);
    } catch (error) {
      if (error.name === "AbortError") {
        results.replaceChildren(el("p", "empty-state", "已取消等待，可以修改问题后重新发送。"));
      } else {
        renderError(results, error, () => form.requestSubmit());
      }
    } finally {
      questionController = undefined;
      setBusy(false);
    }
  });

  function metadata(entries) {
    const list = el("dl", "metadata");
    entries.forEach(([label, value]) => {
      if (value === null || value === undefined || value === "") return;
      const row = el("div");
      row.append(el("dt", "", label), el("dd", "", value));
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
    renderLoading(target, "正在读取原文…");
    const title = el("h2", "", "正在读取原文");
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
      if (!controller.signal.aborted) {
        target.setAttribute("aria-busy", "false");
        const heading = $("#dialog-title", target);
        heading.tabIndex = -1;
        heading.focus({ preventScroll: true });
      }
    }
  }

  function openArticle(id) {
    return showDetail("引用原文与出处", (signal) => request(`/api/v1/articles/${encodeURIComponent(id)}`, { signal }), (target, chunk) => {
      const title = el("h2", "", chunk.article);
      title.id = "dialog-title";
      const source = chunk.source;
      target.append(title, el("p", "document-meta", source.document_title), el("p", "original-text", chunk.text));
      target.append(metadata([
        ["原文位置", sourceLocation(source)], ["资料版本", chunk.version_id],
        ["发布机关", source.publisher], ["发布日期", source.published_on],
        ["施行日期", source.effective_on], ["失效日期", source.expires_on],
        ["效力层级", source.legal_level], ["来源核对日期", source.transcription_checked_on],
      ]));
      const actions = el("div", "source-actions");
      actions.append(documentButton(chunk));
      appendSourceLinks(actions, source);
      target.append(actions);
      if (source.verification_note) target.append(notice(source.verification_note));
    });
  }

  function openDocument(chunk) {
    const path = `/api/v1/documents/${encodeURIComponent(chunk.document_id)}?version_id=${encodeURIComponent(chunk.version_id)}`;
    return showDetail("已收录原文", (signal) => request(path, { signal }), (target, data) => {
      const title = el("h2", "", data.title);
      title.id = "dialog-title";
      const articles = data.articles || [];
      const coverage = coverageNames[data.coverage] || "已收录原文";
      $("#dialog-kicker").textContent = coverage;
      target.append(title, el("p", "document-meta", `${coverage} · ${articles.length} 条原文 · 版本 ${data.version_id}`));
      if (data.coverage_note) target.append(notice(data.coverage_note));
      if (data.coverage !== "full_text") target.append(notice("这里展示目前已收录的内容，不代表整部法规全文。"));
      const articleList = el("div", "document-articles");
      for (const article of articles) {
        const section = el("section", "document-article");
        section.append(el("h3", "", article.article), el("p", "original-text", article.text),
          el("p", "source-location", sourceLocation(article.source)));
        const actions = el("div", "source-actions");
        appendSourceLinks(actions, article.source);
        if (actions.childElementCount) section.append(actions);
        articleList.append(section);
      }
      if (!articles.length) articleList.append(el("p", "empty-state", "这份资料暂时没有可显示的原文。"));
      target.append(articleList);
    });
  }

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
    try {
      const data = await request("/health");
      const configured = data.llm_configured === true && data.embedding_configured === true;
      $("#service-label").textContent = configured ? "在线配置已就绪" : "问答待配置";
      $("#service-dot").className = `status-dot${configured ? "" : " unavailable"}`;
      control.title = configured ? "服务可访问，两项在线配置已就绪；模型连接将在提问时检查。点击刷新状态。" : "服务可访问，在线问答所需配置尚未完成；模型连接将在提问时检查。点击刷新状态。";
    } catch {
      $("#service-label").textContent = "服务未连接";
      $("#service-dot").className = "status-dot unavailable";
      control.title = "暂时无法连接服务，点击重试。";
    } finally {
      healthLoading = false;
      control.disabled = false;
    }
  }

  $("#refresh-health").addEventListener("click", refreshHealth);
  refreshHealth();
})();
