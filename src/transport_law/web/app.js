"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const form = $("#ask-form");
  const input = $("#ask-question");
  const results = $("#ask-results");
  const welcome = $("#chat-welcome");
  const dialog = $("#detail-dialog");
  let questionController;
  let dialogController;
  let chatState = { summary: "", messages: [] };
  let chatGeneration = 0;
  let pendingTurn;
  let healthLoading = false;
  let activeView = "ask";

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

  async function request(path, { body, signal, method = body === undefined ? "GET" : "POST" } = {}) {
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    signal?.addEventListener("abort", forwardAbort, { once: true });
    const timeout = setTimeout(() => controller.abort(), path === "/api/v1/ask" ? 610000 : method === "GET" ? 20000 : 120000);
    const multipart = body instanceof FormData;
    try {
      const response = await fetch(path, {
        method,
        headers: body === undefined || multipart ? { Accept: "application/json" } : {
          Accept: "application/json", "Content-Type": "application/json",
        },
        body: body === undefined ? undefined : multipart ? body : JSON.stringify(body),
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
        if (!message && Array.isArray(data.detail)) {
          message = path === "/api/v1/ask" ? "请检查问题长度，输入 1 至 2000 个字符后重试。" : "请检查上传文件和请求格式后重试。";
        }
        const error = new Error(message || `请求失败（${response.status}），请稍后重试。`);
        error.code = data.error?.code;
        error.status = response.status;
        throw error;
      }
      return data;
    } catch (error) {
      if (signal?.aborted) throw new DOMException("已取消等待", "AbortError");
      if (controller.signal.aborted) throw new Error("等待服务响应超时，请稍后重试。");
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
    const pages = source.pdf_page_end > source.pdf_page
      ? `${source.pdf_page}–${source.pdf_page_end}` : source.pdf_page;
    return `PDF 第 ${pages} 页${source.printed_page ? ` · 印刷页码 ${source.printed_page}` : ""}`;
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

  function renderAnswer(target, data) {
    const sources = data.sources || [];
    const card = el("section", "answer-card");
    const label = el("h2", "assistant-label", "法规助手");
    label.prepend(icon("book"));
    card.append(label);
    const claims = data.claims || [];
    if (!claims.length) card.append(el("p", "answer-claim", data.answer));
    for (const claim of claims) {
      const paragraph = el("p", "answer-claim", claim.text);
      for (const id of new Set(claim.citation_ids || [])) {
        const index = sources.findIndex((source) => source.id === id);
        if (index < 0) continue;
        const citation = button(`[${index + 1}]`, "citation", () => openArticle(id));
        citation.setAttribute("aria-label", `查看引用 ${index + 1}：${sources[index].article}`);
        citation.setAttribute("aria-haspopup", "dialog");
        citation.title = `${sources[index].source.document_title} · ${sourceLocation(sources[index].source)}`;
        paragraph.append(citation);
      }
      card.append(paragraph);
    }
    if (data.follow_up_questions?.length) {
      const followUp = el("div", "answer-follow-up");
      const list = el("ol");
      data.follow_up_questions.forEach((question) => list.append(el("li", "", question)));
      followUp.append(el("p", "", "为了进一步判断，还想请你补充："), list);
      card.append(followUp);
    }
    if (data.notice) card.append(el("p", "answer-notice", data.notice));
    if (data.context_compacted) card.append(el("p", "context-note", "已整理较早的对话，可继续补充情况。"));
    target.replaceChildren(card);
  }

  function updateInput() {
    input.setCustomValidity("");
    $("#ask-count").textContent = `${input.value.length} / 2000`;
    input.style.height = "auto";
    input.style.height = `${Math.max(56, Math.min(input.scrollHeight, 144))}px`;
  }

  function setBusy(busy) {
    $$("button, textarea", form).forEach((control) => { control.disabled = busy; });
    $("#send-question span").textContent = busy ? "正在回答" : "发送";
    results.setAttribute("aria-busy", String(busy));
  }

  input.addEventListener("input", updateInput);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  $$("[data-example]", welcome).forEach((control) => {
    control.addEventListener("click", () => {
      input.value = control.dataset.example;
      updateInput();
      input.focus();
    });
  });
  function clearChat(focus = false) {
    // Invalidate old requests before aborting: their catch/finally may run later.
    chatGeneration += 1;
    questionController?.abort();
    questionController = undefined;
    dialogController?.abort();
    dialogController = undefined;
    if (dialog.open) dialog.close();
    $("#dialog-content").replaceChildren();
    $("#dialog-content").setAttribute("aria-busy", "false");
    chatState = { summary: "", messages: [] };
    pendingTurn = undefined;
    results.replaceChildren(welcome);
    results.scrollTop = 0;
    input.value = "";
    updateInput();
    setBusy(false);
    if (focus) input.focus();
  }

  function createTurn(question) {
    pendingTurn?.container.remove();
    welcome.remove();
    const container = el("article", "chat-turn");
    const user = el("section", "user-message");
    user.append(el("h2", "message-label", "你"), el("p", "", question));
    const response = el("div", "assistant-message");
    container.append(user, response);
    results.append(container);
    return { question, container, response };
  }

  async function submitQuestion(question) {
    if (questionController) return;
    if (question.length < 1 || question.length > 2000) {
      input.setCustomValidity("请输入 1 至 2000 个字符的问题。");
      input.reportValidity();
      return;
    }
    const generation = chatGeneration;
    const turn = pendingTurn?.question === question ? pendingTurn : createTurn(question);
    pendingTurn = turn;
    const controller = new AbortController();
    questionController = controller;
    const isCurrent = () => generation === chatGeneration && questionController === controller;
    const retry = () => {
      if (generation !== chatGeneration || questionController) return;
      input.value = question;
      updateInput();
      submitQuestion(question);
    };
    setBusy(true);
    renderLoading(turn.response, "正在查找依据并在线回答…", () => controller.abort());
    turn.container.scrollIntoView({ block: "start", behavior: "smooth" });
    try {
      const data = await request("/api/v1/ask", { body: { question, chat_state: chatState }, signal: controller.signal });
      if (!isCurrent()) return;
      if (controller.signal.aborted) throw new DOMException("已取消等待", "AbortError");
      // Commit context only after a complete response; failed/cancelled attempts
      // keep the last successful state and can reuse this turn's cards.
      if (!data.chat_state || typeof data.chat_state.summary !== "string" || !Array.isArray(data.chat_state.messages)) {
        throw new Error("服务未返回完整的聊天内容，请重试。");
      }
      renderAnswer(turn.response, data);
      chatState = data.chat_state;
      pendingTurn = undefined;
      input.value = "";
      updateInput();
    } catch (error) {
      if (!isCurrent()) return;
      if (error.name === "AbortError") {
        const cancelled = el("div", "cancelled-message");
        cancelled.append(el("p", "", "已取消等待，可以重试或修改问题。"), button("重新尝试", "button secondary", retry));
        turn.response.replaceChildren(cancelled);
      } else {
        renderError(turn.response, error, retry);
      }
    } finally {
      if (isCurrent()) {
        questionController = undefined;
        turn.response.setAttribute("aria-busy", "false");
        setBusy(false);
      }
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion(input.value.trim());
  });
  $("#clear-chat").addEventListener("click", () => clearChat(true));
  // Clear on both sides of navigation, including restoration from back/forward cache.
  window.addEventListener("pagehide", () => clearChat());
  window.addEventListener("pageshow", () => clearChat());

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
    const version = chunk.version_id ? `?version_id=${encodeURIComponent(chunk.version_id)}` : "";
    const path = `/api/v1/documents/${encodeURIComponent(chunk.document_id)}${version}`;
    return showDetail("已收录原文", (signal) => request(path, { signal }), (target, data) => {
      const title = el("h2", "", data.title);
      title.id = "dialog-title";
      const articles = data.articles || [];
      const coverage = "上传文档";
      $("#dialog-kicker").textContent = coverage;
      target.append(title, el("p", "document-meta", `${coverage} · ${articles.length} 条原文 · 版本 ${data.version_id}`));
      if (data.coverage_note) target.append(notice(data.coverage_note));
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
  dialog.addEventListener("close", () => { if (!dialog.open) dialogController?.abort(); });
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });

  const knowledgePath = "/api/v1/knowledge/documents";
  const uploadForm = $("#upload-form");
  const fileInput = $("#upload-file");
  const documentList = $("#document-list");
  const retrying = new Set();
  const statusNames = { queued: "等待处理", processing: "正在入库", ready: "已入库", failed: "入库失败" };
  let selectedFile;
  let uploadMaxBytes;
  let uploadBusy = false;
  let documents = [];
  let libraryLoaded = false;
  let libraryController;
  let pollTimer;
  let renderedDocuments;

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    const unit = bytes < 1024 * 1024 ? 1024 : 1024 * 1024;
    return `${Number((bytes / unit).toFixed(2))} ${unit === 1024 ? "KiB" : "MiB"}`;
  }

  function fileError(file) {
    if (!file) return "请先选择一份资料。";
    if (!/\.pdf$/i.test(file.name.normalize("NFKC").trim())) return "请选择 PDF 文件。";
    if (!file.size) return "不能上传空文件，请选择有内容的资料。";
    if (uploadMaxBytes && file.size > uploadMaxBytes) return `文件超过当前 ${formatSize(uploadMaxBytes)} 的上传上限，请选择更小的文件。`;
    return "";
  }

  function uploadFeedback(text, error = false) {
    const target = $("#upload-feedback");
    target.hidden = !text;
    target.replaceChildren(notice(text));
    $("p", target).classList.toggle("error", error);
  }

  function syncUploadControls() {
    const selected = $("#selected-file");
    selected.hidden = !selectedFile;
    $("#selected-filename").textContent = selectedFile?.name || "";
    $("#selected-size").textContent = selectedFile ? formatSize(selectedFile.size) : "";
    fileInput.disabled = uploadBusy;
    $("#clear-file").disabled = uploadBusy;
    $("#submit-upload").disabled = uploadBusy || Boolean(fileError(selectedFile));
    $("#submit-upload span").textContent = uploadBusy ? "正在上传…" : "上传并入库";
    uploadForm.setAttribute("aria-busy", String(uploadBusy));
    $("#drop-zone").classList.toggle("is-busy", uploadBusy);
  }

  function chooseFile(file) {
    if (uploadBusy) return;
    selectedFile = file;
    if (!file) fileInput.value = "";
    const error = file ? fileError(file) : "";
    uploadFeedback(error, Boolean(error));
    syncUploadControls();
  }

  function isPending(record) {
    return record.status === "queued" || record.status === "processing";
  }

  function cancelLibraryRead() {
    clearTimeout(pollTimer);
    libraryController?.abort();
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    if (activeView === "knowledge" && !document.hidden && documents.some(isPending)) {
      pollTimer = setTimeout(() => loadDocuments(true), 3000);
    }
  }

  function upsertDocument(record) {
    const index = documents.findIndex((item) => item.id === record.id);
    if (index < 0) documents.unshift(record);
    else documents[index] = record;
  }

  function rowAction(label, record, action, callback) {
    const control = button(label, "text-button", callback);
    control.dataset.documentId = record.id;
    control.dataset.libraryAction = action;
    return control;
  }

  function renderDocuments() {
    const ready = documents.filter((record) => record.status === "ready").length;
    const pending = documents.filter(isPending).length;
    const failed = documents.filter((record) => record.status === "failed").length;
    $("#library-summary").textContent = `共 ${documents.length} 份 · 已入库 ${ready} 份 · 待完成 ${pending} 份 · 失败 ${failed} 份`;
    // Leave unchanged rows in place during polling, preserving keyboard focus.
    const signature = JSON.stringify([documents, [...retrying]]);
    if (signature === renderedDocuments) return;
    renderedDocuments = signature;
    const focused = documentList.contains(document.activeElement) ? { ...document.activeElement.dataset } : null;
    documentList.replaceChildren();
    if (!documents.length) {
      const empty = el("div", "library-empty");
      empty.append(icon("book"), el("h3", "", "还没有上传资料"), el("p", "", "选择一份资料开始，处理进度会显示在这里。"));
      documentList.append(empty);
    }
    for (const record of documents) {
      const card = el("article", "upload-item");
      const heading = el("div", "upload-item-heading");
      const title = el("div", "upload-item-title");
      title.append(icon("file"), el("h3", "", record.filename));
      heading.append(title, el("span", `document-status ${record.status}`, statusNames[record.status] || "状态未知"));
      card.append(heading);
      let progressText;
      if (record.status === "ready") progressText = `${record.chunk_count} 个片段已入库，可参与下一次提问。`;
      else if (record.chunk_count > 0) progressText = `已入库 ${record.embedded_count} / ${record.chunk_count} 个片段${record.status === "failed" ? "，重试会继续未完成的部分。" : "。"}`;
      else progressText = record.status === "queued" ? "已接收文件，等待后台处理。" : record.status === "processing" ? "正在提取与整理文字…" : "资料尚未完成入库。";
      card.append(el("p", "upload-progress-text", progressText));
      if (isPending(record)) {
        const progress = el("progress", "upload-progress");
        progress.setAttribute("aria-label", `${record.filename}的入库进度`);
        if (record.chunk_count > 0) {
          progress.max = record.chunk_count;
          progress.value = Math.min(record.embedded_count, record.chunk_count);
        }
        card.append(progress);
      }
      if (record.status === "failed") {
        const error = el("p", "document-error", record.error_message || "资料处理失败，请检查文件与服务配置后重试。");
        card.append(error);
      }
      const actions = el("div", "source-actions");
      if (record.status === "ready") {
        actions.append(rowAction("查看入库原文", record, "read", () => openDocument({ document_id: record.id })));
        const ask = el("a", "text-link", "去问答");
        ask.href = "#ask";
        ask.append(icon("arrow"));
        actions.append(ask);
      } else if (record.status === "failed" || record.status === "queued") {
        const control = rowAction(retrying.has(record.id) ? "正在提交…" : record.status === "failed" ? "重试入库" : "继续入库", record, "retry", () => retryDocument(record));
        control.disabled = retrying.has(record.id);
        if (record.status === "queued") control.title = "任务长时间等待时可尝试继续；正在执行的任务不会重复启动。";
        actions.append(control);
      }
      if (record.status === "ready" || record.status === "failed" || record.status === "queued") {
        const rebuild = rowAction("重新解析并入库", record, "reprocess", () => retryDocument(record, true));
        rebuild.disabled = retrying.has(record.id);
        rebuild.title = "使用当前解析方式重新整理原文件并生成向量；完成前该文档暂不参与问答，会使用向量服务额度。";
        actions.append(rebuild);
      }
      if (actions.childElementCount) card.append(actions);
      documentList.append(card);
    }
    if (focused?.libraryAction) {
      const replacement = $$("[data-library-action]", documentList).find((control) =>
        control.dataset.libraryAction === focused.libraryAction && control.dataset.documentId === focused.documentId);
      replacement?.focus({ preventScroll: true });
    }
  }

  function libraryError(error) {
    const target = $("#library-error");
    target.hidden = false;
    target.replaceChildren(el("p", "notice error", `${error.message} 自动更新已暂停，请点击“刷新列表”重试。`));
    $("#library-update").textContent = libraryLoaded ? "以下为上次读取的状态。" : "尚未读取到资料记录。";
    if (!libraryLoaded) {
      $("#library-summary").textContent = "资料数量暂不可用";
      documentList.replaceChildren(el("p", "empty-state", "暂时无法读取上传资料。"));
    }
  }

  async function loadDocuments(poll = false) {
    cancelLibraryRead();
    const controller = new AbortController();
    libraryController = controller;
    $("#refresh-documents").disabled = true;
    documentList.setAttribute("aria-busy", "true");
    $("#library-error").hidden = true;
    try {
      let pollError;
      if (poll) {
        const pending = documents.filter(isPending);
        const updates = await Promise.allSettled(pending.map((record) => request(`${knowledgePath}/${encodeURIComponent(record.id)}`, { signal: controller.signal })));
        if (controller.signal.aborted) return;
        for (const result of updates) {
          if (result.status === "fulfilled") upsertDocument(result.value);
          else pollError ||= result.reason;
        }
      } else {
        const records = await request(knowledgePath, { signal: controller.signal });
        if (controller.signal.aborted) return;
        if (!Array.isArray(records)) throw new Error("无法读取资料列表，请确认服务响应格式。");
        documents = records;
      }
      libraryLoaded = true;
      renderDocuments();
      const pending = documents.filter(isPending).length;
      $("#library-update").textContent = pending ? `${pending} 份资料正在等待或处理中，每 3 秒更新进度。` : "已更新；已入库的资料可直接用于问答。";
      if (pollError) libraryError(pollError);
      else schedulePoll();
    } catch (error) {
      if (!controller.signal.aborted) libraryError(error);
    } finally {
      if (libraryController === controller) {
        libraryController = undefined;
        $("#refresh-documents").disabled = false;
        documentList.setAttribute("aria-busy", "false");
      }
    }
  }

  fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
  $("#clear-file").addEventListener("click", () => { chooseFile(undefined); fileInput.focus(); });
  const dropZone = $("#drop-zone");
  let dragDepth = 0;
  dropZone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    if (!uploadBusy) { dragDepth += 1; dropZone.classList.add("drag-over"); }
  });
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = uploadBusy ? "none" : "copy";
  });
  dropZone.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    dropZone.classList.remove("drag-over");
    if (uploadBusy) return;
    if (event.dataTransfer.files.length !== 1) {
      chooseFile(undefined);
      uploadFeedback("请一次选择一份文件上传。", true);
      return;
    }
    fileInput.value = "";
    chooseFile(event.dataTransfer.files[0]);
  });

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (uploadBusy) return;
    const error = fileError(selectedFile);
    if (error) { uploadFeedback(error, true); return; }
    const payload = new FormData();
    payload.append("file", selectedFile);
    uploadBusy = true;
    syncUploadControls();
    uploadFeedback("正在上传文件，请稍候…");
    try {
      const record = await request(knowledgePath, { body: payload });
      cancelLibraryRead();
      upsertDocument(record);
      libraryLoaded = true;
      renderDocuments();
      selectedFile = undefined;
      fileInput.value = "";
      uploadFeedback(`已接收“${record.filename}”，后台正在安排入库。请以下方处理状态为准，只有“已入库”时才可用于问答。`);
    } catch (error) {
      const detail = error.code === "duplicate_filename"
        ? " 可在列表中检查已有任务；失败任务请使用“重试入库”。"
        : !error.status ? " 上传结果尚未确认，请先刷新列表检查是否已接收，再决定是否重新上传。" : "";
      uploadFeedback(error.message + detail, true);
    } finally {
      uploadBusy = false;
      syncUploadControls();
      if (activeView === "knowledge" && !document.hidden) loadDocuments();
    }
  });

  async function retryDocument(record, reprocess = false) {
    if (retrying.has(record.id)) return;
    cancelLibraryRead();
    retrying.add(record.id);
    renderDocuments();
    try {
      const action = reprocess ? "reprocess" : "retry";
      const updated = await request(`${knowledgePath}/${encodeURIComponent(record.id)}/${action}`, { method: "POST" });
      upsertDocument(updated);
      uploadFeedback(`“${record.filename}”已${reprocess ? "安排重新解析，完成前暂不参与问答" : "重新安排入库"}，处理结果将显示在列表中。`);
    } catch (error) {
      uploadFeedback(error.message + (!error.status ? " 重试结果尚未确认，请刷新列表查看状态。" : ""), true);
    } finally {
      retrying.delete(record.id);
      renderDocuments();
      if (activeView === "knowledge" && !document.hidden) loadDocuments();
    }
  }

  $("#refresh-documents").addEventListener("click", () => loadDocuments());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelLibraryRead();
    else if (activeView === "knowledge") loadDocuments();
  });
  window.addEventListener("pagehide", cancelLibraryRead);

  async function refreshHealth() {
    if (healthLoading) return;
    healthLoading = true;
    const control = $("#refresh-health");
    control.disabled = true;
    try {
      const data = await request("/health");
      if (Number.isSafeInteger(data.upload_max_bytes) && data.upload_max_bytes > 0) {
        uploadMaxBytes = data.upload_max_bytes;
        $("#upload-limit").textContent = `单文件最大 ${formatSize(uploadMaxBytes)}`;
        if (selectedFile && fileError(selectedFile)) uploadFeedback(fileError(selectedFile), true);
        syncUploadControls();
      } else {
        $("#upload-limit").textContent = "大小限制由服务检查";
      }
      const configured = data.llm_configured === true && data.embedding_configured === true;
      $("#service-label").textContent = configured ? "在线配置已就绪" : "问答待配置";
      $("#service-dot").className = `status-dot${configured ? "" : " unavailable"}`;
      control.title = configured ? "服务可访问，两项在线配置已就绪；模型连接将在提问时检查。点击刷新状态。" : "服务可访问，在线问答所需配置尚未完成；模型连接将在提问时检查。点击刷新状态。";
    } catch {
      $("#upload-limit").textContent = uploadMaxBytes ? `单文件最大 ${formatSize(uploadMaxBytes)}` : "暂未读取到大小限制";
      $("#service-label").textContent = "服务未连接";
      $("#service-dot").className = "status-dot unavailable";
      control.title = "暂时无法连接服务，点击重试。";
    } finally {
      healthLoading = false;
      control.disabled = false;
    }
  }

  function switchView(focus = false) {
    if (location.hash === "#main-content") return;
    activeView = location.hash === "#knowledge" ? "knowledge" : "ask";
    $("body").dataset.view = activeView;
    $("#view-ask").hidden = activeView !== "ask";
    $("#view-knowledge").hidden = activeView !== "knowledge";
    $$(".page-nav [data-view]").forEach((link) => {
      if (link.dataset.view === activeView) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    document.title = `${activeView === "ask" ? "在线问答" : "上传资料"} · 湖北交通法规助手`;
    if (activeView === "knowledge") loadDocuments();
    else { cancelLibraryRead(); updateInput(); }
    if (focus) {
      const heading = $(activeView === "ask" ? "#page-title" : "#knowledge-title");
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
      window.scrollTo({ top: 0 });
    }
  }

  $("#refresh-health").addEventListener("click", refreshHealth);
  window.addEventListener("hashchange", () => switchView(true));
  $(".skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    $("#main-content").focus();
  });
  switchView();
  refreshHealth();
})();
