// Run with: node --test tests/test_web_chat.mjs
// Exercise the complete browser script with a small DOM and controllable network.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../src/transport_law/web/index.html", import.meta.url), "utf8");
const script = readFileSync(new URL("../src/transport_law/web/app.js", import.meta.url), "utf8");

class Element {
  constructor(tag, ownerDocument) {
    Object.assign(this, { tagName: tag.toLowerCase(), ownerDocument, children: [], attributes: {}, listeners: {}, dataset: {}, style: {}, scrollHeight: 56, value: "", disabled: false, open: false });
    this.classList = {
      add: (name) => { this.className = [...new Set([...this.className.split(" "), name])].join(" "); },
      remove: (name) => { this.className = this.className.split(" ").filter((item) => item !== name).join(" "); },
      toggle: (name, enabled) => enabled ? this.classList.add(name) : this.classList.remove(name),
    };
  }
  get className() { return this.attributes.class || ""; }
  set className(value) { this.attributes.class = value; }
  get id() { return this.attributes.id; }
  set id(value) { this.attributes.id = value; }
  get textContent() { return (this.text || "") + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this.replaceChildren(); this.text = String(value); }
  get childElementCount() { return this.children.length; }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = value;
  }
  removeAttribute(name) { delete this.attributes[name]; }
  append(...nodes) { for (const node of nodes) { node.remove(); node.parent = this; this.children.push(node); } }
  prepend(node) { node.remove(); node.parent = this; this.children.unshift(node); }
  replaceChildren(...nodes) {
    for (const child of this.children) child.parent = undefined;
    this.children = [];
    this.text = "";
    this.append(...nodes);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((node) => node !== this);
    this.parent = undefined;
  }
  matches(selector) {
    if (selector.startsWith("#")) return this.id === selector.slice(1);
    if (selector.startsWith(".")) return this.className.split(" ").includes(selector.slice(1));
    if (selector.startsWith("[")) return selector.slice(1, -1) in this.attributes;
    return this.tagName === selector;
  }
  querySelectorAll(selector) {
    const choices = selector.split(",").map((part) => part.trim().split(/\s+/));
    const match = (node, parts) => {
      if (!node.matches(parts.at(-1))) return false;
      if (parts.length === 1) return true;
      for (let ancestor = node.parent; ancestor; ancestor = ancestor.parent) {
        if (match(ancestor, parts.slice(0, -1))) return true;
      }
      return false;
    };
    const descendants = this.children.flatMap((node) => [node, ...node.querySelectorAll("*")]);
    return selector === "*" ? descendants : descendants.filter((node) => choices.some((parts) => match(node, parts)));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  dispatch(type, extras = {}) {
    const event = { target: this, preventDefault() {}, ...extras };
    for (const callback of this.listeners[type] || []) callback(event);
  }
  click() { if (!this.disabled) this.dispatch("click"); }
  requestSubmit() { this.dispatch("submit"); }
  setCustomValidity(value) { this.validationMessage = value; }
  reportValidity() {}
  focus() { this.ownerDocument.activeElement = this; }
  scrollIntoView() { this.ownerDocument.viewportScrollTarget = this; }
  showModal() { this.open = true; }
  close() { this.open = false; this.dispatch("close"); }
  contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
}

function createDocument() {
  const document = new Element("document");
  document.ownerDocument = document;
  document.createElement = (tag) => new Element(tag, document);
  document.createElementNS = (_, tag) => document.createElement(tag);
  const stack = [document];
  const voidTags = new Set(["meta", "link", "img", "br", "input", "path", "use"]);
  for (const token of html.matchAll(/<\/?[a-z][^>]*>|[^<]+/gi)) {
    const value = token[0];
    if (value.startsWith("</")) {
      const tag = value.match(/^<\/([\w-]+)/)[1].toLowerCase();
      if (stack.at(-1).tagName === tag) stack.pop();
    } else if (value.startsWith("<")) {
      const tag = value.match(/^<([\w-]+)/)[1].toLowerCase();
      const node = document.createElement(tag);
      for (const match of value.matchAll(/([\w-]+)="([^"]*)"/g)) node.setAttribute(match[1], match[2]);
      if (node.attributes.maxlength) node.maxLength = Number(node.attributes.maxlength);
      stack.at(-1).append(node);
      if (!voidTags.has(tag) && !value.endsWith("/>")) stack.push(node);
    } else {
      stack.at(-1).text = (stack.at(-1).text || "") + value;
    }
  }
  return document;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

const flush = async () => { for (let index = 0; index < 12; index += 1) await Promise.resolve(); };

async function setup() {
  const document = createDocument();
  const window = new Element("window", document);
  window.scrollTo = () => {};
  const location = { hash: "#ask" };
  const calls = [];
  const fetch = (path, options) => {
    const call = { path, options, body: options.body ? JSON.parse(options.body) : undefined };
    calls.push(call);
    if (path === "/health") return Promise.resolve({ ok: true, json: async () => ({ upload_max_bytes: 1000000 }) });
    if (path === "/api/v1/knowledge/documents") return Promise.resolve({ ok: true, json: async () => [] });
    // Deliberately allow completion after abort to test stale-response guards.
    call.pending = deferred();
    return call.pending.promise;
  };
  const context = {
    document, window, location, fetch, AbortController, DOMException, FormData, URL,
    setTimeout: () => 1, clearTimeout: () => {},
  };
  for (const key of ["localStorage", "sessionStorage", "indexedDB"]) {
    Object.defineProperty(context, key, { get() { throw new Error(`Unexpected persistence: ${key}`); } });
  }
  vm.runInNewContext(script, context, { filename: "app.js" });
  window.dispatch("pageshow");
  await flush();
  const $ = (selector) => document.querySelector(selector);
  return {
    $, document, window, location, calls,
    asks: () => calls.filter((call) => call.path === "/api/v1/ask"),
    send(question) { $("#ask-question").value = question; $("#ask-form").requestSubmit(); },
    async respond(call, data) { call.pending.resolve({ ok: true, json: async () => data }); await flush(); },
    async fail(call) { call.pending.reject(new Error("暂时失败")); await flush(); },
  };
}

function answer(question, { messages = [], summary = "", compacted = false, suffix = "一" } = {}) {
  const id = `article-${suffix}`;
  return {
    status: "answered", answer: `回答${suffix}`,
    claims: [{ text: `回答${suffix}`, citation_ids: [id] }],
    sources: [{ id, article: `第${suffix}条`, text: `原文${suffix}`, document_id: "doc", version_id: "v1", source: { document_title: "资料", pdf_page: 1, official_url: "https://example.gov.cn/law" } }],
    source_documents: [], follow_up_questions: ["还有什么情况？"],
    chat_state: { summary, messages: [...messages, { role: "user", content: question }, { role: "assistant", content: `回答${suffix}` }] },
    context_compacted: compacted, retrieval_query: question,
  };
}

test("successive questions carry confirmed history, show free-form prompts by the composer and preserve citations", async () => {
  const app = await setup();
  app.send("未经许可施工");
  assert.deepEqual(app.asks()[0].body, { question: "未经许可施工", chat_state: { summary: "", messages: [] } });
  const first = answer("未经许可施工");
  await app.respond(app.asks()[0], first);
  assert.equal(app.$("#ask-question").value, "");
  assert.ok(app.$("#follow-up-panel").textContent.includes("还有什么情况？"));
  assert.equal(app.$("#ask-results").textContent.includes("还有什么情况？"), false);
  assert.equal(app.$(".assistant-message").childElementCount, 1);
  app.send("已经恢复原状");
  assert.deepEqual(app.asks()[1].body.chat_state, first.chat_state);
  const second = answer("已经恢复原状", { summary: "早期情况", compacted: true, suffix: "二" });
  second.sources[0].source.pdf_page_end = 2;
  await app.respond(app.asks()[1], second);
  assert.equal(app.document.querySelectorAll(".chat-turn").length, 2);
  assert.equal(app.document.querySelectorAll(".user-message").length, 2);
  assert.equal(app.document.querySelectorAll(".answer-card").length, 2);
  assert.equal(app.document.querySelectorAll(".citation").length, 2);
  assert.equal(app.$("#ask-results").textContent.includes("原文一"), false);
  assert.equal(app.$("#ask-results").textContent.includes("原文二"), false);
  assert.equal(app.$("#ask-results").querySelectorAll(".original-text").length, 0);
  assert.ok(app.$(".context-note"));
  app.$(".citation").click();
  assert.equal(app.calls.at(-1).path, "/api/v1/articles/article-%E4%B8%80");
  assert.equal(app.$("#detail-dialog").open, true);
  await app.respond(app.calls.at(-1), first.sources[0]);
  assert.equal(app.$("#dialog-content .original-text").textContent, "原文一");
  app.$("#close-dialog").click();
  app.document.querySelectorAll(".citation")[1].click();
  assert.equal(app.calls.at(-1).path, "/api/v1/articles/article-%E4%BA%8C");
  await app.respond(app.calls.at(-1), second.sources[0]);
  assert.equal(app.$("#dialog-content .original-text").textContent, "原文二");
  assert.ok(app.$("#dialog-content").textContent.includes("PDF 第 1–2 页"));
  app.$("#close-dialog").click();
  app.send("还有哪些依据");
  assert.deepEqual(app.asks()[2].body.chat_state, second.chat_state);
});

test("uncited retrieved text stays out of the chat and citations remain tied to known sources", async () => {
  const app = await setup();
  app.send("问题");
  const data = answer("问题");
  data.claims[0].text = "<script>这是资料中的文字</script>";
  data.claims[0].citation_ids.push("article-一", "unknown-source");
  data.source_documents = [{ ...data.sources[0], id: "retrieved-only", text: "未引用的检索原文" }];
  await app.respond(app.asks()[0], data);
  assert.equal(app.document.querySelectorAll(".citation").length, 1);
  assert.equal(app.$(".citation").attributes["aria-haspopup"], "dialog");
  assert.equal(app.$("#ask-results").textContent.includes("未引用的检索原文"), false);
  assert.equal(app.$(".answer-card").textContent.includes(data.claims[0].text), true);
  assert.equal(app.$(".answer-card").querySelectorAll("script").length, 0);
});

test("sending reveals the newest turn inside the chat and a late answer leaves reading position alone", async () => {
  const app = await setup();
  app.send("第一个问题");
  await app.respond(app.asks()[0], answer("第一个问题"));
  const history = app.$("#ask-results");
  history.scrollHeight = 1800;
  history.scrollTop = 200;
  app.send("新的问题");
  assert.ok(history.scrollTop > 200);
  assert.equal(app.document.viewportScrollTarget, undefined);
  history.scrollTop = 45;  // The reader returns to an earlier message while waiting.
  await app.respond(app.asks()[1], answer("新的问题"));
  assert.equal(history.scrollTop, 45);
  assert.equal(app.document.viewportScrollTarget, undefined);
});

test("a free-form clarification accepts typed facts while its notice stays in the answer", async () => {
  const app = await setup();
  app.send("是的");
  const data = answer("是的");
  Object.assign(data, { status: "insufficient_evidence", answer: "暂时还不能判断。", claims: [], sources: [], notice: "请先补充事实。" });
  await app.respond(app.asks()[0], data);
  const reply = app.$(".assistant-message");
  assert.equal(reply.childElementCount, 1);
  for (const text of [data.answer, data.notice]) {
    assert.equal(app.$(".answer-card").textContent.includes(text), true);
  }
  const panel = app.$("#follow-up-panel");
  assert.equal(panel.hidden, false);
  assert.ok(panel.textContent.includes(data.follow_up_questions[0]));
  assert.equal(panel.querySelectorAll("button").length, 0);
  assert.equal(app.$("#ask-results").textContent.includes(data.follow_up_questions[0]), false);
  assert.equal(reply.querySelectorAll(".citation").length, 0);
  app.send("没有设置非机动车道。");
  assert.deepEqual(app.asks()[1].body, { question: "没有设置非机动车道。", chat_state: data.chat_state });
});

test("shows at most three current questions as text and submits only what the user types", async () => {
  const app = await setup();
  const first = answer("骑车穿行会怎么处理？");
  first.follow_up_questions = [
    "骑行的是什么车辆？", "行驶在哪条车道？", "当时是否从两辆机动车之间穿过？", "有没有发生碰撞？",
  ];
  app.send("骑车穿行会怎么处理？");
  await app.respond(app.asks()[0], first);

  const panel = app.$("#follow-up-panel");
  const history = app.$("#ask-results");
  const input = app.$("#ask-question");
  assert.equal(panel.hidden, false);
  assert.ok(app.$(".chat-composer").contains(panel));
  assert.equal(history.contains(panel), false);
  assert.deepEqual(panel.querySelectorAll("li").map((item) => item.textContent), first.follow_up_questions.slice(0, 3));
  assert.equal(panel.querySelectorAll("button, input, textarea").length, 0);
  for (const question of first.follow_up_questions) assert.equal(history.textContent.includes(question), false);
  assert.equal(input.value, "");

  const typed = "我骑的是电动自行车。现场在武汉，没有发生碰撞。";
  input.value = typed;
  input.dispatch("input");
  assert.equal(app.asks().length, 1);
  app.$("#ask-form").requestSubmit();
  assert.deepEqual(app.asks()[1].body, { question: typed, chat_state: first.chat_state });
  assert.equal(panel.hidden, true);
  assert.equal(panel.textContent, "");
  const second = answer(typed, { messages: first.chat_state.messages, suffix: "二" });
  second.follow_up_questions = [];
  await app.respond(app.asks()[1], second);
  assert.equal(panel.hidden, true);
  assert.ok(history.textContent.includes(typed));
  for (const question of first.follow_up_questions) assert.equal(history.textContent.includes(question), false);
});

test("welcome examples fill the shared composer and remain usable after clearing chat", async () => {
  const app = await setup();
  const example = app.$("[data-example]");
  example.click();
  assert.equal(app.$("#ask-question").value, example.dataset.example);
  app.$("#ask-form").requestSubmit();
  assert.equal(app.$("#chat-welcome"), null);
  const first = answer(example.dataset.example);
  await app.respond(app.asks()[0], first);
  assert.ok(app.$("#follow-up-panel").textContent.includes(first.follow_up_questions[0]));
  app.$("#clear-chat").click();
  assert.ok(app.$("#chat-welcome"));
  assert.equal(app.$("#follow-up-panel").hidden, true);
  assert.equal(app.$("#follow-up-panel").textContent, "");
  app.$("[data-example]").click();
  assert.equal(app.$("#ask-question").value, example.dataset.example);
  assert.equal(app.document.activeElement, app.$("#ask-question"));
});

test("failed and cancelled retries retain successful state without duplicate cards", async () => {
  const app = await setup();
  app.send("原有情况");
  const first = answer("原有情况");
  first.follow_up_questions = ["事发地在哪里？", "是什么车辆？"];
  await app.respond(app.asks()[0], first);
  const panel = app.$("#follow-up-panel");
  assert.ok(panel.textContent.includes(first.follow_up_questions[0]));
  app.send("补充情况");
  await app.fail(app.asks()[1]);
  assert.equal(panel.hidden, true);
  assert.equal(panel.textContent, "");
  assert.equal(app.$("#ask-question").value, "补充情况");
  app.$(".error-panel button").click();
  assert.deepEqual(app.asks()[2].body.chat_state, first.chat_state);
  assert.equal(app.document.querySelectorAll(".chat-turn").length, 2);
  app.$(".loading-state button").click();
  const cancelled = answer("不应写入的响应");
  cancelled.follow_up_questions = ["碰撞发生在路口吗？"];
  await app.respond(app.asks()[2], cancelled);
  assert.ok(app.$(".cancelled-message"));
  assert.equal(panel.hidden, true);
  assert.equal(panel.textContent, "");
  app.$(".cancelled-message button").click();
  assert.deepEqual(app.asks()[3].body, { question: "补充情况", chat_state: first.chat_state });
  const retried = answer("补充情况", { messages: first.chat_state.messages });
  retried.follow_up_questions = ["事发时是否在机动车道内行驶？", "现场有没有非机动车道？"];
  await app.respond(app.asks()[3], retried);
  assert.deepEqual(panel.querySelectorAll("li").map((item) => item.textContent), retried.follow_up_questions);
  assert.equal(app.document.querySelectorAll(".chat-turn").length, 2);
  assert.equal(app.document.querySelectorAll(".answer-card").length, 2);
});

test("clear during generation isolates stale responses and finally from a new chat", async () => {
  const app = await setup();
  app.send("旧问题");
  const old = app.asks()[0];
  assert.equal(app.$("#clear-chat").disabled, false);
  app.$("#clear-chat").click();
  assert.equal(old.options.signal.aborted, true);
  assert.equal(app.document.querySelectorAll(".chat-turn").length, 0);
  assert.equal(app.$("#ask-question").value, "");
  app.send("新问题");
  const current = app.asks()[1];
  assert.deepEqual(current.body.chat_state, { summary: "", messages: [] });
  const stale = answer("旧问题");
  stale.follow_up_questions = ["之前是否经过这个路口？"];
  await app.respond(old, stale);
  assert.equal(app.$("#follow-up-panel").textContent, "");
  assert.equal(app.$("#follow-up-panel").hidden, true);
  assert.equal(app.$("#send-question").disabled, true);
  assert.equal(app.$("#ask-results").attributes["aria-busy"], "true");
  assert.equal(app.$("#ask-results").textContent.includes("旧问题"), false);
  await app.respond(current, answer("新问题"));
  assert.equal(app.$("#send-question").disabled, false);
  assert.equal(app.document.querySelectorAll(".answer-card").length, 1);
});

test("clear aborts old original-text dialog and prevents its late content returning", async () => {
  const app = await setup();
  app.send("提问内容");
  const first = answer("提问内容");
  await app.respond(app.asks()[0], first);
  app.$(".citation").click();
  const detail = app.calls.at(-1);
  app.$("#clear-chat").click();
  assert.equal(detail.options.signal.aborted, true);
  assert.equal(app.$("#detail-dialog").open, false);
  await app.respond(detail, first.sources[0]);
  assert.equal(app.$("#dialog-content").textContent, "");
  assert.equal(app.$("#detail-dialog").open, false);
});

test("switching upload tabs preserves chat; navigation and back-forward restore clear it", async () => {
  const app = await setup();
  app.send("原有问题");
  const first = answer("原有问题");
  await app.respond(app.asks()[0], first);
  app.location.hash = "#knowledge";
  app.window.dispatch("hashchange");
  await flush();
  assert.equal(app.document.querySelectorAll(".answer-card").length, 1);
  app.location.hash = "#ask";
  app.window.dispatch("hashchange");
  app.send("直接补充");
  assert.deepEqual(app.asks()[1].body.chat_state, first.chat_state);
  const inFlight = app.asks()[1];
  app.window.dispatch("pagehide", { persisted: true });
  assert.equal(inFlight.options.signal.aborted, true);
  assert.equal(app.document.querySelectorAll(".chat-turn").length, 0);
  app.$("#ask-question").value = "浏览器恢复的旧草稿";
  app.window.dispatch("pageshow", { persisted: true });
  assert.equal(app.$("#ask-question").value, "");
  app.send("新聊天");
  assert.deepEqual(app.asks()[2].body.chat_state, { summary: "", messages: [] });
  assert.equal(app.calls.some((call) => call.options.method === "DELETE"), false);
});

test("missing response context is retryable and a changed question replaces the failed attempt", async () => {
  const app = await setup();
  app.send("旧问题");
  const invalid = answer("旧问题");
  delete invalid.chat_state;
  await app.respond(app.asks()[0], invalid);
  assert.ok(app.$(".error-panel"));
  app.$("#ask-question").value = "改过的问题";
  app.$("#ask-question").dispatch("input");
  app.$(".error-panel button").click();
  assert.equal(app.document.querySelectorAll(".chat-turn").length, 1);
  assert.deepEqual(app.asks()[1].body, { question: "改过的问题", chat_state: { summary: "", messages: [] } });
  assert.equal(app.$("#ask-results").textContent.includes("旧问题"), false);
});
