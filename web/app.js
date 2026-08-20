/**
 * Frontend do assistente de EPI.
 *
 * Consome /api/ask/stream via EventSource. A ordem dos eventos é garantida pelo
 * backend: `sources` chega antes do primeiro `delta`, então o inspector já está
 * montado quando o texto começa a aparecer.
 *
 * Sem framework, sem build step — é um PoC de pesquisa, e a interface tem que
 * poder ser lida inteira em cinco minutos.
 */

const REFUSAL_MARKER = "CONTEXTO_INSUFICIENTE";

const conversation = document.getElementById("conversation");
const emptyState = document.getElementById("empty-state");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question");
const strategySelect = document.getElementById("strategy");
const sendButton = document.getElementById("send");
const badges = document.getElementById("badges");

let streaming = false;

/* ------------------------------------------------------------------ utils */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

/* ------------------------------------------------------------------ setup */

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error(await response.text());
    const config = await response.json();

    document.getElementById("company-name").textContent = config.company_display_name;
    document.getElementById("index-summary").textContent =
      `${config.document_count} documentos · ${config.chunk_count} chunks · índice de ${config.index_built_at.slice(0, 10)}`;

    badges.replaceChildren(
      el("span", "badge", `busca: ${config.strategy}`),
      el("span", "badge", `embeddings: ${config.embedding_model.split("/").pop()} (local)`),
      el(
        "span",
        `badge ${config.llm_is_local ? "local" : "cloud"}`,
        `LLM: ${config.llm_model}${config.llm_is_local ? " (local)" : " (nuvem)"}`,
      ),
    );
    strategySelect.value = "";
  } catch (error) {
    badges.replaceChildren(el("span", "badge cloud", "índice indisponível"));
    document.getElementById("index-summary").textContent =
      "não foi possível carregar o índice — rode `python -m rag_ic.ingest.build_index`";
    console.error(error);
  }
}

/* -------------------------------------------------------------- inspector */

function renderSource(scored, number) {
  const chunk = scored.chunk;
  const card = el("article", "source");
  card.id = `source-${number}`;

  const head = el("div", "source-head");
  head.append(
    el("span", "source-number", `[${number}]`),
    el("span", "source-title", chunk.doc_title),
    el(
      "span",
      "source-path",
      [chunk.section_path.join(" > "), chunk.page ? `p. ${chunk.page}` : null, chunk.chunk_id]
        .filter(Boolean)
        .join(" · "),
    ),
  );

  const scores = el("div", "scores");
  scores.append(
    el("span", "score", `RRF ${fmt(scored.fused_score, 4)}`),
    el(
      "span",
      `score${scored.dense_score === null ? " absent" : ""}`,
      `denso ${fmt(scored.dense_score)}${scored.dense_rank ? ` (#${scored.dense_rank})` : ""}`,
    ),
    el(
      "span",
      `score${scored.bm25_score === null ? " absent" : ""}`,
      `BM25 ${fmt(scored.bm25_score, 2)}${scored.bm25_rank ? ` (#${scored.bm25_rank})` : ""}`,
    ),
    el("span", "score", `${chunk.token_count} tokens`),
    el("span", "score", chunk.source_format),
  );

  card.append(head, scores, el("p", "source-text", chunk.text));
  return card;
}

function renderInspector(payload) {
  const details = el("details", "inspector");
  const count = payload.sources.length;
  details.append(
    el(
      "summary",
      null,
      `${count} trecho${count === 1 ? "" : "s"} recuperado${count === 1 ? "" : "s"} · ` +
        `estratégia ${payload.strategy} · ${payload.retrieval_ms} ms`,
    ),
  );
  const list = el("div", "sources");
  payload.sources.forEach((scored, index) => list.append(renderSource(scored, index + 1)));
  details.append(list);
  return details;
}

/* ---------------------------------------------------------------- answers */

/** Substitui [n] por botões que abrem o inspector e destacam o trecho citado. */
function renderAnswerText(container, text, inspector) {
  container.replaceChildren();

  let refused = false;
  let body = text;
  if (body.trimStart().toUpperCase().startsWith(REFUSAL_MARKER)) {
    refused = true;
    body = body.trimStart().slice(REFUSAL_MARKER.length).replace(/^[\s:.]+/, "");
    container.classList.add("refused");
    container.append(el("span", "refusal-tag", "contexto insuficiente"));
    container.append(document.createElement("br"));
  }

  const pattern = /\[(\d{1,2})\]/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(body)) !== null) {
    if (match.index > cursor) {
      container.append(document.createTextNode(body.slice(cursor, match.index)));
    }
    const number = match[1];
    const button = el("button", "cite", number);
    button.type = "button";
    button.title = `ver trecho ${number}`;
    button.addEventListener("click", () => {
      if (!inspector) return;
      inspector.open = true;
      const target = inspector.querySelector(`#source-${number}`);
      if (!target) return;
      inspector
        .querySelectorAll(".source.highlight")
        .forEach((node) => node.classList.remove("highlight"));
      target.classList.add("highlight");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    container.append(button);
    cursor = match.index + match[0].length;
  }
  container.append(document.createTextNode(body.slice(cursor)));
  return refused;
}

function ask(question) {
  if (streaming) return;
  streaming = true;
  sendButton.disabled = true;
  emptyState?.remove();

  const turn = el("section", "turn");
  turn.append(el("div", "question", question));
  const answer = el("div", "answer");
  answer.append(el("span", "cursor", "▌"));
  turn.append(answer);
  conversation.append(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "start" });

  const params = new URLSearchParams({ q: question });
  if (strategySelect.value) params.set("strategy", strategySelect.value);

  const source = new EventSource(`/api/ask/stream?${params}`);
  let buffer = "";
  let inspector = null;
  let finished = false;

  const finish = () => {
    if (finished) return;
    finished = true;
    streaming = false;
    sendButton.disabled = false;
    source.close();
    questionInput.focus();
  };

  source.addEventListener("sources", (event) => {
    const payload = JSON.parse(event.data);
    inspector = renderInspector(payload);
    turn.append(inspector);
  });

  source.addEventListener("delta", (event) => {
    buffer += JSON.parse(event.data).text;
    renderAnswerText(answer, buffer, inspector);
    answer.append(el("span", "cursor", "▌"));
  });

  source.addEventListener("done", (event) => {
    const payload = JSON.parse(event.data);
    renderAnswerText(answer, buffer, inspector);
    turn.append(
      el(
        "div",
        "meta",
        `${payload.provider}/${payload.model} · geração ${payload.generation_ms} ms` +
          (payload.refused ? " · recusa" : ""),
      ),
    );
    finish();
  });

  const fail = (message) => {
    answer.classList.add("error");
    answer.textContent = `Erro: ${message}`;
    finish();
  };

  // Erro reportado pelo backend (recuperação ou geração falhou).
  source.addEventListener("failure", (event) => fail(JSON.parse(event.data).message));

  // Erro de transporte do próprio EventSource. Sem isto o navegador tentaria
  // reconectar sozinho e reexecutaria a pergunta — inclusive a chamada ao LLM.
  source.addEventListener("error", () => {
    if (!finished) fail("conexão interrompida");
  });
}

/* ---------------------------------------------------------------- eventos */

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question.length < 3) return;
  questionInput.value = "";
  ask(question);
});

document.getElementById("examples")?.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button) ask(button.textContent.trim());
});

loadConfig();
questionInput.focus();
