# RAG-IC — assistente interno de EPI com RAG local

Prova de conceito de Iniciação Científica: um sistema de perguntas e respostas sobre a
documentação interna de EPI (Equipamento de Proteção Individual) de uma empresa, em que
**a indexação e a busca acontecem inteiramente na máquina local**.

A motivação da pesquisa é privacidade. Uma empresa que quer um assistente sobre seus
próprios procedimentos de segurança do trabalho não quer enviar o acervo inteiro para um
provedor de nuvem. Aqui o acervo é carregado, fatiado, vetorizado e pesquisado localmente;
apenas os trechos efetivamente recuperados chegam ao modelo de linguagem — e mesmo isso é
uma escolha de configuração, não da arquitetura (ver [Limitações](#limitações)).

---

## Sumário

- [O que o sistema faz](#o-que-o-sistema-faz)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso](#uso)
- [Avaliação](#avaliação)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Decisões de projeto](#decisões-de-projeto)
- [Limitações](#limitações)
- [Guia de estudos](#guia-de-estudos)

---

## O que o sistema faz

O acervo de teste é o de uma empresa fictícia, a **Metalúrgica Aurora Ltda.**: 32
documentos em português do Brasil — políticas, procedimentos operacionais, diretrizes
técnicas, atas da CIPA, relatório de auditoria, indicadores e planilhas de entrega —
todos ancorados no conteúdo real da NR-6, da NR-10, da NR-33 e da NR-35.

Perguntas como *"de quanto em quanto tempo troco o protetor auricular?"* ou *"qual o CA
da luva de vaqueta?"* são respondidas com citação obrigatória do trecho de origem. Quando
a resposta não está no acervo, o sistema **recusa explicitamente** em vez de improvisar —
o domínio é segurança do trabalho, e uma resposta plausível e errada sobre classe de
respirador é um risco real, não um problema de experiência de uso.

---

## Arquitetura

```
                       INDEXAÇÃO (offline, local)
  data/aurora/docs/
    *.md  *.pdf  *.csv
          |
          v
   loaders por formato ──> chunking estrutural ──> multilingual-e5-small
   (heading / página / linha)   (~400 tokens, 15% overlap)      (local, 384-d)
                                                                     |
                                                                     v
                                              index/aurora/{chunks.jsonl, vectors.npz}


                        CONSULTA (online)
  pergunta
     |
     +──> e5 "query: " ──> cosseno exato (NumPy)  ──┐
     |                                              ├──> Reciprocal Rank Fusion ──> top-5
     +──> tokenização ───> BM25 (rank_bm25)       ──┘         (k = 60)
                                                                     |
                                                                     v
                                              prompt aterrado em pt-BR + Gemini 2.5 Flash
                                                                     |
                                                                     v
                                              resposta com citações [n] via SSE
```

| Camada | Escolha | Onde roda |
|---|---|---|
| Loaders | markdown com headings, PDF via `pypdf` com página, CSV linha a linha | local |
| Chunking | estrutural com fallback recursivo por sentença | local |
| Embeddings | `intfloat/multilingual-e5-small`, 384 dimensões | local |
| Índice vetorial | matriz NumPy `float32` em `.npz`, cosseno exato | local |
| Busca léxica | BM25 com tokenizador que preserva `NR-35`, `CA 31.028`, `6.6.1` | local |
| Fusão | Reciprocal Rank Fusion, k = 60 | local |
| Geração | Gemini 2.5 Flash via `google-genai` | **nuvem** |
| API | FastAPI + Pydantic + Server-Sent Events | local |
| Interface | HTML/CSS/JS sem framework | local |

---

## Instalação

Requer Python 3.11 ou superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

O primeiro uso baixa os pesos do `multilingual-e5-small` (~470 MB) do Hugging Face. É o
único download da fase de indexação, e depois disso a indexação funciona sem rede.

### Chave da API

```bash
cp .env.example .env
```

Preencha `RAG_GEMINI_API_KEY` com uma chave obtida em <https://aistudio.google.com/apikey>.
O arquivo `.env` está no `.gitignore` — **nunca comite a chave**.

Só a geração precisa da chave. A indexação, a busca e toda a avaliação de recuperação
funcionam sem ela.

---

## Uso

### 1. Gerar os PDFs do acervo

Quatro documentos do acervo existem como PDF, para que o pipeline enfrente extração real
de texto e não apenas markdown limpo:

```bash
python scripts/make_pdfs.py --company aurora
```

### 2. Construir o índice

```bash
python -m rag_ic.ingest.build_index --company aurora
```

Escreve `index/aurora/chunks.jsonl`, `vectors.npz` e `manifest.json`. Leva alguns segundos
em CPU. O comando avisa se algum chunk ultrapassar os 512 tokens que o encoder trunca em
silêncio.

### 3. Subir o servidor

```bash
uvicorn rag_ic.api.main:app --reload
```

Abra <http://localhost:8000>. A interface mostra a resposta em streaming, as citações
clicáveis e um painel de inspeção com os trechos recuperados e seus scores denso, BM25 e
fundido — é o que transforma o protótipo em instrumento de pesquisa: quando a resposta sai
errada, dá para ver na hora se a falha foi de recuperação ou de geração.

### Endpoints

| Método | Rota | Para quê |
|---|---|---|
| `GET` | `/api/health` | estado do pipeline |
| `GET` | `/api/config` | modelo, estratégia, tamanho do índice |
| `GET` | `/api/retrieve?q=...` | **só recuperação**, sem gastar tokens |
| `POST` | `/api/ask` | resposta completa em JSON |
| `GET` | `/api/ask/stream?q=...` | resposta em streaming (SSE) |
| `GET` | `/docs` | OpenAPI gerado pelo FastAPI |

`/api/retrieve` é o endpoint mais útil durante o desenvolvimento: inspeciona o índice sem
custo nenhum.

```bash
curl "http://localhost:8000/api/retrieve?q=CA%2031.028&strategy=bm25" | python -m json.tool
```

---

## Avaliação

O conjunto de referência tem **50 perguntas respondíveis e 10 sem resposta no acervo**,
escritas à mão em `data/aurora/gold/questions.jsonl`. Cada pergunta é rotulada com o
documento e a seção que contêm a resposta — não com o identificador do chunk, que muda
sempre que o chunking muda.

As 10 perguntas sem resposta incluem distratores deliberados: o nível de ruído do setor de
Manutenção (o PGR mede quatro setores, mas não esse), o preço unitário da luva (há custo
total, nunca unitário), o resultado de uma avaliação que a ata registra como pendente.
Elas medem recusa, que é o comportamento crítico neste domínio.

```bash
python -m rag_ic.evaluation.run_eval               # só recuperação: gratuito e determinístico
python -m rag_ic.evaluation.run_eval --with-llm    # + recusa e validade das citações
```

A primeira forma imprime a ablação **denso × BM25 × híbrido** com MRR, Recall@k e Hit@k, e
grava `reports/retrieval_aurora.csv`. É o resultado central do trabalho, e não gasta um
único token.

Recall@k aqui é estrito: conta a fração das *regiões rotuladas* recuperadas, então uma
pergunta com dois documentos relevantes pontua 0,5 quando só um aparece. `Hit@k` é a
métrica leniente, para comparação.

---

## Estrutura do repositório

```
src/rag_ic/
  config.py               configuração via pydantic-settings (.env)
  models.py               Chunk, ScoredChunk, GoldQuestion, AnswerResult
  prompts.py              prompt pt-BR, citações [n], marcador de recusa
  pipeline.py             recuperação + geração, streaming e não streaming
  ingest/
    loaders.py            markdown / PDF / CSV -> blocos com seção, página, linha
    chunking.py           agrupamento estrutural + split por sentença com overlap
    tokens.py             contagem de tokens (tokenizer real ou aproximação)
    build_index.py        CLI de indexação
  embeddings/
    base.py               interface (query e passage são métodos distintos de propósito)
    local_e5.py           multilingual-e5-small local
  retrieval/
    store.py              formato do índice em disco
    dense.py              cosseno exato sobre NumPy
    lexical.py            BM25 + tokenizador que preserva identificadores
    fusion.py             Reciprocal Rank Fusion
    retriever.py          orquestra as três estratégias
  llm/
    base.py               interface de geração — o ponto central da tese
    gemini.py             Gemini 2.5 Flash
    ollama.py             provedor local (escrito, ainda não validado)
  api/main.py             FastAPI, SSE, arquivos estáticos
  evaluation/
    metrics.py            Recall@k, Hit@k, MRR, recusa
    run_eval.py           ablação e relatórios CSV
data/aurora/
  docs/                   acervo indexado (markdown, CSV, PDFs gerados)
  pdf_sources/            markdown de origem dos PDFs
  gold/questions.jsonl    50 + 10 perguntas rotuladas
web/                      interface (index.html, app.js, style.css)
scripts/make_pdfs.py      renderiza os PDFs do acervo
tests/                    pytest sobre a lógica determinística
docs/                     guia de estudos e lista de leituras
```

### Testes e lint

```bash
pytest
ruff check .
ruff format --check .
```

Os testes cobrem chunking, tokenização léxica, fusão, métricas, prompts, loaders e o
formato do índice — a parte determinística. Chamadas ao modelo de linguagem não são
testadas: são não determinísticas e custam dinheiro.

---

## Decisões de projeto

**Por que NumPy e não um banco vetorial.** O acervo produz alguns milhares de vetores. A
busca exata é uma única multiplicação de matriz, abaixo de um milissegundo. Um índice
aproximado (FAISS, HNSW) acrescentaria uma dependência, uma etapa de construção e um botão
de recall/latência para calibrar, em troca de nada. Essa conta vira por volta de 10⁵–10⁶
vetores.

**Por que híbrido e não só denso.** As perguntas deste domínio carregam identificadores:
`CA 31.028`, `NR-35`, `item 6.6.1`, lote `PN-2533-A`. Um vetor de 384 dimensões borra
`CA 31.028` e `CA 31.029` em pontos quase idênticos; BM25 não. O tokenizador em
`retrieval/lexical.py` preserva o identificador inteiro *e* emite suas partes, para que
"NR 35" na pergunta alcance "NR-35" no documento.

**Por que Reciprocal Rank Fusion.** Similaridades de cosseno vivem em [-1, 1] e ficam
espremidas; scores BM25 são ilimitados e dependem do corpus. Somar os dois exige
normalizar duas distribuições que não têm relação, e a normalização acaba decidindo o
resultado. RRF descarta os scores e usa só as posições.

**Por que rotular o gold set por documento e seção.** Identificadores de chunk mudam a cada
experimento de chunking. Um conjunto de referência que precisa ser rerrotulado a cada
experimento é um conjunto que ninguém mantém.

**Por que um marcador literal de recusa.** `CONTEXTO_INSUFICIENTE` na primeira linha é
detectável por programa. "Não tenho certeza, mas provavelmente..." não é — e é exatamente
a resposta perigosa neste domínio.

---

## Limitações

Todas conhecidas e declaradas, não descobertas depois:

1. **A privacidade é parcial nesta configuração.** O acervo e o índice nunca saem da
   máquina, mas os trechos recuperados vão para o Google no momento da geração. É por isso
   que `llm/base.py` é uma interface e `llm/ollama.py` existe: trocar para geração local é
   mudança de configuração, não reescrita.
2. **`OllamaProvider` está escrito mas não foi validado.** O Ollama não estava instalado no
   ambiente de desenvolvimento. Nenhum número deve ser citado a partir dele antes de uma
   execução real. Na GTX 1650 de 4 GB do ambiente alvo, um modelo 7B quantizado em Q4_K_M
   (~4,4 GB) transborda para a RAM; o realista é 3B–4B.
3. **O acervo é sintético.** É uma ameaça à validade externa. A mitigação é que todo
   conteúdo normativo é ancorado em NR real e o processo de geração está documentado — e a
   contrapartida é que o acervo pode ser publicado junto com o trabalho, o que um acervo
   real de empresa não poderia.
4. **Ruído de extração de PDF** entra como confundidor nas métricas. Por isso os PDFs são
   gerados a partir do mesmo texto dos markdown: o conteúdo é idêntico, só o container
   muda, então o efeito do formato pode ser medido em vez de estimado.
5. **Turno único.** Não há histórico de conversa nem reescrita de pergunta. Perguntas de
   acompanhamento ("e para soldadores?") não funcionam — decisão deliberada, para que as
   métricas do gold set descrevam exatamente o que o sistema faz em produção.
6. **Sem reranker.** Um cross-encoder seria o próximo ganho de qualidade mais provável.
7. **Risco de depreciação do modelo.** `gemini-2.5-flash` é configuração, não código. O
   parâmetro de raciocínio nem sequer é o mesmo campo entre famílias: o Gemini 3 usa o
   `thinking_level` categórico e o Gemini 2.5 usa o `thinking_budget` inteiro, recusando
   o outro com `400 INVALID_ARGUMENT` na hora da requisição. `llm/gemini.py` expõe um
   único nível semântico e mapeia para o campo que o modelo configurado aceita.

---

## Guia de estudos

`docs/guia-de-estudos.md` explica os fundamentos e as ferramentas deste sistema do zero,
com exemplos numéricos resolvidos à mão. `docs/leituras.md` traz a bibliografia primária
verificada. Ambos em português.
