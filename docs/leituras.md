# Leituras — RAG local para documentação de EPI

Lista anotada para acompanhar o [guia de estudos](./guia-de-estudos.md). Todos os links abaixo foram acessados e conferidos em **14 de agosto de 2026**; se algum tiver mudado, o título e os autores são suficientes para reencontrá-lo.

**Como está organizada:**

- **Leitura essencial** — o mínimo para entender o que você está construindo. Total: cerca de **55 minutos**.
- **Aprofundamento** — para quando o essencial já estiver assentado, ou quando uma seção específica do trabalho exigir.
- **Documentação de referência** — para consultar, não para ler de ponta a ponta.

Cada entrada está marcada como **[primária]** (artigo científico, documentação oficial ou repositório oficial) ou **[secundária]** (blog, post de fornecedor). Trate as secundárias com o ceticismo devido: são úteis e concretas, mas escritas por quem vende alguma coisa.

Os tempos de leitura são estimativas para um leitor de graduação lendo com atenção, não folheando.

---

## Leitura essencial (≈ 55 min no total)

### 1. Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods
**Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher** — SIGIR '09, Boston · 2009 · **[primária]**
http://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf

Por que ler: são **duas páginas** e contêm a fórmula inteira do RRF, incluindo a origem da constante mágica $k=60$ que aparece em toda implementação de busca híbrida — inclusive na sua. É o melhor retorno sobre tempo investido desta lista inteira. Leia antes de escrever a seção de fusão do relatório.
*Nota: o link é HTTP, não HTTPS, e é a URL final (a versão frequentemente citada em `plg.uwaterloo.ca` redireciona duas vezes).*
⏱️ **15–20 min**

### 2. Multilingual E5 Text Embeddings: A Technical Report
**Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei** (Microsoft) · 2024 · **[primária]**
https://arxiv.org/abs/2402.05672

Por que ler: descreve exatamente o modelo que você está usando (`multilingual-e5-small`) — pré-treino contrastivo sobre ~1 bilhão de pares multilíngues seguido de fine-tuning supervisionado. É um relatório técnico curto, não um artigo longo, e é a citação correta para justificar a escolha do modelo de embeddings num trabalho em português.
⏱️ **20–25 min**

### 3. sentence-transformers — Semantic Search
**Documentação oficial (Hugging Face / UKP Lab)** · **[primária]**
https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

Por que ler: a página mais diretamente aplicável ao seu código. Explica a distinção entre busca **simétrica** e **assimétrica** (que é exatamente o que motiva os prefixos `query:`/`passage:` do e5), traz o utilitário `semantic_search()`, as opções de ANN e o padrão *retrieve-and-rerank* — que é o seu trabalho futuro mais provável.
⏱️ **15–20 min**

---

## Aprofundamento

### 4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
**Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela** — NeurIPS 2020 · **[primária]**
https://arxiv.org/abs/2005.11401

Por que ler: é **o** artigo que cunhou o termo RAG, e a citação obrigatória do seu trabalho. Atenção a uma diferença importante: aqui retriever e gerador são **treinados conjuntamente** (RAG-Sequence e RAG-Token), enquanto o que a indústria chama de "RAG" hoje — e o que você está construindo — é a versão desacoplada, com um LLM congelado. Saber articular essa diferença é ponto garantido em banca.
⏱️ **50–60 min**

### 5. Retrieval-Augmented Generation for Large Language Models: A Survey
**Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang** (Tongji / Fudan) · 2023, rev. 2024 · **[primária]**
https://arxiv.org/abs/2312.10997

Por que ler: o mapa da área. Organiza tudo em Naive RAG / Advanced RAG / Modular RAG e cataloga as técnicas de recuperação, geração e avaliação. Bom para localizar o seu sistema no espaço de projeto (você está fazendo "Advanced RAG" pela busca híbrida, sem os módulos de reescrita de query) e para a seção de trabalhos relacionados. **Sugestão prática:** leia este *antes* dos artigos primários, folheando figuras e taxonomias, e depois volte aos primários que interessarem.
⏱️ **90–120 min completo; ~40 min folheando figuras e taxonomia**

### 6. The Probabilistic Relevance Framework: BM25 and Beyond
**Stephen Robertson, Hugo Zaragoza** — *Foundations and Trends in Information Retrieval*, vol. 3, n. 4, p. 333–389 · 2009 · **[primária]**
https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

Por que ler: a fonte definitiva do BM25, escrita por um dos seus autores, disponível gratuitamente. Deriva a fórmula a partir do *probability ranking principle* — ou seja, explica **de onde vêm** a saturação de TF, o IDF e a normalização por comprimento, em vez de apresentá-los como heurísticas. São 57 páginas; para o seu propósito, as seções 1 a 3.5 bastam, e a seção 3.4 ("The Eliteness Model and BM25") é o ponto alto.
⏱️ **~45 min (§1–3.5); 2,5–3 h completo**

### 7. Dense Passage Retrieval for Open-Domain Question Answering
**Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih** — EMNLP 2020 · **[primária]**
https://arxiv.org/abs/2004.04906

Por que ler: a receita de *dual-encoder* (encoders separados para pergunta e passagem, treinados com *in-batch negatives*) que fez a busca densa superar o BM25. É a base conceitual da arquitetura assimétrica do e5, e a leitura que faz o prefixo `query:`/`passage:` deixar de parecer arbitrário.
⏱️ **45–55 min**

### 8. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
**Nils Reimers, Iryna Gurevych** (UKP Lab) — EMNLP 2019 · **[primária]**
https://arxiv.org/abs/1908.10084

Por que ler: explica por que você **não** pode simplesmente alimentar um BERT com pares de textos para fazer busca por similaridade (o custo cai de ~65 horas para ~5 segundos com a abordagem siamesa), e por que *mean pooling* funciona. É a fundamentação teórica da biblioteca `sentence-transformers` que o seu código importa.
⏱️ **40–50 min**

### 9. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
**Yu. A. Malkov, D. A. Yashunin** · 2016, revisado em 2018 · **[primária]**
https://arxiv.org/abs/1603.09320

Por que ler: HNSW é o índice que está por trás de praticamente todo banco vetorial moderno (FAISS-HNSW, Qdrant, Weaviate, pgvector, Azure AI Search). Leia quando for justificar, na seção de limitações, por que a busca exata em NumPy é adequada agora e o que exatamente você adotaria ao escalar — é a diferença entre "não sabíamos" e "não precisávamos".
⏱️ **50–60 min**

### 10. Billion-scale similarity search with GPUs
**Jeff Johnson, Matthijs Douze, Hervé Jégou** (FAIR) · 2017 · **[primária]**
https://arxiv.org/abs/1702.08734

Por que ler: o artigo do FAISS — seleção de top-k em GPU, IVF e product quantization. Menos relevante que o HNSW para um sistema local, mas é a referência canônica se você citar FAISS pelo nome.
⏱️ **45–60 min**

### 11. Survey of Hallucination in Natural Language Generation
**Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Delong Chen, Wenliang Dai, Ho Shu Chan, Andrea Madotto, Pascale Fung** (HKUST) — *ACM Computing Surveys* · 2022, rev. 2024 · **[primária]**
https://arxiv.org/abs/2202.03629

Por que ler: a taxonomia de referência — alucinação **intrínseca** (contradiz a fonte) versus **extrínseca** (não verificável a partir da fonte). Essa distinção é exatamente o que você precisa para descrever com precisão o que o aterramento resolve e o que não resolve. É longo: leia §1–4 e a parte de QA, e pule o resto.
⏱️ **~45 min (§1–4 + QA); 2–3 h completo**

### 12. Ragas: Automated Evaluation of Retrieval Augmented Generation
**Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert** (Cardiff University / Exploding Gradients) · 2023 · **[primária]**
https://arxiv.org/abs/2309.15217

Por que ler: define métricas **sem referência** — *faithfulness*, *answer relevance*, *context relevance* — que podem ser calculadas sem gold set anotado. Complementa (não substitui) o seu Recall@k e MRR: essas medem recuperação, aquelas medem se a resposta gerada realmente se apoia no contexto. Útil se você quiser avaliar a geração além da checagem de citações.
⏱️ **30–40 min**

### 13. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
**Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica** (LMSYS / UC Berkeley) — NeurIPS 2023, Datasets & Benchmarks · 2023 · **[primária]**
https://arxiv.org/abs/2306.05685

Por que ler: **leia a seção de vieses antes de usar qualquer LLM como avaliador.** Documenta viés de posição, viés de verbosidade e auto-preferência — os três motivos pelos quais um juiz automático pode dar ao seu sistema uma nota que ele não merece. Diretamente ligado à discussão da seção 8.5 do guia sobre gold sets gerados por LLM.
⏱️ **50–60 min**

### 14. Introducing Contextual Retrieval
**Anthropic** · 19 de setembro de 2024 · **[secundária — post de fornecedor]**
https://www.anthropic.com/news/contextual-retrieval

Por que ler: ataca exatamente o problema da seção 3.4 do guia (chunks que perdem o contexto ao serem isolados), com uma solução mais cara que a nossa: em vez de prefixar título e caminho de seções, gerar com um LLM 50–100 tokens de contexto específicos para cada chunk antes de indexar. Traz ablações concretas de redução de taxa de erro e números de custo. Leia como referência para o trabalho futuro — e note que é um post de fornecedor, com o viés que isso implica.
⏱️ **20–25 min**

### 15. Hybrid search scoring (RRF) — Azure AI Search
**Microsoft Learn** · atualizado em 2 de julho de 2026 · **[secundária — documentação de fornecedor]**
https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking

Por que ler: a descrição mais clara em nível de implementação do RRF, com o algoritmo em quatro passos e tabelas de fusão ponderada. Contém um alerta que vale ouro para quem está começando: **o $k$ do RRF não tem relação nenhuma com o $k$ do top-k da busca vetorial** — é uma confusão comum e constrangedora numa apresentação. Traz também as faixas típicas de score por algoritmo (BM25 ilimitado; cosseno em HNSW entre ~0,333 e 1; RRF limitado por ~n/k), que ajudam a ler o painel de inspeção do seu frontend.
⏱️ **15–20 min**

---

## Documentação de referência

Para consultar quando precisar, não para ler de cabo a rabo.

| Ferramenta | Link | Para que serve consultar | Tempo |
|---|---|---|---|
| **sentence-transformers** (geral) **[primária]** | https://sbert.net/ | Ponto de entrada da biblioteca; catálogo de modelos e de cross-encoder rerankers (confirma que o caminho de reranking existe, se você for adiante). | 10–15 min |
| **Model card `intfloat/multilingual-e5-small`** **[primária]** | https://huggingface.co/intfloat/multilingual-e5-small | Licença (MIT), cobertura de 100+ idiomas e tabelas MTEB. **Leia a seção "Usage" no navegador** e confirme ali os prefixos `query:`/`passage:` e o limite de sequência — a extração automática dessa página não trouxe essa seção, então confirme com os próprios olhos antes de citar números. | 10–15 min |
| **Gemini API — Quickstart** **[primária]** | https://ai.google.dev/gemini-api/docs/quickstart | Instalação do `google-genai`, inicialização do `Client`, primeira chamada. | 10–15 min |
| **Gemini API — Text generation** **[primária]** | https://ai.google.dev/gemini-api/docs/text-generation | Geração de texto, system instructions e **streaming** — o lado que você conecta ao SSE do FastAPI. | ~15 min |
| **Gemini API — Thinking** **[primária]** | https://ai.google.dev/gemini-api/docs/thinking | **Consulte antes de escrever o relatório.** A página atual documenta `thinking_level` (`low`/`medium`/`high`), não um `thinking_budget` numérico, e indica que modelos Flash da linha 2.5 não permitem desligar o raciocínio; tokens de raciocínio são cobrados junto com os de saída. Tutoriais com `thinking_budget=0` estão desatualizados. | 10–15 min |
| **FastAPI** **[primária]** | https://fastapi.tiangolo.com/ | Rotas tipadas, OpenAPI automático, `StreamingResponse` (herdado do Starlette) para SSE. | ~15 min (home) |
| **Pydantic** **[primária]** | https://pydantic.dev/docs/validation/latest/get-started/ | Validação por type hints, coerção estrita versus permissiva, geração de JSON Schema. *(A URL antiga `docs.pydantic.dev/latest/` agora redireciona para esta.)* | ~15 min |
| **MDN — Using server-sent events** **[primária]** | https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events | O formato de fio do `text/event-stream`: campos `event:`, `data:`, `id:`, `retry:`, a linha em branco que encerra a mensagem, `EventSource`, reconexão automática e o limite de seis conexões do HTTP/1.1. Leia **antes** de escrever o endpoint de streaming à mão. | ~20 min |
| **rank_bm25** **[primária]** | https://github.com/dorianbrown/rank_bm25 | API mínima (`get_scores`, `get_top_n`) e as variantes implementadas (Okapi, BM25L, BM25+, BM25-Adpt, BM25T). Vale ler o código de `BM25Okapi` — são poucas linhas e explicam o tratamento de IDF negativo. | ~10 min |
| **pypdf** **[primária]** | https://pypdf.readthedocs.io/en/stable/ | Especificamente as páginas "Extract Text from a PDF" e **"Post-Processing of Text Extraction"** — a segunda trata ligaduras, espaçamento e ordem de leitura, exatamente o ruído que degrada os chunks vindos de PDF. | 15–20 min |
| **Ruff** **[primária]** | https://docs.astral.sh/ruff/ | Catálogo de regras e configuração via `pyproject.toml` (útil para entender os códigos `E,F,I,N,UP,B,SIM,RUF` já configurados no projeto). | ~15 min |
| **pytest** **[primária]** | https://docs.pytest.org/en/stable/ | Fixtures e `parametrize` — o suficiente para escrever os testes de chunker, tokenizer e RRF descritos na seção 9.7 do guia. | 20–30 min |
| **`numpy.savez_compressed`** **[primária]** | https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html | Assinatura e comportamento do `.npz` (um ZIP de arquivos `.npy`). Confirma que usar argumentos nomeados faz `np.load()` devolver um `NpzFile` com chaves legíveis. | 5–10 min |

---

## Notas de verificação

Coisas que ficaram **deliberadamente de fora** ou que exigem cuidado:

- **DOI do artigo de RRF na ACM Digital Library** — deixado de fora. Apenas o PDF gratuito de Waterloo foi conferido; o DOI comumente citado não foi verificado e não deve ser copiado sem checagem.
- **Detalhes internos do `multilingual-e5-small`** (dimensão do embedding, número de camadas, texto exato das instruções de prefixo) — a página do model card existe e é legítima, mas a extração automática não trouxe a seção "Usage". Confirme no navegador antes de citar números específicos.
- **`thinking_budget` numérico na Gemini API** — não foi possível confirmar nenhuma página oficial vigente que documente esse parâmetro; a documentação atual descreve `thinking_level`. A configuração `RAG_GEMINI_THINKING_BUDGET=0` do projeto reflete uma forma anterior da API e precisa ser reconciliada com a versão do SDK que você instalar.
- **Posts do Pinecone, Elastic e afins sobre estratégias de chunking e busca híbrida** — não incluídos porque não foram acessados e verificados. Os itens 14 e 15 cobrem os mesmos dois assuntos com fontes que foram efetivamente conferidas.
- **Contagem de páginas dos artigos do arXiv** — as páginas de resumo não a exibem, então as estimativas de tempo acima são julgamento baseado no tipo de publicação. As exceções verificadas são o BM25 (p. 333–389, isto é, 57 páginas) e o RRF (artigo curto de SIGIR, 2 páginas).
