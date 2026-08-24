import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const TMP = path.join(ROOT, "tmp/pdfs/functional_slowheat_deck");
const OUT_PPTX = path.join(TMP, "functional_slowheat_apresentacao.pptx");
const W = 1280;
const H = 720;

const C = {
  ink: "#0B1220",
  muted: "#526071",
  light: "#EDEDED",
  panel: "#F4F7FA",
  rule: "#B8BCC4",
  blue: "#3D8DFF",
  cyan: "#6DCBF4",
  pale: "#D0EDFA",
  green: "#2E9D72",
  red: "#C64E4E",
  white: "#FFFFFF",
};

const URL = {
  adam: "https://arxiv.org/abs/1412.6980",
  adamw: "https://arxiv.org/abs/1711.05101",
  replay: "https://arxiv.org/abs/1902.10486",
  derpp: "https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html",
  erace: "https://openreview.net/forum?id=N8MaByOzUfb",
  agem: "https://arxiv.org/abs/1812.00420",
  gem: "https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html",
  ewc: "https://doi.org/10.1073/pnas.1611835114",
  onlineEwc: "https://proceedings.mlr.press/v80/schwarz18a.html",
  si: "https://proceedings.mlr.press/v70/zenke17a.html",
  lwf: "https://doi.org/10.1007/978-3-319-46493-0_37",
  distill: "https://arxiv.org/abs/1503.02531",
  taylor: "https://research.nvidia.com/publication/2017-04_pruning-convolutional-neural-networks-resource-efficient-inference",
  scenarios: "https://arxiv.org/abs/1904.07734",
  early: "https://doi.org/10.1016/S0893-6080(98)00010-0",
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, i === 0 ? vals[i] : Number(vals[i])]));
  });
}

const perm = parseCsv(await fs.readFile(path.join(ROOT, "results/split_mnist_protocol/permuted_mnist/aggregate.csv"), "utf8"));
const split = parseCsv(await fs.readFile(path.join(ROOT, "results/split_mnist_protocol/split_mnist_all_methods/aggregate.csv"), "utf8"));
const byName = (rows, name) => rows.find((r) => r.method === name);
const pct = (x, digits = 2) => `${(100 * x).toFixed(digits).replace(".", ",")}%`;
const pp = (x, digits = 2) => `${(100 * x).toFixed(digits).replace(".", ",")} p.p.`;

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function shape(slide, name, left, top, width, height, fill = "none", line = "none", geometry = "rect") {
  return slide.shapes.add({
    name,
    geometry,
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { style: "solid", width: 0, fill: "none" } : { style: "solid", width: 1, fill: line },
  });
}

function textBox(slide, name, text, left, top, width, height, opts = {}) {
  const s = shape(slide, name, left, top, width, height, opts.fill ?? "none", opts.line ?? "none", opts.geometry ?? "rect");
  s.text = text;
  s.text.style = {
    fontSize: opts.fontSize ?? 22,
    typeface: "Arial",
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.vAlign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: "square",
    insets: opts.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
    lineSpacing: opts.lineSpacing ?? 1.0,
  };
  return s;
}

function rule(slide, x, y, w, color = C.rule, height = 1) {
  shape(slide, `rule-${x}-${y}`, x, y, w, height, color);
}

function baseSlide(title, eyebrow = "FUNCTIONAL SLOWHEAT") {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  textBox(slide, "eyebrow", eyebrow, 42, 28, 520, 24, { fontSize: 16, bold: true, color: C.blue });
  textBox(slide, "title", title, 42, 55, 1170, 62, { fontSize: 46, bold: true, autoFit: "shrinkText" });
  rule(slide, 42, 125, 1196, C.ink, 1);
  return slide;
}

function footer(slide, page) {
  textBox(slide, `footer-${page}`, `Resultados preliminares  |  n = 10 seeds`, 42, 684, 600, 18, { fontSize: 14, color: C.muted });
  textBox(slide, `page-${page}`, String(page).padStart(2, "0"), 1170, 681, 68, 20, { fontSize: 14, color: C.muted, align: "right" });
}

function note(slide, sources) {
  slide.speakerNotes.textFrame.setText(`[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

function bullets(slide, items, x, y, w, h, opts = {}) {
  const runs = items.map((item) => ({
    bulletCharacter: "•",
    marginLeft: 24,
    indent: -14,
    spaceAfter: opts.spaceAfter ?? 12,
    runs: Array.isArray(item) ? item : [item],
  }));
  return textBox(slide, `bullets-${x}-${y}`, runs, x, y, w, h, { fontSize: opts.fontSize ?? 23, color: opts.color ?? C.ink, autoFit: "shrinkText" });
}

function linkedText(slide, name, label, uri, x, y, w, h, opts = {}) {
  return textBox(slide, name, [[{
    run: label,
    textStyle: { underline: "sng", color: C.blue, bold: opts.bold ?? false },
    link: { uri, isExternal: true },
  }]], x, y, w, h, { fontSize: opts.fontSize ?? 21.5, color: C.blue, autoFit: "shrinkText" });
}

function metricBlock(slide, x, y, w, number, label, color = C.blue) {
  textBox(slide, `metric-num-${x}-${y}`, number, x, y, w, 62, { fontSize: 48, bold: true, color });
  textBox(slide, `metric-lab-${x}-${y}`, label, x, y + 62, w, 55, { fontSize: 21.5, color: C.muted });
}

function methodCell(slide, x, y, w, h, method, desc, refLabel, refUrl) {
  textBox(slide, `method-${method}`, method, x, y, w, 46, { fontSize: 21.5, bold: true, color: C.ink, autoFit: "shrinkText" });
  textBox(slide, `desc-${method}`, desc, x, y + 48, w, h - 76, { fontSize: 21.5, color: C.muted, autoFit: "shrinkText" });
  linkedText(slide, `ref-${method}`, refLabel, refUrl, x, y + h - 24, w, 24, { fontSize: 21.5 });
}

function methodGridSlide(title, items, page) {
  const slide = baseSlide(title, "CATALOGO DOS 31 METODOS");
  const cols = 2;
  const colW = 545;
  const gap = 68;
  const rowsPerCol = Math.ceil(items.length / cols);
  const usableH = 520;
  const rowH = usableH / rowsPerCol;
  rule(slide, 620, 145, 2, C.rule, 520);
  items.forEach((it, i) => {
    const col = Math.floor(i / rowsPerCol);
    const row = i % rowsPerCol;
    const x = 42 + col * (colW + gap);
    const y = 145 + row * rowH;
    methodCell(slide, x, y, colW, rowH - 8, ...it);
    if (row < rowsPerCol - 1) rule(slide, x, y + rowH - 4, colW, C.light, 1);
  });
  footer(slide, page);
  return slide;
}

let page = 1;

{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  textBox(slide, "cover-eyebrow", "APRENDIZAGEM CONTINUA · PROTOTIPO DE PESQUISA", 42, 34, 720, 26, { fontSize: 17, bold: true, color: C.blue });
  textBox(slide, "cover-title", "Functional\nSlowHeat", 42, 104, 600, 150, { fontSize: 66, bold: true, autoFit: "none" });
  textBox(slide, "cover-subtitle", "Plasticidade por neuronio, baselines publicados e resultados preliminares em Split-MNIST e Permuted-MNIST", 42, 292, 575, 126, { fontSize: 29, color: C.muted });
  shape(slide, "cover-field", 680, 42, 558, 590, C.panel, C.rule, "roundRect");
  textBox(slide, "cover-equation", "uᵢ = |zᵢ · ∂L/∂zᵢ|", 727, 126, 470, 72, { fontSize: 43, bold: true, color: C.ink, align: "center" });
  textBox(slide, "cover-flow", "medir utilidade\n↓\nconsolidar evidencia\n↓\nreservar capacidade\n↓\nmascarar o update final", 755, 235, 414, 250, { fontSize: 29, color: C.ink, align: "center", vAlign: "middle" });
  textBox(slide, "cover-status", "31 configuracoes · 2 benchmarks · 10 seeds", 726, 543, 470, 34, { fontSize: 22, bold: true, color: C.blue, align: "center" });
  textBox(slide, "cover-date", "Revisao: 19 ago 2026", 42, 645, 400, 24, { fontSize: 18, color: C.muted });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "README.md"), path.join(ROOT, "article/manuscript.md")]);
}

{
  const slide = baseSlide("Estabilidade e plasticidade precisam coexistir");
  textBox(slide, "problem-left-head", "Quando tudo permanece plastico", 42, 164, 530, 40, { fontSize: 30, bold: true });
  textBox(slide, "problem-left-copy", "Tarefas novas sobrescrevem caminhos uteis. O modelo aprende o presente e perde desempenho nas tarefas anteriores.", 42, 220, 530, 132, { fontSize: 25, color: C.muted });
  textBox(slide, "problem-right-head", "Quando tudo e protegido", 668, 164, 530, 40, { fontSize: 30, bold: true });
  textBox(slide, "problem-right-copy", "A rede preserva o passado, mas bloqueia a aquisicao de tarefas novas. Retencao sem plasticidade tambem falha.", 668, 220, 530, 132, { fontSize: 25, color: C.muted });
  rule(slide, 620, 154, 1, C.rule, 235);
  textBox(slide, "question", "Pergunta central", 42, 430, 250, 34, { fontSize: 24, bold: true, color: C.blue });
  textBox(slide, "question-copy", "E possivel proteger seletivamente neuronios importantes e, ao mesmo tempo, garantir uma fracao minima de capacidade livre?", 42, 482, 1120, 96, { fontSize: 34, bold: true });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "README.md"), path.join(ROOT, "docs/functional_slowheat.md")]);
}

{
  const slide = baseSlide("Os benchmarks isolam dois tipos diferentes de interferencia");
  textBox(slide, "split-head", "Split-MNIST · class-incremental", 42, 154, 535, 42, { fontSize: 30, bold: true });
  bullets(slide, [
    "5 tarefas: (0,1), (2,3), (4,5), (6,7), (8,9)",
    "Uma unica cabeca de 10 classes; sem task ID no teste principal",
    "MLP 784 → 256 → 128 → 10; 10 epocas por tarefa",
    "Principal risco: competicao e calibracao entre tarefas",
  ], 42, 218, 535, 292, { fontSize: 22.5 });
  textBox(slide, "perm-head", "Permuted-MNIST · domain-incremental", 665, 154, 550, 42, { fontSize: 30, bold: true });
  bullets(slide, [
    "5 dominios com permutacoes fixas dos 784 pixels",
    "Todas as tarefas usam as mesmas 10 classes",
    "MLP 784 → 512 → 256 → 10; 10 epocas por dominio",
    "Principal risco: preservar representacoes entre dominios",
  ], 665, 218, 550, 292, { fontSize: 22.5 });
  shape(slide, "protocol-band", 42, 555, 1173, 78, C.panel, "none", "roundRect");
  textBox(slide, "protocol-band-text", "Pareado por seed · mesma inicializacao e minibatches · IC95% normal · n = 10", 69, 574, 1120, 44, { fontSize: 21.5, bold: true, align: "center", vAlign: "middle" });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "experiments/visual_generalization.py"), path.join(ROOT, "experiments/split_mnist.py"), URL.scenarios]);
}

{
  const slide = baseSlide("SlowHeat converte utilidade em protecao seletiva");
  const steps = [
    ["1", "Medir", "Acumula |z · ∂L/∂z| por unidade e normaliza dentro da camada."],
    ["2", "Consolidar", "Na fronteira da tarefa, combina a evidencia persistente por max, media ou soma."],
    ["3", "Reservar", "Um ranking limita quantas unidades ficam protegidas; o budget preserva plasticidade."],
    ["4", "Aplicar", "A mascara fatorada protege a linha de entrada e as colunas de saida do neuronio."],
  ];
  steps.forEach((s, i) => {
    const x = 42 + i * 298;
    textBox(slide, `step-num-${i}`, s[0], x, 158, 60, 58, { fontSize: 42, bold: true, color: C.blue });
    textBox(slide, `step-title-${i}`, s[1], x, 230, 260, 34, { fontSize: 29, bold: true });
    textBox(slide, `step-copy-${i}`, s[2], x, 282, 260, 190, { fontSize: 22, color: C.muted });
    if (i < 3) rule(slide, x + 278, 160, 1, C.light, 345);
  });
  shape(slide, "equation-band", 42, 532, 1173, 96, C.ink, "none", "roundRect");
  textBox(slide, "equation", "mᵢ = 1 / (1 + β · slow_heatᵢ)    ·    Mₗ[i,j] = min(m_destino[i], m_origem[j])", 72, 559, 1113, 45, { fontSize: 29, bold: true, color: C.white, align: "center", vAlign: "middle" });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "docs/functional_slowheat.md"), path.join(ROOT, "src/dual_heater/slow_heat.py"), URL.taylor]);
}

{
  const slide = baseSlide("A mascara atua no delta final - inclusive no weight decay");
  const xs = [42, 313, 584, 855];
  const labels = [
    ["Otimizador nativo", "AdamW ou SGD calcula momentum, precondicionamento e decay."],
    ["Delta completo", "O wrapper mede θ_depois − θ_antes depois do step nativo."],
    ["Mascara SlowHeat", "O delta e interpolado elemento a elemento pela plasticidade M."],
    ["Estado coerente", "follow_update mascara tambem deltas de momentos tensoriais."],
  ];
  labels.forEach((it, i) => {
    shape(slide, `opt-box-${i}`, xs[i], 190, 230, 260, i === 2 ? C.pale : C.panel, C.rule, "roundRect");
    textBox(slide, `opt-num-${i}`, String(i + 1), xs[i] + 18, 209, 44, 38, { fontSize: 29, bold: true, color: C.blue });
    textBox(slide, `opt-title-${i}`, it[0], xs[i] + 18, 266, 194, 65, { fontSize: 26, bold: true });
    textBox(slide, `opt-copy-${i}`, it[1], xs[i] + 18, 345, 194, 83, { fontSize: 21.5, color: C.muted });
    if (i < 3) textBox(slide, `opt-arrow-${i}`, "→", xs[i] + 234, 287, 36, 44, { fontSize: 36, bold: true, color: C.blue, align: "center" });
  });
  textBox(slide, "opt-warning", "Por que isso importa: escalar apenas o gradiente bruto pode ser parcialmente cancelado pela normalizacao do AdamW; o decay desacoplado tambem moveria parametros protegidos.", 42, 515, 1170, 90, { fontSize: 25, color: C.ink, bold: true });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "docs/optimizer_semantics.md"), path.join(ROOT, "src/dual_heater/optim.py"), URL.adamw]);
}

{
  const slide = baseSlide("As 31 configuracoes pertencem a quatro familias");
  const fams = [
    ["Controles", "vanilla · hard_freeze · slowheat_none · reducao global de LR", "Separam ganho real de efeitos de wiring, congelamento e taxa de aprendizado."],
    ["Regularizacao", "SlowHeat · EWC · SI", "Protegem parametros ou unidades usando importancia acumulada, sem guardar imagens antigas."],
    ["Memoria e logits", "Replay · DER++ · ER-ACE · A-GEM", "Usam memoria episodica para reintroduzir informacao antiga e controlar interferencia."],
    ["Distillation e hibridos", "LwF · distillation · SlowHeat + Replay/DER++", "Preservam respostas antigas e combinam estabilidade funcional com memoria."],
  ];
  fams.forEach((f, i) => {
    const x = 42 + (i % 2) * 606;
    const y = 155 + Math.floor(i / 2) * 235;
    textBox(slide, `fam-title-${i}`, f[0], x, y, 555, 40, { fontSize: 30, bold: true, color: i === 2 ? C.blue : C.ink });
    textBox(slide, `fam-list-${i}`, f[1], x, y + 52, 555, 50, { fontSize: 22, bold: true, color: C.muted });
    textBox(slide, `fam-copy-${i}`, f[2], x, y + 112, 555, 88, { fontSize: 22, color: C.ink });
  });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "docs/project_methods_and_results.md"), path.join(ROOT, "experiments/split_mnist_suite.py")]);
}

{
  const slide = baseSlide("Replay e DER++ fornecem o sinal antigo que o fluxo atual nao contem");
  textBox(slide, "replay-head", "Replay", 42, 166, 340, 38, { fontSize: 31, bold: true });
  bullets(slide, [
    "Armazena 20 exemplos por classe em memoria episodica.",
    "Mistura exemplos atuais e antigos no mesmo step.",
    "Evita depender apenas de uma penalidade sobre parametros.",
  ], 42, 225, 505, 240, { fontSize: 23 });
  linkedText(slide, "replay-paper", "Chaudhry et al. (2019) · Tiny Episodic Memories", URL.replay, 42, 487, 505, 30, { fontSize: 20 });
  rule(slide, 620, 154, 1, C.rule, 445);
  textBox(slide, "der-head", "DER++", 668, 166, 340, 38, { fontSize: 31, bold: true });
  textBox(slide, "der-eq", "L = CE_atual + α·MSE(logits) + β·CE_replay", 668, 233, 550, 52, { fontSize: 27, bold: true, color: C.blue });
  bullets(slide, [
    "Guarda imagens, rotulos e logits produzidos no passado.",
    "O MSE preserva as respostas escuras; a CE ancora os rotulos.",
    "Neste projeto, α = 0,5 e β = 0,5.",
  ], 668, 315, 550, 174, { fontSize: 23 });
  linkedText(slide, "der-paper", "Buzzega et al. (NeurIPS 2020) · DER++", URL.derpp, 668, 508, 550, 30, { fontSize: 20 });
  footer(slide, page++);
  note(slide, [URL.replay, URL.derpp, path.join(ROOT, "experiments/split_mnist.py")]);
}

{
  const slide = baseSlide("ER-ACE restringe classes; A-GEM projeta gradientes");
  textBox(slide, "erace-title", "ER-ACE · restringir a competicao", 42, 162, 560, 42, { fontSize: 30, bold: true });
  textBox(slide, "erace-copy", "A loss do lote atual considera apenas classes presentes; o replay continua cobrindo todas as classes vistas. A meta e reduzir mudancas abruptas e interferencia assimetrica.", 42, 225, 535, 190, { fontSize: 24, color: C.muted });
  linkedText(slide, "erace-paper", "Caccia et al. (ICLR 2022) · ER-ACE", URL.erace, 42, 440, 535, 30, { fontSize: 20 });
  textBox(slide, "agem-title", "A-GEM · projetar o gradiente", 665, 162, 560, 42, { fontSize: 30, bold: true });
  textBox(slide, "agem-copy", "Compara o gradiente atual ao gradiente de referencia da memoria. Se o produto interno for negativo, remove a componente conflitante antes do update.", 665, 225, 550, 190, { fontSize: 24, color: C.muted });
  linkedText(slide, "agem-paper", "Chaudhry et al. (ICLR 2019) · A-GEM", URL.agem, 665, 440, 550, 30, { fontSize: 20 });
  shape(slide, "interpretation-band", 42, 530, 1173, 92, C.panel, "none", "roundRect");
  textBox(slide, "interpretation-text", "A mesma memoria pode servir para repetir exemplos, preservar logits ou construir uma restricao geometrica. O mecanismo - nao apenas o buffer - determina o comportamento.", 70, 553, 1115, 52, { fontSize: 23, bold: true, align: "center", vAlign: "middle" });
  footer(slide, page++);
  note(slide, [URL.erace, URL.agem, URL.gem, path.join(ROOT, "experiments/split_mnist.py")]);
}

{
  const slide = baseSlide("EWC, SI e LwF preservam o passado sem guardar imagens");
  const methods = [
    ["EWC", "Fisher diagonal + penalidade quadratica em torno dos parametros consolidados.", URL.ewc],
    ["SI", "Importancia sinaptica acumulada pela contribuicao −gradiente × deslocamento.", URL.si],
    ["LwF / distillation", "O modelo anterior atua como professor sobre classes antigas; temperatura 2 no runner.", URL.lwf],
  ];
  methods.forEach((m, i) => {
    const x = 42 + i * 399;
    textBox(slide, `reg-title-${i}`, m[0], x, 171, 350, 43, { fontSize: 31, bold: true });
    textBox(slide, `reg-copy-${i}`, m[1], x, 239, 350, 190, { fontSize: 23.5, color: C.muted });
    linkedText(slide, `reg-link-${i}`, "Abrir artigo", m[2], x, 451, 350, 28, { fontSize: 20 });
    if (i < 2) rule(slide, x + 370, 160, 1, C.light, 356);
  });
  textBox(slide, "reg-takeaway", "No Split-MNIST class-incremental, esses metodos mantiveram informacao task-aware em graus distintos, mas nao resolveram a calibracao global sem replay.", 42, 553, 1170, 68, { fontSize: 25, bold: true });
  footer(slide, page++);
  note(slide, [URL.ewc, URL.onlineEwc, URL.si, URL.lwf, URL.distill, path.join(ROOT, "experiments/split_mnist.py")]);
}

{
  const slide = baseSlide("As ablacoes SlowHeat testam onde a protecao realmente ajuda");
  const ab = [
    ["β = 10 / 30 / 100", "forca crescente de protecao"],
    ["adaptive", "budget guiado por validacao"],
    ["native_state", "estado do AdamW segue o step nativo"],
    ["unidirectional", "protege apenas a linha de entrada"],
    ["unbudgeted", "remove a garantia de capacidade livre"],
    ["none", "wiring SlowHeat sem consolidacao"],
    ["hard_freeze", "mascara binaria exata"],
    ["hidden", "cabeca de saida fica plastica"],
  ];
  ab.forEach((a, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 42 + col * 606;
    const y = 154 + row * 111;
    textBox(slide, `ab-name-${i}`, a[0], x, y, 210, 32, { fontSize: 24, bold: true, color: C.blue });
    textBox(slide, `ab-desc-${i}`, a[1], x + 220, y, 335, 48, { fontSize: 22.5, color: C.ink });
    rule(slide, x, y + 70, 555, C.light, 1);
  });
  textBox(slide, "ab-warning", "Ablacao nao e um novo artigo: cada nome estruturado combina escolhas locais sobre sinal, escopo, budget, estado do otimizador e metodo auxiliar.", 42, 600, 1170, 48, { fontSize: 22, bold: true, color: C.muted });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "docs/project_methods_and_results.md"), path.join(ROOT, "docs/functional_slowheat.md"), URL.taylor]);
}

{
  const shDerP = byName(perm, "slowheat_derpp_hidden_beta_30_budget_0.25");
  const derP = byName(perm, "derpp");
  const selected = [
    ["Vanilla", byName(perm, "vanilla")],
    ["SlowHeat", byName(perm, "slowheat")],
    ["SI", byName(perm, "si")],
    ["Replay", byName(perm, "replay")],
    ["SH+R", byName(perm, "slowheat_replay_hidden_beta_30_budget_0.25")],
    ["DER++", derP],
    ["SH+D", shDerP],
  ];
  const slide = baseSlide("Permuted-MNIST: DER++ domina a acuracia; SlowHeat reduz o forgetting");
  textBox(slide, "perm-acc-label", "ACC final (%)", 42, 148, 540, 30, { fontSize: 24, bold: true, color: C.blue });
  slide.charts.add("bar", {
    position: { left: 42, top: 184, width: 540, height: 392 },
    categories: selected.map((x) => x[0]),
    series: [{ name: "ACC final", values: selected.map((x) => Number((100 * x[1].final_average_accuracy_mean).toFixed(1))), fill: C.blue }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: "15px", color: C.ink } },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
    xAxis: { visible: true, line: { style: "solid", width: 1, fill: C.rule }, textStyle: { fontSize: "14px", color: C.ink } },
    yAxis: { visible: true, min: 65, max: 100, majorUnit: 5, majorGridlines: { style: "solid", width: 1, fill: C.light }, textStyle: { fontSize: "15px", color: C.muted } },
  });
  textBox(slide, "perm-forget-label", "Forgetting (%)", 658, 148, 540, 30, { fontSize: 24, bold: true, color: C.green });
  slide.charts.add("bar", {
    position: { left: 658, top: 184, width: 540, height: 392 },
    categories: selected.map((x) => x[0]),
    series: [{ name: "Forgetting", values: selected.map((x) => Number((100 * x[1].average_forgetting_mean).toFixed(1))), fill: C.green }],
    hasLegend: false,
    dataLabels: { showValue: false },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
    xAxis: { visible: true, line: { style: "solid", width: 1, fill: C.rule }, textStyle: { fontSize: "14px", color: C.ink } },
    yAxis: { visible: true, min: 0, max: 35, majorUnit: 5, majorGridlines: { style: "solid", width: 1, fill: C.light }, textStyle: { fontSize: "15px", color: C.muted } },
  });
  textBox(slide, "perm-interpret", `SH + DER++: ${pct(shDerP.final_average_accuracy_mean)} ACC e ${pct(shDerP.average_forgetting_mean)} forgetting · DER++: ${pct(derP.final_average_accuracy_mean)} e ${pct(derP.average_forgetting_mean)}.`, 42, 600, 1156, 42, { fontSize: 22, color: C.ink, bold: true, align: "center" });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "results/permutated_mnist_download.csv"), path.join(ROOT, "experiments/visual_generalization.py")]);
}

{
  const shDer = byName(split, "slowheat_derpp_hidden_beta_30_budget_0.25");
  const der = byName(split, "derpp");
  const cal = byName(split, "slowheat_replay_hidden_beta_30_budget_0.25_calibrated");
  const selected = [
    ["Vanilla", byName(split, "vanilla")],
    ["ER-ACE", byName(split, "er_ace")],
    ["Replay", byName(split, "replay")],
    ["Early stop", byName(split, "replay_early_stopping")],
    ["DER++", der],
    ["SH + Replay cal.", cal],
    ["SH + DER++", shDer],
  ];
  const slide = baseSlide("Split-MNIST: SlowHeat + DER++ alcanca a maior acuracia media");
  textBox(slide, "split-acc-label", "ACC final (%)", 42, 148, 540, 30, { fontSize: 24, bold: true, color: C.blue });
  slide.charts.add("bar", {
    position: { left: 42, top: 184, width: 540, height: 392 },
    categories: selected.map((x) => x[0]),
    series: [{ name: "ACC final", values: selected.map((x) => Number((100 * x[1].final_average_accuracy_mean).toFixed(1))), fill: C.blue }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: "15px", color: C.ink } },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
    xAxis: { visible: true, line: { style: "solid", width: 1, fill: C.rule }, textStyle: { fontSize: "14px", color: C.ink } },
    yAxis: { visible: true, min: 15, max: 90, majorUnit: 15, majorGridlines: { style: "solid", width: 1, fill: C.light }, textStyle: { fontSize: "15px", color: C.muted } },
  });
  textBox(slide, "split-forget-label", "Forgetting (%)", 658, 148, 540, 30, { fontSize: 24, bold: true, color: C.green });
  slide.charts.add("bar", {
    position: { left: 658, top: 184, width: 540, height: 392 },
    categories: selected.map((x) => x[0]),
    series: [{ name: "Forgetting", values: selected.map((x) => Number((100 * x[1].average_forgetting_mean).toFixed(1))), fill: C.green }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: "15px", color: C.ink } },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
    xAxis: { visible: true, line: { style: "solid", width: 1, fill: C.rule }, textStyle: { fontSize: "14px", color: C.ink } },
    yAxis: { visible: true, min: 0, max: 100, majorUnit: 20, majorGridlines: { style: "solid", width: 1, fill: C.light }, textStyle: { fontSize: "15px", color: C.muted } },
  });
  textBox(slide, "split-interpret", `SH + DER++: ${pct(shDer.final_average_accuracy_mean)} ACC e ${pct(shDer.average_forgetting_mean)} forgetting · DER++: ${pct(der.final_average_accuracy_mean)} e ${pct(der.average_forgetting_mean)}.`, 42, 600, 1156, 42, { fontSize: 22, color: C.ink, bold: true, align: "center" });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "results/split_mniist_results.csv"), path.join(ROOT, "experiments/split_mnist.py")]);
}

{
  const names = [
    ["Vanilla", "vanilla"],
    ["LwF", "lwf_calibrated"],
    ["A-GEM", "agem"],
    ["Replay", "replay"],
    ["DER++", "derpp"],
    ["SH + DER++", "slowheat_derpp_hidden_beta_30_budget_0.25"],
  ];
  const slide = baseSlide("Classifier gap separa representacao de calibracao");
  slide.charts.add("bar", {
    position: { left: 42, top: 158, width: 760, height: 465 },
    categories: names.map((n) => n[0]),
    series: [
      { name: "Class-incremental", values: names.map((n) => 100 * byName(split, n[1]).final_average_accuracy_mean), fill: C.blue },
      { name: "Task-aware", values: names.map((n) => 100 * byName(split, n[1]).task_aware_final_accuracy_mean), fill: C.cyan },
    ],
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: "16px", color: C.ink } },
    dataLabels: { showValue: false },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
    xAxis: { visible: true, line: { style: "solid", width: 1, fill: C.rule }, textStyle: { fontSize: "16px", color: C.ink } },
    yAxis: { visible: true, min: 0, max: 100, majorUnit: 20, majorGridlines: { style: "solid", width: 1, fill: C.light }, textStyle: { fontSize: "16px", color: C.muted } },
  });
  textBox(slide, "gap-callout", "A-GEM e LwF ainda distinguem classes dentro da tarefa quando recebem o task ID, mas falham na competicao global entre os dez logits.", 850, 178, 350, 190, { fontSize: 26, bold: true });
  textBox(slide, "gap-metric", "71,41 p.p.", 850, 407, 350, 60, { fontSize: 44, bold: true, color: C.red });
  textBox(slide, "gap-label", "classifier gap do A-GEM", 850, 478, 350, 50, { fontSize: 22, color: C.muted });
  textBox(slide, "gap-meaning", "A cabeca global - nao apenas a representacao - e parte central do problema.", 850, 553, 350, 58, { fontSize: 22, bold: true, color: C.blue });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "results/split_mniist_results.csv"), path.join(ROOT, "docs/project_methods_and_results.md")]);
}

{
  const slide = baseSlide("SlowHeat melhora acuracia, mas aumenta o tempo");
  const rows = [
    ["Replay + early stopping", "77,46%", "3,85 s", "135,6 G", "0,629 MB"],
    ["Replay", "76,27%", "5,51 s", "201,9 G", "0,629 MB"],
    ["DER++", "82,50%", "5,59 s", "201,9 G", "0,629 MB + logits"],
    ["SlowHeat + DER++", "85,06%", "8,15 s", "202,3 G", "0,629 MB + logits"],
  ];
  const headers = ["Metodo", "ACC", "Tempo", "FLOPs", "Memoria"];
  const xs = [42, 470, 645, 815, 995];
  const ws = [410, 145, 145, 150, 220];
  headers.forEach((h, i) => textBox(slide, `cost-h-${i}`, h, xs[i], 162, ws[i], 32, { fontSize: 21.5, bold: true, color: C.muted }));
  rule(slide, 42, 204, 1173, C.ink, 1);
  rows.forEach((r, ri) => {
    const y = 226 + ri * 76;
    r.forEach((v, ci) => textBox(slide, `cost-${ri}-${ci}`, v, xs[ci], y, ws[ci], 40, { fontSize: 23, bold: ci === 0 || ri === 3, color: ri === 3 && ci === 1 ? C.blue : C.ink }));
    rule(slide, 42, y + 55, 1173, C.light, 1);
  });
  metricBlock(slide, 42, 548, 330, "+45,8%", "tempo: SlowHeat + DER++ vs. DER++", C.red);
  metricBlock(slide, 455, 548, 330, "+0,20%", "FLOPs estimados no mesmo contraste", C.green);
  textBox(slide, "cost-why", "A diferenca sugere overhead de snapshots, estado e aplicacao de mascaras que a contagem teorica de FLOPs nao captura.", 850, 555, 365, 76, { fontSize: 22, bold: true });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "results/split_mniist_results.csv"), path.join(ROOT, "docs/optimizer_semantics.md")]);
}

{
  const slide = baseSlide("Quatro conclusoes resistem aos dois benchmarks");
  const concl = [
    ["01", "Replay e essencial em Class-IL", "SlowHeat isolado nao evita o colapso para ~20% no Split-MNIST."],
    ["02", "DER++ e o baseline mais forte", "Melhora claramente sobre Replay com custo e memoria adicionais pequenos."],
    ["03", "SlowHeat funciona melhor como complemento", "A combinacao com DER++ e o resultado mais promissor; com Replay simples, o ganho e pequeno."],
    ["04", "A cabeca de saida importa", "Hidden-only e calibracao de logits mudam fortemente o resultado class-incremental."],
  ];
  concl.forEach((c, i) => {
    const x = 42 + (i % 2) * 606;
    const y = 154 + Math.floor(i / 2) * 235;
    textBox(slide, `con-num-${i}`, c[0], x, y, 65, 42, { fontSize: 32, bold: true, color: C.blue });
    textBox(slide, `con-title-${i}`, c[1], x + 76, y, 475, 43, { fontSize: 27, bold: true });
    textBox(slide, `con-copy-${i}`, c[2], x + 76, y + 63, 475, 104, { fontSize: 23, color: C.muted });
  });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "results/split_mniist_results.csv"), path.join(ROOT, "results/permutated_mnist_download.csv")]);
}

{
  const slide = baseSlide("A proxima etapa e confirmacao pareada");
  metricBlock(slide, 42, 162, 290, "10", "seeds agregadas em cada CSV");
  metricBlock(slide, 410, 162, 300, "2", "benchmarks MNIST complementares");
  metricBlock(slide, 790, 162, 380, "0", "resultados Split-CIFAR versionados");
  rule(slide, 42, 320, 1173, C.rule, 1);
  bullets(slide, [
    "Reportar diferencas pareadas por seed, bootstrap/t de Student e tamanho de efeito.",
    "Pre-registrar uma confirmacao especifica de SlowHeat + DER++ contra DER++.",
    "Separar comparacoes por mesmas epocas, mesmos exemplos e mesmo custo observado.",
    "Validar em Split-CIFAR-10/100 e arquivar configuracao, matrizes, ambiente e hash do commit.",
  ], 42, 365, 1120, 235, { fontSize: 24 });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "README.md"), path.join(ROOT, "docs/confirmatory_protocol.md"), path.join(ROOT, "docs/reproducibility.md")]);
}

methodGridSlide("Controles e nucleo do SlowHeat", [
  ["vanilla", "Fine-tuning sequencial com AdamW; controle sem mecanismo de continual learning.", "AdamW", URL.adamw],
  ["slowheat", "SlowHeat completo, β=30 e budget 0,25; protege tambem a saida por padrao.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_beta_10", "Mesma regra, protecao suave mais fraca: m = 1/(1+10·heat).", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_beta_30", "Configuracao explicita equivalente ao SlowHeat padrao do runner.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_beta_100", "Protecao mais forte; privilegia estabilidade e reduz plasticidade efetiva.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_adaptive", "Atualiza o budget por sinal de aquisicao em validacao separada.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_native_state", "Mascara parametros, mas deixa momentos do AdamW seguirem o step nativo.", "AdamW", URL.adamw],
], page++);
note(presentation.slides.items.at(-1), [path.join(ROOT, "docs/functional_slowheat.md"), path.join(ROOT, "docs/optimizer_semantics.md"), URL.taylor, URL.adamw]);

methodGridSlide("Ablacoes estruturais do SlowHeat", [
  ["slowheat_unidirectional", "Remove a protecao fatorada downstream; protege apenas linhas de entrada.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_unbudgeted", "Remove a garantia de uma fracao minima de unidades totalmente plasticas.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_none", "Registra o mecanismo, mas nao consolida importancia; controle de wiring.", "AdamW", URL.adamw],
  ["hard_freeze", "Converte a protecao em mascara binaria: unidades consolidadas ficam congeladas.", "Controle local · contexto EWC", URL.ewc],
  ["slowheat_hidden_beta_30_budget_0.25", "SlowHeat apenas nas camadas ocultas, sem replay; cabeca livre.", "Contexto Taylor (metodo local)", URL.taylor],
  ["slowheat_replay_hidden_adaptive\n_beta_30_budget_0.25", "Replay + hidden-only + budget adaptado por validacao.", "SlowHeat local + Replay", URL.replay],
], page++);
note(presentation.slides.items.at(-1), [path.join(ROOT, "docs/functional_slowheat.md"), URL.taylor, URL.adamw, URL.ewc, URL.replay]);

methodGridSlide("Memoria, logits, restricoes e hibridos", [
  ["replay", "Mistura minibatches atuais e amostras de uma memoria episodica pequena.", "Tiny Episodic Memories", URL.replay],
  ["distillation", "Professor anterior preserva logits antigos, sem guardar imagens antigas.", "Knowledge Distillation", URL.distill],
  ["slowheat_replay", "Combina replay com SlowHeat completo, incluindo protecao da cabeca.", "SlowHeat local + Replay", URL.replay],
  ["slowheat_distillation", "Combina protecao funcional com professor anterior e distillation.", "SlowHeat local + Distillation", URL.distill],
  ["derpp", "CE atual + MSE de logits armazenados + CE sobre exemplos de replay.", "DER++", URL.derpp],
  ["slowheat_derpp_hidden_beta_30_budget_0.25", "DER++ + SlowHeat nas camadas ocultas; cabeca permanece plastica.", "SlowHeat local + DER++", URL.derpp],
], page++);
note(presentation.slides.items.at(-1), [URL.replay, URL.distill, URL.derpp]);

methodGridSlide("Restricoes, regularizacao e Replay balanceado", [
  ["er_ace", "Restringe classes na loss atual e usa replay sobre todas as classes vistas.", "ER-ACE", URL.erace],
  ["agem", "Projeta o gradiente atual quando ele conflita com o gradiente da memoria.", "A-GEM", URL.agem],
  ["ewc", "Fisher diagonal online + penalidade quadratica sobre parametros consolidados.", "EWC", URL.ewc],
  ["si", "Importancia sinaptica acumulada ao longo da trajetoria de otimizacao.", "Synaptic Intelligence", URL.si],
  ["lwf_calibrated", "LwF com pesos das losses definidos pela fracao de classes antigas e novas.", "Learning without Forgetting", URL.lwf],
  ["replay_balanced", "Da peso 0,5 para loss atual e 0,5 para replay, independentemente do batch.", "Variacao local de Replay", URL.replay],
], page++);
note(presentation.slides.items.at(-1), [URL.erace, URL.agem, URL.ewc, URL.si, URL.lwf, URL.replay]);

methodGridSlide("Politicas de treino e hibridos finais", [
  ["replay_more_epochs", "Executa 20 epocas por tarefa; controle de orcamento, nao novo algoritmo.", "Replay", URL.replay],
  ["replay_early_stopping", "Ate 30 epocas, paciencia 3 e restauracao do melhor estado de validacao.", "Early stopping + Replay", URL.early],
  ["replay_global_lr_reduction", "Reduz globalmente a LR por 1/31; controle para comparar seletividade.", "AdamW + Replay", URL.adamw],
  ["slowheat_replay_hidden_beta_30_budget_0.25", "Replay + SlowHeat hidden-only, β=30 e pelo menos 25% de unidades livres.", "SlowHeat local + Replay", URL.replay],
  ["slowheat_replay_partial_output_beta_30_budget_0.25", "Replay com protecao parcial da camada de saida, em vez de totalmente livre.", "SlowHeat local + Replay", URL.replay],
  ["slowheat_replay_hidden_beta_30_budget_0.25\n_calibrated", "Adiciona offset local de logits para reduzir viés entre tarefas.", "SlowHeat local + Replay", URL.replay],
], page++);
note(presentation.slides.items.at(-1), [path.join(ROOT, "docs/project_methods_and_results.md"), URL.replay, URL.early, URL.adamw]);

{
  const slide = baseSlide("Referencias principais - links clicaveis", "ARTIGOS E FONTES");
  const refs = [
    ["Adam · Kingma & Ba (2015)", URL.adam],
    ["AdamW · Loshchilov & Hutter (2019)", URL.adamw],
    ["Tiny Episodic Memories · Chaudhry et al. (2019)", URL.replay],
    ["DER++ · Buzzega et al. (NeurIPS 2020)", URL.derpp],
    ["ER-ACE · Caccia et al. (ICLR 2022)", URL.erace],
    ["A-GEM · Chaudhry et al. (ICLR 2019)", URL.agem],
    ["EWC · Kirkpatrick et al. (PNAS 2017)", URL.ewc],
    ["Synaptic Intelligence · Zenke et al. (ICML 2017)", URL.si],
    ["Learning without Forgetting · Li & Hoiem (ECCV 2016)", URL.lwf],
    ["Knowledge Distillation · Hinton et al. (2015)", URL.distill],
    ["Taylor saliency · Molchanov et al. (ICLR 2017)", URL.taylor],
    ["Cenarios de continual learning · van de Ven & Tolias (2019)", URL.scenarios],
  ];
  refs.forEach((r, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    linkedText(slide, `main-ref-${i}`, r[0], r[1], 42 + col * 606, 156 + row * 76, 555, 36, { fontSize: 22 });
  });
  textBox(slide, "local-note", "Functional SlowHeat, DualHeat e as combinacoes/ablacoes sao contribuicoes locais ainda sem artigo externo revisado por pares.", 42, 625, 1120, 34, { fontSize: 21.5, bold: true });
  footer(slide, page++);
  note(slide, refs.map((r) => r[1]).concat([path.join(ROOT, "article/manuscript.md")]));
}

{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  textBox(slide, "closing-eyebrow", "SINTESE", 42, 40, 220, 26, { fontSize: 17, bold: true, color: C.blue });
  textBox(slide, "closing-title", "SlowHeat e promissor como complemento ao DER++ - ainda nao como melhoria geral confirmada.", 42, 173, 1100, 220, { fontSize: 58, bold: true, autoFit: "shrinkText" });
  textBox(slide, "closing-sub", "Proxima decisao: confirmar o contraste pareado e validar fora do MNIST.", 42, 510, 790, 75, { fontSize: 30, color: C.muted });
  textBox(slide, "closing-data", "85,06% Split · 95,56% Permuted · n=10", 42, 614, 700, 36, { fontSize: 23, bold: true, color: C.blue });
  footer(slide, page++);
  note(slide, [path.join(ROOT, "results/split_mniist_results.csv"), path.join(ROOT, "results/permutated_mnist_download.csv"), path.join(ROOT, "README.md")]);
}

await fs.mkdir(path.join(TMP, "rendered"), { recursive: true });
for (const [i, slide] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide, format: "png", scale: 2 });
  await fs.writeFile(path.join(TMP, "rendered", `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(TMP, "rendered", `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(TMP, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT_PPTX);
console.log(JSON.stringify({ slides: presentation.slides.items.length, pptx: OUT_PPTX }));
