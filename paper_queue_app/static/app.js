const STORAGE_KEY = "paper_queue_v1";
const FILTERS_KEY = "paper_queue_filters_v1";

const STATUS = {
  QUEUED: "queued",
  READING: "reading",
  FINISHED: "finished",
  ARCHIVED: "archived",
};

const defaultState = () => ({
  version: 1,
  projects: [{ id: "main", name: "Main" }],
  papers: [],
});

const defaultFilters = () => ({
  search: "",
  project: "all",
  status: "queue",
  includeArchived: false,
});

function uid(prefix = "id") {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function nowISO() {
  return new Date().toISOString();
}

function normalizePaper(raw) {
  const t = nowISO();
  return {
    id: raw.id || uid("paper"),
    arxivId: raw.arxivId || raw.arxiv_id || "",
    title: raw.title || "Untitled",
    abstract: raw.abstract || "",
    authors: Array.isArray(raw.authors) ? raw.authors : [],
    projectId: raw.projectId || "main",
    status: Object.values(STATUS).includes(raw.status) ? raw.status : STATUS.QUEUED,
    progress: Number.isFinite(raw.progress) ? Math.max(0, Math.min(100, raw.progress)) : 0,
    tags: Array.isArray(raw.tags) ? raw.tags : [],
    rating: [1, 2, 3, 4, 5].includes(raw.rating) ? raw.rating : null,
    starred: Boolean(raw.starred),
    notes: raw.notes || "",
    arxivUrl: raw.arxivUrl || raw.arxiv_url || "",
    pdfUrl: raw.pdfUrl || raw.pdf_url || "",
    categories: Array.isArray(raw.categories) ? raw.categories : [],
    published: raw.published || "",
    createdAt: raw.createdAt || t,
    updatedAt: raw.updatedAt || t,
    finishedAt: raw.finishedAt || null,
  };
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState();
    const parsed = JSON.parse(raw);
    const state = {
      version: 1,
      projects: Array.isArray(parsed.projects) && parsed.projects.length ? parsed.projects : [{ id: "main", name: "Main" }],
      papers: Array.isArray(parsed.papers) ? parsed.papers.map(normalizePaper) : [],
    };
    if (!state.projects.some((p) => p.id === "main")) {
      state.projects.unshift({ id: "main", name: "Main" });
    }
    return state;
  } catch {
    return defaultState();
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function loadFilters() {
  try {
    return { ...defaultFilters(), ...(JSON.parse(localStorage.getItem(FILTERS_KEY) || "{}")) };
  } catch {
    return defaultFilters();
  }
}

function saveFilters() {
  localStorage.setItem(FILTERS_KEY, JSON.stringify(filters));
}

let state = loadState();
let filters = loadFilters();

const el = {
  arxivInput: document.getElementById("arxivInput"),
  projectForAdd: document.getElementById("projectForAdd"),
  fetchBtn: document.getElementById("fetchBtn"),
  addStatus: document.getElementById("addStatus"),
  searchInput: document.getElementById("searchInput"),
  projectFilter: document.getElementById("projectFilter"),
  statusFilter: document.getElementById("statusFilter"),
  includeArchived: document.getElementById("includeArchived"),
  newProjectName: document.getElementById("newProjectName"),
  createProjectBtn: document.getElementById("createProjectBtn"),
  exportBtn: document.getElementById("exportBtn"),
  importInput: document.getElementById("importInput"),
  paperGrid: document.getElementById("paperGrid"),
  emptyState: document.getElementById("emptyState"),
  cardTpl: document.getElementById("paperCardTpl"),
};

function projectName(id) {
  return state.projects.find((p) => p.id === id)?.name || "Unknown";
}

function setStatus(msg, isError = false) {
  el.addStatus.textContent = msg;
  el.addStatus.style.color = isError ? "#c44" : "";
}

function refreshProjectSelects() {
  const options = state.projects.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  el.projectForAdd.innerHTML = options;

  const filterOpts = [`<option value="all">All projects</option>`, ...state.projects.map((p) => `<option value="${p.id}">${p.name}</option>`)].join("");
  el.projectFilter.innerHTML = filterOpts;
  el.projectFilter.value = filters.project;
}

function saveAndRender() {
  saveState();
  render();
}

async function addArxivPaper() {
  const query = el.arxivInput.value.trim();
  if (!query) {
    setStatus("Enter an arXiv ID or URL.", true);
    return;
  }

  setStatus("Fetching metadata...");
  try {
    const resp = await fetch(`/api/arxiv?query=${encodeURIComponent(query)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Failed to fetch metadata.");

    const existing = state.papers.find((p) => p.arxivId.toLowerCase() === data.arxiv_id.toLowerCase());
    if (existing) {
      existing.title = data.title;
      existing.abstract = data.abstract;
      existing.authors = data.authors;
      existing.categories = data.categories;
      existing.published = data.published;
      existing.arxivUrl = data.arxiv_url;
      existing.pdfUrl = data.pdf_url;
      existing.updatedAt = nowISO();
      existing.projectId = el.projectForAdd.value || existing.projectId;
      setStatus(`Updated existing paper ${data.arxiv_id}.`);
      saveAndRender();
      return;
    }

    const paper = normalizePaper({
      id: uid("paper"),
      arxivId: data.arxiv_id,
      title: data.title,
      abstract: data.abstract,
      authors: data.authors,
      categories: data.categories,
      published: data.published,
      arxivUrl: data.arxiv_url,
      pdfUrl: data.pdf_url,
      projectId: el.projectForAdd.value || "main",
      status: STATUS.QUEUED,
    });

    state.papers.unshift(paper);
    setStatus(`Added ${paper.arxivId}.`);
    el.arxivInput.value = "";
    saveAndRender();
  } catch (err) {
    setStatus(err.message, true);
  }
}

function matchesSearch(paper) {
  const q = filters.search.trim().toLowerCase();
  if (!q) return true;
  const bag = [paper.title, paper.abstract, ...paper.authors, ...paper.tags, paper.notes, ...paper.categories].join(" ").toLowerCase();
  return bag.includes(q);
}

function visiblePapers() {
  let out = [...state.papers];

  if (filters.project !== "all") {
    out = out.filter((p) => p.projectId === filters.project);
  }

  if (filters.status === "queue") {
    out = out.filter((p) => p.status === STATUS.QUEUED || p.status === STATUS.READING);
  } else if (filters.status === "reading") {
    out = out.filter((p) => p.status === STATUS.READING);
  } else if (filters.status === "finished") {
    out = out.filter((p) => p.status === STATUS.FINISHED);
  } else if (filters.status === "starred") {
    out = out.filter((p) => p.starred);
  }

  if (!filters.includeArchived && !filters.search.trim()) {
    out = out.filter((p) => p.status !== STATUS.ARCHIVED && p.status !== STATUS.FINISHED);
  }

  out = out.filter(matchesSearch);

  if (!filters.includeArchived && filters.search.trim()) {
    out = out.filter((p) => p.status !== STATUS.ARCHIVED && p.status !== STATUS.FINISHED);
  }

  out.sort((a, b) => {
    const sA = Number(a.starred);
    const sB = Number(b.starred);
    if (sA !== sB) return sB - sA;
    return (b.updatedAt || "").localeCompare(a.updatedAt || "");
  });

  return out;
}

function setPaperStatus(paper, status) {
  paper.status = status;
  if (status === STATUS.FINISHED) {
    paper.finishedAt = nowISO();
    paper.progress = 100;
  }
  paper.updatedAt = nowISO();
}

function renderCard(paper) {
  const node = el.cardTpl.content.firstElementChild.cloneNode(true);
  node.querySelector(".title").textContent = paper.title;
  node.querySelector(".meta").textContent = `${paper.arxivId || "manual"} • ${paper.authors.slice(0, 4).join(", ")} • ${projectName(paper.projectId)}`;
  node.querySelector(".abstract").textContent = paper.abstract.length > 320 ? `${paper.abstract.slice(0, 320)}…` : paper.abstract;

  const starBtn = node.querySelector(".starBtn");
  starBtn.textContent = paper.starred ? "★" : "☆";
  starBtn.onclick = () => {
    paper.starred = !paper.starred;
    paper.updatedAt = nowISO();
    saveAndRender();
  };

  const badges = node.querySelector(".badges");
  const badgeValues = [paper.status, ...paper.tags, ...(paper.categories || []).slice(0, 3)].filter(Boolean);
  badges.innerHTML = badgeValues.map((b) => `<span class="badge">${b}</span>`).join("");

  node.querySelector(".startBtn").onclick = () => {
    setPaperStatus(paper, STATUS.READING);
    if (paper.progress < 5) paper.progress = 5;
    saveAndRender();
  };
  node.querySelector(".finishBtn").onclick = () => {
    setPaperStatus(paper, STATUS.FINISHED);
    saveAndRender();
  };
  node.querySelector(".archiveBtn").onclick = () => {
    setPaperStatus(paper, STATUS.ARCHIVED);
    saveAndRender();
  };
  node.querySelector(".deleteBtn").onclick = () => {
    state.papers = state.papers.filter((p) => p.id !== paper.id);
    saveAndRender();
  };

  const progressInput = node.querySelector(".progressInput");
  const progressLabel = node.querySelector(".progressLabel");
  progressInput.value = String(paper.progress || 0);
  progressLabel.textContent = `${paper.progress || 0}%`;
  progressInput.oninput = () => {
    paper.progress = Number(progressInput.value);
    progressLabel.textContent = `${paper.progress}%`;
    if (paper.progress >= 100) {
      setPaperStatus(paper, STATUS.FINISHED);
    } else if (paper.progress > 0 && paper.status === STATUS.QUEUED) {
      setPaperStatus(paper, STATUS.READING);
    }
    saveAndRender();
  };

  const ratingSelect = node.querySelector(".ratingSelect");
  ratingSelect.value = paper.rating || "";
  ratingSelect.onchange = () => {
    paper.rating = ratingSelect.value ? Number(ratingSelect.value) : null;
    paper.updatedAt = nowISO();
    saveAndRender();
  };

  const projectSelect = node.querySelector(".projectSelect");
  projectSelect.innerHTML = state.projects.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  projectSelect.value = paper.projectId;
  projectSelect.onchange = () => {
    paper.projectId = projectSelect.value;
    paper.updatedAt = nowISO();
    saveAndRender();
  };

  const tagsInput = node.querySelector(".tagsInput");
  tagsInput.value = paper.tags.join(", ");
  tagsInput.onchange = () => {
    paper.tags = tagsInput.value.split(",").map((t) => t.trim()).filter(Boolean);
    paper.updatedAt = nowISO();
    saveAndRender();
  };

  const notesInput = node.querySelector(".notesInput");
  notesInput.value = paper.notes;
  notesInput.onchange = () => {
    paper.notes = notesInput.value;
    paper.updatedAt = nowISO();
    saveAndRender();
  };

  node.querySelector(".links").innerHTML = [
    paper.arxivUrl ? `<a href="${paper.arxivUrl}" target="_blank" rel="noreferrer">ArXiv</a>` : "",
    paper.pdfUrl ? `<a href="${paper.pdfUrl}" target="_blank" rel="noreferrer">PDF</a>` : "",
  ].filter(Boolean).join(" · ");

  return node;
}

function render() {
  refreshProjectSelects();

  el.searchInput.value = filters.search;
  el.projectFilter.value = filters.project;
  el.statusFilter.value = filters.status;
  el.includeArchived.checked = filters.includeArchived;

  const papers = visiblePapers();
  el.paperGrid.innerHTML = "";
  papers.forEach((paper) => el.paperGrid.appendChild(renderCard(paper)));

  if (papers.length === 0) {
    el.emptyState.textContent = "No papers match the current filters.";
  } else {
    el.emptyState.textContent = `Showing ${papers.length} paper(s).`;
  }
}

function addProject() {
  const name = el.newProjectName.value.trim();
  if (!name) return;
  if (state.projects.some((p) => p.name.toLowerCase() === name.toLowerCase())) {
    setStatus("Project already exists.", true);
    return;
  }
  state.projects.push({ id: uid("project"), name });
  el.newProjectName.value = "";
  saveAndRender();
}

function exportData() {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `paper-queue-export-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function importData(file) {
  const text = await file.text();
  const parsed = JSON.parse(text);
  if (!parsed || !Array.isArray(parsed.papers)) {
    throw new Error("Invalid export format.");
  }

  state = {
    version: 1,
    projects: Array.isArray(parsed.projects) && parsed.projects.length ? parsed.projects : [{ id: "main", name: "Main" }],
    papers: parsed.papers.map(normalizePaper),
  };
  if (!state.projects.some((p) => p.id === "main")) {
    state.projects.unshift({ id: "main", name: "Main" });
  }

  saveAndRender();
  setStatus(`Imported ${state.papers.length} paper(s).`);
}

el.fetchBtn.addEventListener("click", addArxivPaper);
el.createProjectBtn.addEventListener("click", addProject);
el.exportBtn.addEventListener("click", exportData);
el.importInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    await importData(file);
  } catch (err) {
    setStatus(`Import failed: ${err.message}`, true);
  } finally {
    e.target.value = "";
  }
});

el.searchInput.addEventListener("input", () => {
  filters.search = el.searchInput.value;
  saveFilters();
  render();
});
el.projectFilter.addEventListener("change", () => {
  filters.project = el.projectFilter.value;
  saveFilters();
  render();
});
el.statusFilter.addEventListener("change", () => {
  filters.status = el.statusFilter.value;
  saveFilters();
  render();
});
el.includeArchived.addEventListener("change", () => {
  filters.includeArchived = el.includeArchived.checked;
  saveFilters();
  render();
});

document.querySelector("label.buttonlike").addEventListener("click", () => el.importInput.click());

render();
