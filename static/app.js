const analyzeButton = document.getElementById("analyzeButton");
const compareToggle = document.getElementById("compareToggle");
const statusLine = document.getElementById("statusLine");
const resultsRoot = document.getElementById("resultsRoot");
const scopeCard2 = document.getElementById("scopeCard2");

const SIGNAL_LABELS = ["Concern", "Purchase", "Avoidance"];
const STORY_DIRECTIONS = ["emerging", "rising", "fading", "stable"];

const state = {
  compareMode: false,
  results: [null, null],
  activeStoryIds: [null, null],
  charts: [null, null],
};

function updateStatus(message) {
  statusLine.textContent = message;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function getScope(index) {
  const suffix = index + 1;
  return {
    topic: document.getElementById(`scopeTopic${suffix}`).value.trim(),
    country: document.getElementById(`scopeCountry${suffix}`).value.trim(),
    source: document.getElementById(`scopeSource${suffix}`).value.trim(),
    language: document.getElementById(`scopeLanguage${suffix}`).value.trim(),
  };
}

function isScopeValid(scope) {
  return Boolean(scope.topic);
}

function toQuery(scope, extra = {}) {
  const params = new URLSearchParams();
  Object.entries({ ...scope, ...extra }).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      params.set(key, String(value).trim());
    }
  });
  return params.toString();
}

function scopeLabel(scope) {
  const pieces = [scope.topic || "untitled scope"];
  if (scope.country) {
    pieces.push(scope.country.toUpperCase());
  }
  if (scope.source) {
    pieces.push(scope.source);
  }
  if (scope.language) {
    pieces.push(scope.language);
  }
  return pieces.join(" / ");
}

async function runScope(scope, index, total) {
  updateStatus(`Running ${index + 1}/${total}: collecting, embedding, clustering, and bundling ${scopeLabel(scope)}...`);
  await fetchJson(`/pipeline/run?${toQuery(scope)}`, { method: "POST" });
  updateStatus(`Loading bundled output for ${scopeLabel(scope)}...`);
  return fetchJson(`/output/${encodeURIComponent(scope.topic)}?${toQuery(scope)}`);
}

function destroyChart(index) {
  if (state.charts[index]) {
    state.charts[index].destroy();
    state.charts[index] = null;
  }
}

function statusNote(status) {
  if (status === "ready") {
    return "Ready: cluster drift and brief confidence are high enough for normal demo use.";
  }
  if (status === "warming") {
    return "Warming: interpret with care while coverage and baselines stabilize.";
  }
  return "Insufficient data: this is a data-availability notice, not a stable drift finding yet.";
}

function strongestSegment(data) {
  return data?.pipeline?.strongest_segment || data?.brief?.most_affected_segment || "n/a";
}

function storyGroups(data) {
  const items = data?.cluster_drift?.data?.clusters || [];
  const groups = { emerging: [], rising: [], fading: [], stable: [] };
  items.forEach((item) => {
    const direction = STORY_DIRECTIONS.includes(item.direction) ? item.direction : "stable";
    groups[direction].push(item);
  });
  return groups;
}

function defaultStoryId(data) {
  const cited = data?.brief_support?.cited_track_ids || [];
  if (cited.length > 0) {
    return cited[0];
  }
  const firstStory = data?.cluster_drift?.data?.clusters?.[0];
  return firstStory ? firstStory.track_id : null;
}

function findStory(data, trackId) {
  return (data?.cluster_drift?.data?.clusters || []).find((item) => item.track_id === trackId) || null;
}

function signalRows(data) {
  return Array.isArray(data?.signals) ? data.signals : [];
}

function formatMetaPill(text, className = "") {
  return `<span class="pill ${className}">${escapeHtml(text)}</span>`;
}

function metricCard(label, value, note) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="panel-scope-note">${escapeHtml(note)}</div>
    </div>
  `;
}

function briefCitations(data, slot) {
  const citedTracks = data?.brief_support?.cited_track_ids || [];
  const citedArticles = data?.brief_support?.cited_article_ids || [];
  const activeStoryId = state.activeStoryIds[slot];
  const trackChips = citedTracks.map((trackId) => `
    <button type="button" class="citation-chip ${trackId === activeStoryId ? "is-active" : ""}" data-slot="${slot}" data-track-id="${escapeHtml(trackId)}">
      Story ${escapeHtml(trackId)}
    </button>
  `).join("");
  const articleChips = citedArticles.map((articleId) => `
    <span class="article-chip">Article ${escapeHtml(articleId)}</span>
  `).join("");
  if (!trackChips && !articleChips) {
    return "";
  }
  return `
    <div class="brief-label">Brief Support</div>
    <div class="citation-row">${trackChips}${articleChips}</div>
  `;
}

function renderStoryBoard(data, slot) {
  const groups = storyGroups(data);
  const cited = new Set(data?.brief_support?.cited_track_ids || []);
  return `
    <div class="story-board" id="storyline-board-${slot}">
      <div class="stack-title">Storyline Board</div>
      <div class="story-columns">
        ${STORY_DIRECTIONS.map((direction) => `
          <div class="story-column">
            <div class="story-column-title">${escapeHtml(direction)}</div>
            ${(groups[direction] || []).map((item) => `
              <button
                type="button"
                class="story-card ${cited.has(item.track_id) ? "is-cited" : ""} ${state.activeStoryIds[slot] === item.track_id ? "is-active" : ""}"
                data-slot="${slot}"
                data-track-id="${escapeHtml(item.track_id)}"
              >
                <div class="story-card-header">
                  <div class="story-title">${escapeHtml(item.topic_label || "Unnamed storyline")}</div>
                  ${formatMetaPill(item.alert_level || "none", item.alert_level || "")}
                </div>
                <div class="story-meta">
                  ${formatMetaPill(item.direction || "stable", item.direction || "stable")}
                  ${formatMetaPill(`${item.member_count || 0} articles`)}
                  ${formatMetaPill(`conf ${Number(item.confidence || 0).toFixed(2)}`)}
                </div>
                <div class="panel-scope-note">
                  Drift ${Number(item.drift_magnitude || 0).toFixed(2)} | Frame ${escapeHtml(item.dominant_frame || "n/a")}
                </div>
              </button>
            `).join("") || `<div class="cluster-status"><p>No ${escapeHtml(direction)} storylines in this scope.</p></div>`}
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderEvidencePanel(data, slot) {
  const story = findStory(data, state.activeStoryIds[slot]);
  if (!story) {
    return `
      <div class="evidence-card">
        <div class="stack-title">Evidence Drawer</div>
        <p class="evidence-body">Select a storyline or brief citation to inspect grounded evidence.</p>
      </div>
    `;
  }
  const evidenceItems = (story.evidence_json || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const articles = (story.exemplar_articles || []).map((article) => `
    <div class="signal-card">
      <div class="evidence-title">${escapeHtml(article.title || "[no title]")}</div>
      <div class="evidence-meta">
        ${formatMetaPill(article.outlet || "unknown")}
        ${formatMetaPill(article.country || "n/a")}
      </div>
      <p class="evidence-body">${escapeHtml(article.summary || article.body_excerpt || "")}</p>
      ${article.url ? `<a href="${escapeHtml(article.url)}" target="_blank" rel="noreferrer">Open article</a>` : ""}
    </div>
  `).join("");
  return `
    <div class="evidence-card">
      <div class="stack-title">Evidence Drawer</div>
      <h3>${escapeHtml(story.topic_label || "Unnamed storyline")}</h3>
      <div class="evidence-meta">
        ${formatMetaPill(story.track_id || "no-track")}
        ${formatMetaPill(story.match_type || "unknown")}
        ${formatMetaPill(`baseline ${story.baseline_member_count || 0}`)}
      </div>
      ${evidenceItems ? `<ul class="bullet-list">${evidenceItems}</ul>` : `<p class="evidence-body">No extracted evidence bullets were stored for this storyline.</p>`}
      <div class="evidence-grid">
        <div class="stack-title">Exemplar Articles</div>
        <div class="evidence-articles">${articles || `<p class="evidence-body">No exemplar articles were bundled for this storyline.</p>`}</div>
      </div>
    </div>
  `;
}

function renderDigest(data) {
  const digest = data.digest || {};
  return `
    <div class="digest-card">
      <div class="stack-title">Digest</div>
      <h3 class="digest-headline">${escapeHtml(digest.summary_headline || "No digest available")}</h3>
      <p class="digest-text">${escapeHtml(digest.summary_text || "Digest output has not been generated yet.")}</p>
      <ul class="bullet-list">
        ${(digest.key_points || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderBrief(data, slot) {
  const brief = data.brief || {};
  const supportNote = data.brief_support?.fallback_note
    ? `<p class="digest-text">${escapeHtml(data.brief_support.fallback_note)}</p>`
    : "";
  return `
    <div class="brief-card">
      <div id="brief-status-banner-${slot}" class="brief-status-banner ${escapeHtml(brief.status || "insufficient_data")}">${escapeHtml(brief.status || "insufficient_data")}</div>
      <p class="digest-text">${escapeHtml(statusNote(brief.status || "insufficient_data"))}</p>
      <h3 class="brief-headline ${brief.status === "insufficient_data" ? "is-muted" : ""}">${escapeHtml(brief.headline || "Brief unavailable")}</h3>
      <p class="brief-text">${escapeHtml(brief.narrative || "No brief narrative is available.")}</p>
      <div class="brief-meta">
        ${formatMetaPill(`segment ${brief.most_affected_segment || "n/a"}`)}
        ${formatMetaPill(`alert ${brief.alert_level || "none"}`)}
        ${formatMetaPill(`mode ${brief.generation_mode || "n/a"}`)}
      </div>
      ${briefCitations(data, slot)}
      ${supportNote}
    </div>
  `;
}

function renderClusterStatus(data) {
  const clusterStatus = data?.pipeline?.cluster_status || "not_run";
  if (!["skipped_small_sample", "all_noise", "not_run"].includes(clusterStatus) && data?.cluster_drift?.status === "ready") {
    return "";
  }
  return `
    <div class="cluster-status">
      <div class="stack-title">Cluster Status</div>
      <p>Stable story groups are not fully available yet. Digest and brief remain usable while event clustering warms up.</p>
    </div>
  `;
}

function renderSignals(data) {
  const rows = signalRows(data);
  return `
    <div class="signals-card">
      <div class="stack-title">Raw Signal Audit</div>
      <table class="signal-table">
        <thead>
          <tr>
            <th>Article</th>
            <th>Source</th>
            <th>Domain</th>
            <th>Concern</th>
            <th>Purchase</th>
            <th>Avoidance</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((item) => `
            <tr>
              <td>${escapeHtml(item.title || item.article_id)}</td>
              <td>${escapeHtml(item.outlet || "")}</td>
              <td>${escapeHtml(item.domain || "generic")}</td>
              <td>${Number(item.concern_level || 0).toFixed(2)}</td>
              <td>${Number(item.purchase_intent || 0).toFixed(2)}</td>
              <td>${Number(item.avoidance_signals || 0).toFixed(2)}</td>
            </tr>
          `).join("") || `<tr><td colspan="6">No filtered signals were bundled for this scope.</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
}

function panelMarkup(data, slot) {
  const scope = data.scope || {};
  const pipeline = data.pipeline || {};
  return `
    <section class="panel" data-slot="${slot}">
      <div class="panel-header">
        <div>
          <div class="section-title">Scoped Output</div>
          <h2 class="panel-scope">${escapeHtml(scopeLabel(scope))}</h2>
          <p class="panel-scope-note">Run ID: ${escapeHtml(data.run_id || "not available")} | Cluster status: ${escapeHtml(pipeline.cluster_status || "not_run")}</p>
        </div>
      </div>

      <div class="summary-rail">
        ${metricCard("Articles", pipeline.article_count || 0, "Scoped article coverage")}
        ${metricCard("Clusters", pipeline.cluster_count || 0, "Stable event groups")}
        ${metricCard("Strongest Segment", strongestSegment(data), "Segment-relevant coverage with the clearest movement")}
        ${metricCard("Brief Status", data.brief?.status || "insufficient_data", "Trust state for the current brief")}
      </div>

      <div class="stack-grid">
        <div class="stack">
          ${renderDigest(data)}
          ${renderBrief(data, slot)}
          ${renderClusterStatus(data)}
        </div>
        <div class="stack">
          <div class="chart-card">
            <div class="stack-title">Segment Radar</div>
            <div class="chart-toolbar">
              <label class="brief-label" for="segmentSelect${slot}">Segment</label>
              <select id="segmentSelect${slot}" data-slot="${slot}"></select>
            </div>
            <div class="chart-wrap">
              <canvas id="segmentChart${slot}"></canvas>
            </div>
          </div>
          ${renderEvidencePanel(data, slot)}
        </div>
      </div>

      ${renderStoryBoard(data, slot)}
      ${renderSignals(data)}
    </section>
  `;
}

function render() {
  if (!state.results[0] && !state.results[1]) {
    return;
  }

  if (state.compareMode) {
    resultsRoot.innerHTML = `
      <div class="compare-root">
        ${state.results.map((item, index) => item ? panelMarkup(item, index) : `<section class="empty-state"><div class="section-title">Scope ${index + 1}</div><h2>No output loaded</h2><p>Run analysis for this scope.</p></section>`).join("")}
      </div>
    `;
  } else if (state.results[0]) {
    resultsRoot.innerHTML = panelMarkup(state.results[0], 0);
  }

  state.results.forEach((data, index) => {
    if (!data) {
      destroyChart(index);
      return;
    }

    const select = document.getElementById(`segmentSelect${index}`);
    const canvas = document.getElementById(`segmentChart${index}`);
    const segments = data?.cluster_drift?.data?.segments || [];
    if (select && canvas) {
      const options = segments.map((segment) => `<option value="${escapeHtml(segment.segment)}">${escapeHtml(segment.segment)}</option>`).join("");
      select.innerHTML = options || `<option value="">no segment data</option>`;
      const selectedValue = segments.find((item) => item.segment === select.value) ? select.value : (segments[0]?.segment || "");
      select.value = selectedValue;
      const renderChart = () => {
        const active = segments.find((item) => item.segment === select.value) || segments[0] || null;
        destroyChart(index);
        if (!active) {
          return;
        }
        state.charts[index] = new Chart(canvas.getContext("2d"), {
          type: "radar",
          data: {
            labels: SIGNAL_LABELS,
            datasets: [
              {
                label: "Current",
                data: [
                  Number(active.current?.concern_level || 0),
                  Number(active.current?.purchase_intent || 0),
                  Number(active.current?.avoidance_signals || 0),
                ],
                borderColor: "#1d7a53",
                backgroundColor: "rgba(29, 122, 83, 0.16)",
              },
              {
                label: "Baseline",
                data: [
                  Number(active.baseline?.concern_level || 0),
                  Number(active.baseline?.purchase_intent || 0),
                  Number(active.baseline?.avoidance_signals || 0),
                ],
                borderColor: "#2d5e97",
                backgroundColor: "rgba(45, 94, 151, 0.10)",
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              r: {
                suggestedMin: 0,
                suggestedMax: 1,
                pointLabels: {
                  font: { family: "IBM Plex Mono", size: 11 },
                },
              },
            },
            plugins: {
              legend: {
                labels: {
                  font: { family: "IBM Plex Mono", size: 11 },
                },
              },
            },
          },
        });
      };
      select.onchange = renderChart;
      renderChart();
    }
  });

  document.querySelectorAll("[data-track-id]").forEach((element) => {
    element.addEventListener("click", () => {
      const slot = Number(element.getAttribute("data-slot") || "0");
      const trackId = element.getAttribute("data-track-id");
      state.activeStoryIds[slot] = trackId;
      render();
    });
  });
}

async function runAnalysis() {
  const scopes = [getScope(0)];
  if (state.compareMode) {
    scopes.push(getScope(1));
  }

  const invalidIndex = scopes.findIndex((scope) => !isScopeValid(scope));
  if (invalidIndex >= 0) {
    updateStatus(`Enter a topic for scope ${invalidIndex + 1}.`);
    return;
  }

  analyzeButton.disabled = true;
  compareToggle.disabled = true;

  try {
    const outputs = [];
    for (let index = 0; index < scopes.length; index += 1) {
      outputs[index] = await runScope(scopes[index], index, scopes.length);
      state.activeStoryIds[index] = defaultStoryId(outputs[index]);
      state.results[index] = outputs[index];
    }
    if (!state.compareMode) {
      state.results[1] = null;
      state.activeStoryIds[1] = null;
      destroyChart(1);
    }
    render();
    updateStatus("Stage 7 bundle loaded.");
  } catch (error) {
    updateStatus(`Error: ${error.message}`);
  } finally {
    analyzeButton.disabled = false;
    compareToggle.disabled = false;
  }
}

function syncCompareState() {
  compareToggle.classList.toggle("is-active", state.compareMode);
  compareToggle.setAttribute("aria-pressed", String(state.compareMode));
  scopeCard2.classList.toggle("hidden", !state.compareMode);
  if (!state.compareMode) {
    state.results[1] = null;
    state.activeStoryIds[1] = null;
    destroyChart(1);
  }
  render();
}

compareToggle.addEventListener("click", () => {
  state.compareMode = !state.compareMode;
  syncCompareState();
});

analyzeButton.addEventListener("click", runAnalysis);
