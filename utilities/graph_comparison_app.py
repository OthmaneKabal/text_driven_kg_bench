"""Local web interface for comparing TDG-Bench graph characteristics.

Run from the repository root:
    python utilities\\graph_comparison_app.py

The interface always uses ``UMLS_nci_kg`` as the clean reference and lets the
user select any other JSON graph available directly under ``datasets/``.
``ijson`` is used when installed so the large UMLS JSON is processed as a
stream rather than loaded entirely into memory.
"""

from __future__ import annotations

import json
import sys
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from flask import Flask, jsonify, render_template_string, request

try:
    import ijson
except ImportError:  # The standard-library fallback remains usable for small graphs.
    ijson = None


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIRECTORY = ROOT / "datasets"
CLEAN_GRAPH_NAME = "UMLS_nci_kg"
app = Flask(__name__)


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TDG-Bench graph comparison</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; color: #172033; background: #f4f7fb; }
    body { margin: 0; }
    main { max-width: 1500px; margin: 0 auto; padding: 30px; }
    h1 { margin: 0 0 8px; font-size: 1.75rem; }
    p { color: #4c596e; line-height: 1.45; }
    .panel { background: white; padding: 20px; border: 1px solid #dfe6f1; border-radius: 12px; box-shadow: 0 3px 13px #1720330a; margin-top: 20px; }
    .controls { display: flex; align-items: end; gap: 16px; flex-wrap: wrap; }
    label { display: grid; gap: 8px; font-weight: 650; }
    select { min-width: 430px; height: 190px; border: 1px solid #b9c7dc; border-radius: 8px; padding: 7px; font: inherit; }
    button { border: 0; border-radius: 8px; background: #1d4ed8; color: #fff; padding: 11px 16px; font: inherit; font-weight: 650; cursor: pointer; }
    button:disabled { background: #93a4be; cursor: wait; }
    #status { min-height: 22px; margin-top: 14px; color: #40516d; }
    #error { color: #b42318; font-weight: 600; }
    .hidden { display: none; }
    .scroll { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .91rem; }
    th, td { border-bottom: 1px solid #e6ebf3; padding: 10px 11px; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; font-weight: 650; }
    th { color: #44536a; background: #f7f9fc; }
    .reference td { background: #eff6ff; }
    .curve-scroll { overflow-x: auto; border: 1px solid #e0e7f0; border-radius: 10px; background: #fff; }
    #relation-curve { display: block; min-height: 370px; }
    .distribution-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 18px; }
    .distribution-grid h3 { margin: 0 0 8px; font-size: 1rem; }
    .degree-curve { display: block; min-height: 330px; }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 14px 0 10px; color: #45556e; font-size: .9rem; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 12px; height: 12px; border-radius: 50%; }
    .chart-tooltip { position: fixed; z-index: 10; max-width: 380px; padding: 9px 11px; border-radius: 7px; background: #172033; color: #fff; box-shadow: 0 4px 16px #17203344; font-size: .84rem; line-height: 1.45; pointer-events: none; }
    .tooltip-title { font-weight: 700; margin-bottom: 5px; overflow-wrap: anywhere; }
    .tooltip-row { display: grid; grid-template-columns: 10px minmax(0, 1fr); gap: 7px; align-items: start; margin-top: 3px; }
    .tooltip-dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 5px; }
    code { background: #eef2f7; padding: 2px 4px; border-radius: 4px; }
  </style>
</head>
<body>
<main>
  <h1>Graph comparison</h1>
  <p>Compare selected graphs against the clean reference <code>UMLS_nci_kg</code>. The reference is always included. Select several graphs with Ctrl+click or Shift+click.</p>
  <section class="panel">
    <div class="controls">
      <label>Graphs to compare
        <select id="graphs" multiple aria-label="Graphs to compare">
          {% for graph in graphs %}<option value="{{ graph }}">{{ graph }}</option>{% endfor %}
        </select>
      </label>
      <button id="compare">Compare selected graphs</button>
    </div>
    <div id="status"></div><div id="error" class="hidden"></div>
  </section>
  <section id="results" class="panel hidden">
    <h2>Structural characteristics</h2>
    <div class="scroll"><table id="metrics"></table></div>
    <h2>Relation-frequency curves</h2>
    <p>Each curve shows the relative frequency of every relation. The x-axis contains all relations, ordered by decreasing frequency in the clean UMLS-NCI reference; ties use alphabetical order.</p>
    <div id="relation-legend" class="legend"></div>
    <div class="curve-scroll"><svg id="relation-curve" role="img" aria-label="Comparison of relation-frequency distributions"></svg></div>
    <h2>Directed-degree distributions</h2>
    <p>Y is the percentage of nodes. Degree intervals use a logarithmic grouping: <code>0</code>, <code>1</code>, <code>2–3</code>, <code>4–7</code>, and so on.</p>
    <div class="distribution-grid">
      <section><h3>In-degree distribution</h3><div class="curve-scroll"><svg id="in-degree-curve" class="degree-curve" role="img" aria-label="Comparison of in-degree distributions"></svg></div></section>
      <section><h3>Out-degree distribution</h3><div class="curve-scroll"><svg id="out-degree-curve" class="degree-curve" role="img" aria-label="Comparison of out-degree distributions"></svg></div></section>
    </div>
  </section>
</main>
<div id="chart-tooltip" class="chart-tooltip hidden"></div>
<script>
const select = document.getElementById('graphs');
const button = document.getElementById('compare');
const status = document.getElementById('status');
const error = document.getElementById('error');
const results = document.getElementById('results');
const metricHeaders = [
  ['graph_name', 'Graph'], ['nodes', 'Nodes'], ['edges', 'Edges'],
  ['mean_undirected_degree', 'Mean degree'], ['mean_in_degree', 'Mean in-degree'], ['mean_out_degree', 'Mean out-degree'],
  ['weakly_connected_components', 'Components'],
  ['largest_component_fraction', 'Largest component'], ['unique_relations', 'Unique relations'],
  ['unique_semantic_types', 'Semantic types']
];
function number(value, digits=0) { return new Intl.NumberFormat().format(Number(value).toFixed(digits)); }
function cell(metric, value) {
  if (metric === 'mean_undirected_degree' || metric === 'mean_in_degree' || metric === 'mean_out_degree') return Number(value).toFixed(4);
  if (metric === 'largest_component_fraction') return (100 * Number(value)).toFixed(2) + '%';
  return number(value);
}
function render(data) {
  const table = document.getElementById('metrics');
  table.innerHTML = '<thead><tr>' + metricHeaders.map(([, label]) => `<th>${label}</th>`).join('') + '</tr></thead>' +
    '<tbody>' + data.graphs.map((row, index) => '<tr class="' + (index === 0 ? 'reference' : '') + '">' +
      metricHeaders.map(([key]) => `<td>${key === 'graph_name' ? row[key] : cell(key, row[key])}</td>`).join('') +
      '</tr>').join('') + '</tbody>';
  renderRelationCurves(data.graphs);
  renderDegreeDistribution(data.graphs, 'in_degree_distribution', 'in-degree-curve', 'In-degree');
  renderDegreeDistribution(data.graphs, 'out_degree_distribution', 'out-degree-curve', 'Out-degree');
  results.classList.remove('hidden');
}
function escapeXml(value) { return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char])); }
function renderRelationCurves(graphs) {
  const colors = ['#1d4ed8', '#dc2626', '#059669', '#9333ea', '#ea580c', '#0891b2', '#4f46e5'];
  const referenceDistribution = graphs[0].relation_distribution;
  const relations = [...new Set(graphs.flatMap(graph => Object.keys(graph.relation_distribution)))].sort((a, b) =>
    (referenceDistribution[b] || 0) - (referenceDistribution[a] || 0) || a.localeCompare(b));
  const svg = document.getElementById('relation-curve');
  const legend = document.getElementById('relation-legend');
  if (!relations.length) { svg.innerHTML = ''; legend.innerHTML = ''; return; }
  // One position and one visible label per relation.  The SVG can be scrolled
  // horizontally instead of omitting low-frequency predicates.
  const width = Math.max(950, relations.length * 75 + 95), height = 370;
  const left = 62, right = 18, top = 24, bottom = 92;
  const chartWidth = width - left - right, chartHeight = height - top - bottom;
  const maxValue = Math.max(...graphs.flatMap(graph => Object.values(graph.relation_distribution)), 0.01);
  const percentageDigits = maxValue * 100 < 1 ? 3 : (maxValue * 100 < 10 ? 2 : 1);
  const x = index => left + (relations.length === 1 ? chartWidth / 2 : index * chartWidth / (relations.length - 1));
  const y = value => top + chartHeight - value / maxValue * chartHeight;
  let markup = `<line x1="${left}" y1="${top}" x2="${left}" y2="${top + chartHeight}" stroke="#8190a5"/><line x1="${left}" y1="${top + chartHeight}" x2="${width-right}" y2="${top + chartHeight}" stroke="#8190a5"/>`;
  markup += `<text transform="translate(15,${top + chartHeight / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#45556e">Relative frequency (%)</text>`;
  for (let tick = 0; tick <= 4; tick++) {
    const value = maxValue * tick / 4, lineY = y(value);
    markup += `<line x1="${left}" y1="${lineY}" x2="${width-right}" y2="${lineY}" stroke="#e7edf5"/><text x="${left-8}" y="${lineY+4}" text-anchor="end" font-size="11" fill="#596a82">${(100*value).toFixed(percentageDigits)}%</text>`;
  }
  relations.forEach((relation, index) => {
    markup += `<text transform="translate(${x(index)+3},${height-bottom+13}) rotate(55)" font-size="10" fill="#596a82">${escapeXml(relation)}</text>`;
  });
  graphs.forEach((graph, graphIndex) => {
    const color = colors[graphIndex % colors.length];
    const points = relations.map((relation, index) => `${x(index)},${y(graph.relation_distribution[relation] || 0)}`).join(' ');
    markup += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"/>`;
    relations.forEach((relation, index) => {
      const value = graph.relation_distribution[relation] || 0;
      markup += `<circle cx="${x(index)}" cy="${y(value)}" r="2.7" fill="${color}" data-relation="${escapeXml(relation)}"></circle>`;
    });
  });
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('width', width); svg.setAttribute('height', height); svg.innerHTML = markup;
  enableGroupedTooltips(svg, point => {
    const relation = point.dataset.relation;
    return {
      title: relation,
      lines: graphs.map((graph, index) => {
        const value = graph.relation_distribution[relation] || 0;
        const count = graph.relation_counts[relation] || 0;
        return {color: colors[index % colors.length], text: `${graph.graph_name}: ${number(count)} / ${number(graph.edges)} edges (${(100 * value).toFixed(6)}%)`};
      }),
    };
  });
  legend.innerHTML = graphs.map((graph, index) => `<span class="legend-item"><span class="swatch" style="background:${colors[index % colors.length]}"></span>${escapeXml(graph.graph_name)}</span>`).join('');
}
function enableGroupedTooltips(svg, buildContent) {
  const tooltip = document.getElementById('chart-tooltip');
  const position = event => { tooltip.style.left = `${event.clientX + 14}px`; tooltip.style.top = `${event.clientY + 14}px`; };
  const show = (event, point) => {
    const content = buildContent(point);
    tooltip.innerHTML = `<div class="tooltip-title">${escapeXml(content.title)}</div>` + content.lines.map(line =>
      `<div class="tooltip-row"><span class="tooltip-dot" style="background:${line.color}"></span><span>${escapeXml(line.text)}</span></div>`).join('');
    tooltip.classList.remove('hidden'); position(event);
  };
  svg.querySelectorAll('circle').forEach(point => {
    point.addEventListener('mouseenter', event => show(event, point));
    point.addEventListener('mousemove', position);
    point.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));
  });
}
function renderDegreeDistribution(graphs, distributionKey, svgId, axisLabel) {
  const colors = ['#1d4ed8', '#dc2626', '#059669', '#9333ea', '#ea580c', '#0891b2', '#4f46e5'];
  const binsByLowerBound = new Map();
  graphs.forEach(graph => graph[distributionKey].forEach(bin => binsByLowerBound.set(bin.lower, bin.label)));
  const bins = [...binsByLowerBound.entries()].sort((left, right) => left[0] - right[0]);
  const svg = document.getElementById(svgId);
  if (!bins.length) { svg.innerHTML = ''; return; }
  const width = Math.max(560, bins.length * 78 + 95), height = 330;
  const left = 58, right = 16, top = 24, bottom = 58;
  const chartWidth = width - left - right, chartHeight = height - top - bottom;
  const valueByGraph = graph => new Map(graph[distributionKey].map(bin => [bin.lower, bin.share]));
  const graphValues = graphs.map(valueByGraph);
  const maxValue = Math.max(...graphValues.flatMap(values => [...values.values()]), 0.01);
  const percentageDigits = maxValue * 100 < 1 ? 3 : (maxValue * 100 < 10 ? 2 : 1);
  const x = index => left + (bins.length === 1 ? chartWidth / 2 : index * chartWidth / (bins.length - 1));
  const y = value => top + chartHeight - value / maxValue * chartHeight;
  let markup = `<line x1="${left}" y1="${top}" x2="${left}" y2="${top + chartHeight}" stroke="#8190a5"/><line x1="${left}" y1="${top + chartHeight}" x2="${width-right}" y2="${top + chartHeight}" stroke="#8190a5"/>`;
  for (let tick = 0; tick <= 4; tick++) {
    const value = maxValue * tick / 4, lineY = y(value);
    markup += `<line x1="${left}" y1="${lineY}" x2="${width-right}" y2="${lineY}" stroke="#e7edf5"/><text x="${left-7}" y="${lineY+4}" text-anchor="end" font-size="11" fill="#596a82">${(100 * value).toFixed(percentageDigits)}%</text>`;
  }
  markup += `<text transform="translate(14,${top + chartHeight / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#45556e">Nodes (%)</text>`;
  bins.forEach(([lower, label], index) => {
    markup += `<text x="${x(index)}" y="${height-bottom+18}" text-anchor="middle" font-size="11" fill="#596a82">${escapeXml(label)}</text>`;
  });
  markup += `<text x="${left + chartWidth / 2}" y="${height-8}" text-anchor="middle" font-size="12" fill="#45556e">${axisLabel} interval</text>`;
  graphs.forEach((graph, graphIndex) => {
    const values = graphValues[graphIndex], color = colors[graphIndex % colors.length];
    const points = bins.map(([lower], index) => `${x(index)},${y(values.get(lower) || 0)}`).join(' ');
    markup += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"/>`;
    bins.forEach(([lower, label], index) => {
      const value = values.get(lower) || 0;
      markup += `<circle cx="${x(index)}" cy="${y(value)}" r="3" fill="${color}" data-lower="${lower}"></circle>`;
    });
  });
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('width', width); svg.setAttribute('height', height); svg.innerHTML = markup;
  enableGroupedTooltips(svg, point => {
    const lower = Number(point.dataset.lower), label = binsByLowerBound.get(lower);
    return {
      title: `${axisLabel}: ${label}`,
      lines: graphs.map((graph, index) => {
        const bin = graph[distributionKey].find(candidate => candidate.lower === lower);
        return {color: colors[index % colors.length], text: `${graph.graph_name}: ${number(bin ? bin.count : 0)} nodes (${(100 * (bin ? bin.share : 0)).toFixed(6)}%)`};
      }),
    };
  });
}
button.addEventListener('click', async () => {
  const graphNames = [...select.selectedOptions].map(option => option.value);
  button.disabled = true; error.classList.add('hidden');
  status.textContent = 'Computing graph characteristics. The first comparison reads the JSON files and may take time.';
  try {
    const response = await fetch('/api/compare', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({graphs: graphNames})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Comparison failed.');
    render(data); status.textContent = 'Comparison complete. Cached graphs are reused while their files remain unchanged.';
  } catch (exception) {
    error.textContent = exception.message; error.classList.remove('hidden'); status.textContent = '';
  } finally { button.disabled = false; }
});
</script>
</body>
</html>"""


def normalize_term(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node: str) -> str:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while node != root:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records without loading a large graph when ijson is installed."""
    if ijson is not None:
        with path.open("rb") as handle:
            for record in ijson.items(handle, "item"):
                if isinstance(record, dict):
                    yield record
        return
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    yield from (record for record in payload if isinstance(record, dict))


def record_types(record: dict[str, Any], endpoint: str) -> Iterable[str]:
    values = record.get(f"{endpoint}_types")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("name"), str) and value["name"].strip():
                yield value["name"]
        return
    value = record.get(f"{endpoint}_type")
    if isinstance(value, str) and value.strip():
        yield value


@dataclass
class GraphProfile:
    graph_name: str
    nodes: int
    edges: int
    mean_undirected_degree: float
    mean_in_degree: float
    mean_out_degree: float
    weakly_connected_components: int
    largest_component_fraction: float
    unique_relations: int
    unique_semantic_types: int
    in_degree_distribution: list[dict[str, int | float | str]]
    out_degree_distribution: list[dict[str, int | float | str]]
    predicates: Counter[str]
    semantic_types: Counter[str]


def degree_distribution(
    degree: Counter[str], nodes: Iterable[str]
) -> list[dict[str, int | float | str]]:
    """Return a node-normalized, logarithmically binned degree distribution."""
    node_list = list(nodes)
    counts: Counter[int] = Counter()
    for node in node_list:
        value = degree[node]
        lower = 0 if value == 0 else 1 << (value.bit_length() - 1)
        counts[lower] += 1
    total = len(node_list)
    distribution: list[dict[str, int | float | str]] = []
    for lower in sorted(counts):
        if lower == 0:
            label, upper = "0", 0
        elif lower == 1:
            label, upper = "1", 1
        else:
            upper = 2 * lower - 1
            label = f"{lower}–{upper}"
        distribution.append(
            {
                "lower": lower,
                "upper": upper,
                "label": label,
                "count": counts[lower],
                "share": counts[lower] / total if total else 0.0,
            }
        )
    return distribution


def calculate_profile(path: Path) -> GraphProfile:
    union_find = UnionFind()
    degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    predicates: Counter[str] = Counter()
    semantic_types: Counter[str] = Counter()
    edges = 0
    for record in iter_records(path):
        edges += 1
        predicates[str(record.get("predicate", ""))] += 1
        semantic_types.update(record_types(record, "subject"))
        semantic_types.update(record_types(record, "object"))
        subject, obj = record.get("subject"), record.get("object")
        if not isinstance(subject, str) or not subject.strip() or not isinstance(obj, str) or not obj.strip():
            continue
        source, target = normalize_term(subject), normalize_term(obj)
        union_find.add(source)
        union_find.add(target)
        union_find.union(source, target)
        degree[source] += 1
        degree[target] += 1
        out_degree[source] += 1
        in_degree[target] += 1
    component_sizes = Counter(union_find.find(node) for node in union_find.parent)
    nodes = len(union_find.parent)
    return GraphProfile(
        graph_name=path.stem,
        nodes=nodes,
        edges=edges,
        mean_undirected_degree=(sum(degree.values()) / nodes) if nodes else 0.0,
        mean_in_degree=(sum(in_degree.values()) / nodes) if nodes else 0.0,
        mean_out_degree=(sum(out_degree.values()) / nodes) if nodes else 0.0,
        weakly_connected_components=len(component_sizes),
        largest_component_fraction=(max(component_sizes.values()) / nodes) if nodes else 0.0,
        unique_relations=len(predicates),
        unique_semantic_types=len(semantic_types),
        in_degree_distribution=degree_distribution(in_degree, union_find.parent),
        out_degree_distribution=degree_distribution(out_degree, union_find.parent),
        predicates=predicates,
        semantic_types=semantic_types,
    )


PROFILE_CACHE: dict[Path, tuple[int, GraphProfile]] = {}
CACHE_LOCK = threading.Lock()


def cached_profile(path: Path) -> GraphProfile:
    modified_time_ns = path.stat().st_mtime_ns
    with CACHE_LOCK:
        cached = PROFILE_CACHE.get(path)
        if cached is not None and cached[0] == modified_time_ns:
            return cached[1]
    profile = calculate_profile(path)
    with CACHE_LOCK:
        PROFILE_CACHE[path] = (modified_time_ns, profile)
    return profile


def available_graphs() -> dict[str, Path]:
    return {path.stem: path for path in sorted(DATASETS_DIRECTORY.glob("*.json"))}


def profile_payload(profile: GraphProfile) -> dict[str, Any]:
    return {
        "graph_name": profile.graph_name,
        "nodes": profile.nodes,
        "edges": profile.edges,
        "mean_undirected_degree": profile.mean_undirected_degree,
        "mean_in_degree": profile.mean_in_degree,
        "mean_out_degree": profile.mean_out_degree,
        "weakly_connected_components": profile.weakly_connected_components,
        "largest_component_fraction": profile.largest_component_fraction,
        "unique_relations": profile.unique_relations,
        "unique_semantic_types": profile.unique_semantic_types,
        "in_degree_distribution": profile.in_degree_distribution,
        "out_degree_distribution": profile.out_degree_distribution,
        "relation_distribution": {
            predicate: count / profile.edges if profile.edges else 0.0
            for predicate, count in profile.predicates.items()
        },
        "relation_counts": dict(profile.predicates),
    }


@app.get("/")
def index() -> str:
    graphs = [name for name in available_graphs() if name != CLEAN_GRAPH_NAME]
    return render_template_string(PAGE, graphs=graphs)


@app.post("/api/compare")
def compare() -> tuple[Any, int] | Any:
    graph_paths = available_graphs()
    clean_path = graph_paths.get(CLEAN_GRAPH_NAME)
    if clean_path is None:
        return jsonify(error=f"Missing clean graph: {DATASETS_DIRECTORY / (CLEAN_GRAPH_NAME + '.json')}"), 404
    payload = request.get_json(silent=True) or {}
    requested = payload.get("graphs", [])
    if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
        return jsonify(error="'graphs' must be a list of graph names."), 400
    unknown = sorted(set(requested) - set(graph_paths))
    if unknown:
        return jsonify(error=f"Unknown graph name(s): {', '.join(unknown)}"), 400

    selected_names = [name for name in dict.fromkeys(requested) if name != CLEAN_GRAPH_NAME]
    profiles = [cached_profile(clean_path)] + [cached_profile(graph_paths[name]) for name in selected_names]
    return jsonify(graphs=[profile_payload(profile) for profile in profiles])


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
