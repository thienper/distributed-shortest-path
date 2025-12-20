// ===============================
// GLOBAL STATE
// ===============================
let cy = null;
let selectedSourceNode = null;
let selectedTargetNode = null;
let isPathActive = false;

// ===============================
// INIT
// ===============================
async function initCytoscape() {
    try {
        const res = await fetch('/api/graph');
        const data = await res.json();

        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: [...data.nodes, ...data.edges],
            style: getStyle(),
            autoungrabify: true,
            autounselectify: true,
            pannable: true,
            zoomable: true,
            wheelSensitivity: 0.15,
            boxSelectionEnabled: false
        });

        cy.userPanningEnabled(true);

        applyGridLayout();
        bindEvents();

        document.getElementById('graph-info').textContent =
            `${data.num_nodes} nodes • ${data.num_edges} edges`;

    } catch (e) {
        console.error(e);
        document.getElementById('graph-info').textContent = 'Failed to load graph';
    }
}

// ===============================
// STYLE (ANTI-CLUTTER)
// ===============================
function getStyle() {
    return [
        // ===== NODE =====
        {
            selector: 'node',
            style: {
                'background-color': '#FF6B6B',
                'label': 'data(label)',
                'width': 34,
                'height': 34,
                'font-size': 9,
                'text-valign': 'center',
                'text-halign': 'center',
                'color': '#fff',
                'text-outline-width': 2,
                'text-outline-color': '#1E1E1E'
            }
        },
        { selector: 'node.source', style: { 'background-color': '#51CF66', 'width': 48, 'height': 48 } },
        { selector: 'node.target', style: { 'background-color': '#FFD93D', 'width': 48, 'height': 48 } },
        { selector: 'node.path', style: { 'background-color': '#4ECDC4', 'width': 44, 'height': 44 } },

        // ===== EDGE (DEFAULT = FADED) =====
        {
            selector: 'edge',
            style: {
                'line-color': '#fff',
                'opacity': 0.02,
                'width': 0.5,
                'curve-style': 'straight',
                'target-arrow-shape': 'none',
                'z-index-compare': 'manual',
                'z-index': 0
            }
        },

        // Edge visible (context)
        {
            selector: 'edge.visible',
            style: {
                'opacity': 0.15,
                'width': 1.2
            }
        },

        // Dimmed styling
        {
            selector: '.dimmed',
            style: {
                'opacity': 0
            }
        },

        // Edge path (MUST be after .dimmed for proper override)
        {
            selector: 'edge.path',
            style: {
                'opacity': 1,
                'line-color': '#00ACC1',
                'width': 10,
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#00ACC1',
                'arrow-scale': 1.5,
                'curve-style': 'straight',
                'z-index': 999999,
                'z-index-compare': 'manual',
                'line-cap': 'round',
                'line-join': 'round',
                'shadow-blur': 10,
                'shadow-color': '#00ACC1',
                'shadow-opacity': 0.8
            }
        },

        // Node path
        {
            selector: 'node.path',
            style: {
                'opacity': 1
            }
        }
    ];
}

// ===============================
// GRID LAYOUT
// ===============================
function applyGridLayout() {
    const n = cy.nodes().length;
    const size = Math.ceil(Math.sqrt(n));

    cy.layout({
        name: 'grid',
        rows: size,
        cols: size,
        padding: 70,
        fit: true,
        animate: true,
        animationDuration: 500
    }).run();
}

// ===============================
// EVENTS
// ===============================
function bindEvents() {
    let nodeDragInfo = null;

    const resetNodeDrag = () => {
        nodeDragInfo = null;
    };

    // Track tapstart trên node
    cy.on('tapstart', 'node', evt => {
        const { x, y } = evt.renderedPosition;
        nodeDragInfo = {
            startX: x,
            startY: y,
            lastX: x,
            lastY: y,
            dragging: false
        };
    });

    // Tapdrag trên node → pan viewport
    cy.on('tapdrag', 'node', evt => {
        if (!nodeDragInfo) return;
        const { x, y } = evt.renderedPosition;

        if (!nodeDragInfo.dragging && Math.hypot(x - nodeDragInfo.startX, y - nodeDragInfo.startY) > 4) {
            nodeDragInfo.dragging = true;
        }

        if (nodeDragInfo.dragging) {
            const dx = x - nodeDragInfo.lastX;
            const dy = y - nodeDragInfo.lastY;
            if (dx !== 0 || dy !== 0) {
                cy.panBy({ x: dx, y: dy });
            }
        }

        nodeDragInfo.lastX = x;
        nodeDragInfo.lastY = y;
    });

    // Dừng tracking khi kết thúc drag
    cy.on('tapend', resetNodeDrag);

    // Node tap → select (chỉ khi click không drag)
    cy.on('tap', 'node', evt => {
        if (nodeDragInfo && nodeDragInfo.dragging) {
            resetNodeDrag();
            return;
        }

        const node = evt.target;

        cy.edges().removeClass('visible');
        node.connectedEdges().addClass('visible');

        selectNode(node.id());
        resetNodeDrag();
    });

    // Background click → hide edges
    cy.on('tap', evt => {
        if (evt.target === cy) {
            cy.edges().removeClass('visible');
        }
    });

    // Zoom-based LOD
    cy.on('zoom', () => {
        const z = cy.zoom();
        // Nếu path đang active, tất cả non-path edges ẩn
        if (isPathActive) {
            cy.edges().not('.path').style('opacity', 0);
        } else {
            // Không thay đổi opacity của edges có class 'path'
            const nonPathEdges = cy.edges().not('.path');

            if (z < 0.6) nonPathEdges.style('opacity', 0);
            else if (z < 1.2) nonPathEdges.style('opacity', 0.04);
            else nonPathEdges.style('opacity', 0.12);
        }
    });

    // Enable pan with mouse drag anywhere
    cy.on('vmousedown', () => {
        cy.panningEnabled(true);
    });
}

// ===============================
// NODE SELECTION
// ===============================
function selectNode(nodeId) {
    cy.nodes().removeClass('source target');

    if (!selectedSourceNode) {
        selectedSourceNode = nodeId;
        cy.$id(nodeId).addClass('source');
    }
    else if (!selectedTargetNode && nodeId !== selectedSourceNode) {
        selectedTargetNode = nodeId;
        cy.$id(selectedSourceNode).addClass('source');
        cy.$id(nodeId).addClass('target');
        document.getElementById('find-path-btn').disabled = false;
    }
    else {
        resetSelection();
        return;
    }

    updateNodeDisplay();
}

function updateNodeDisplay() {
    document.getElementById('source-node').textContent =
        selectedSourceNode ?? 'None';
    document.getElementById('target-node').textContent =
        selectedTargetNode ?? 'None';
}

// ===============================
// RESET
// ===============================
function resetSelection() {
    selectedSourceNode = null;
    selectedTargetNode = null;
    isPathActive = false;

    // Remove tất cả classes
    cy.nodes().removeClass('source target path dimmed');
    cy.edges().removeClass('path visible dimmed');
    cy.edges().style('opacity', '');

    document.getElementById('find-path-btn').disabled = true;
    document.getElementById('result-box').style.display = 'none';

    updateNodeDisplay();

    // Restore metrics và samples
    updateMetrics();
    updateSamples();
}

// ===============================
// FIND PATH
// ===============================
async function findPath() {
    if (!selectedSourceNode || !selectedTargetNode) return;

    const btn = document.getElementById('find-path-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Finding...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source: Number(selectedSourceNode),
                target: Number(selectedTargetNode)
            })
        });

        const result = await res.json();
        displayPath(result);

    } catch (e) {
        alert('Prediction failed');
    }

    btn.disabled = false;
    btn.textContent = '⚡ Find Shortest Path';
}

// ===============================
// PATH VISUALIZATION
// ===============================
function displayPath(result) {
    highlightPath(result.ml_path);

    document.getElementById('ml-path').textContent = result.ml_path.join(' → ');
    document.getElementById('ml-cost').textContent = `Cost: ${result.ml_cost.toFixed(2)}`;
    document.getElementById('dij-path').textContent = result.dijkstra_path.join(' → ');
    document.getElementById('dij-cost').textContent = `Cost: ${result.dijkstra_cost.toFixed(2)}`;
    document.getElementById('latency').textContent = `${result.latency_ms.toFixed(2)} ms`;

    const badge = document.getElementById('path-status');
    badge.textContent = result.is_correct ? '✓ CORRECT' : '✗ INCORRECT';
    badge.className = result.is_correct ? 'status-badge correct' : 'status-badge incorrect';

    document.getElementById('result-box').style.display = 'block';
}

function highlightPath(path) {
    isPathActive = true;

    // 1. Reset toàn bộ state trước
    cy.nodes().removeClass('path dimmed');
    cy.edges().removeClass('path visible dimmed');
    
    // Reset ALL edge styles
    cy.edges().forEach(edge => {
        edge.style('opacity', 0.02);
        edge.style('z-index', 0);
        edge.style('line-color', '#fff');
        edge.style('width', 0.5);
    });

    // 2. Làm mờ TẤT CẢ node & edge
    cy.nodes().addClass('dimmed');
    cy.edges().addClass('dimmed');

    // 3. Làm nổi bật node trong path
    path.forEach(id => {
        const node = cy.$id(String(id));
        node.removeClass('dimmed');
        node.addClass('path');
    });

    // 4. Highlight ONLY the path edges - no batching, direct style
    console.log('🔍 Looking for path edges for:', path);
    let pathEdgeCount = 0;

    for (let i = 0; i < path.length - 1; i++) {
        const nodeA = String(path[i]);
        const nodeB = String(path[i + 1]);

        // Search ALL edges for this connection
        const matchedEdges = cy.edges().filter(edge => {
            const src = String(edge.data('source'));
            const tgt = String(edge.data('target'));
            return (src === nodeA && tgt === nodeB) || (src === nodeB && tgt === nodeA);
        });

        console.log(`Segment ${nodeA}→${nodeB}: found ${matchedEdges.length} edge(s)`);

        // Apply path styling to ALL matched edges
        matchedEdges.forEach(edge => {
            edge.removeClass('dimmed');
            edge.addClass('path');
            // Force inline styles to override CSS
            edge.style('opacity', 1);
            edge.style('line-color', '#00ACC1');
            edge.style('width', 10);
            edge.style('z-index', 999999);
            pathEdgeCount++;
        });
    }

    console.log(`✓ Total path edges styled: ${pathEdgeCount}`);

    // 5. Force hide all non-path edges
    cy.edges().not('.path').forEach(edge => {
        edge.style('opacity', 0);
    });
}

// ===============================
// UPDATE METRICS & SAMPLES
// ===============================
async function updateMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const metrics = await res.json();

        document.getElementById('metric-latency').textContent =
            metrics.avg_latency_ms.toFixed(2) + ' ms';
        document.getElementById('metric-accuracy').textContent =
            (metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('stat-predictions').textContent =
            metrics.total_predictions;
        document.getElementById('stat-accuracy').textContent =
            (metrics.accuracy * 100).toFixed(1) + '%';

        const maxLatency = 100;
        const latencyPercent = Math.min((metrics.avg_latency_ms / maxLatency) * 100, 100);
        document.getElementById('latency-bar').style.width = latencyPercent + '%';
        document.getElementById('accuracy-bar').style.width = (metrics.accuracy * 100) + '%';
    } catch (e) {
        console.error('Error updating metrics:', e);
    }
}

async function updateSamples() {
    try {
        const res = await fetch('/api/samples');
        const data = await res.json();

        const table = document.getElementById('samples-table');

        if (data.samples.length === 0) {
            table.innerHTML = '<div class="empty-state">No predictions yet</div>';
            return;
        }

        table.innerHTML = data.samples.slice(-10).reverse().map(sample => `
            <div class="sample-item">
                <strong>${sample.source} → ${sample.target}</strong>
                ML: ${sample.ml_cost.toFixed(2)} | 
                True: ${sample.dijkstra_cost.toFixed(2)} |
                ${sample.is_correct ? '✓' : '✗'}
            </div>
        `).join('');
    } catch (e) {
        console.error('Error updating samples:', e);
    }
}

async function loadSummary() {
    try {
        const res = await fetch('/api/summary');
        const data = await res.json();

        document.getElementById('stat-nodes').textContent = data.graph_stats.num_nodes;
        document.getElementById('stat-edges').textContent = data.graph_stats.num_edges;
    } catch (e) {
        console.error('Error loading summary:', e);
    }
}

// ===============================
// START
// ===============================
document.addEventListener('DOMContentLoaded', () => {
    initCytoscape();
    loadSummary();
    updateMetrics();
    updateSamples();

    document.getElementById('find-path-btn').addEventListener('click', findPath);
    document.getElementById('reset-btn').addEventListener('click', resetSelection);

    // Auto-update metrics every 2 seconds
    setInterval(() => {
        updateMetrics();
    }, 2000);
});
