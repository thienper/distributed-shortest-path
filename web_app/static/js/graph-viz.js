// Global variables
let cy = null;
let selectedSourceNode = null;
let selectedTargetNode = null;

// Initialize Cytoscape
async function initCytoscape() {
    try {
        const response = await fetch('/api/graph');
        const data = await response.json();

        if (data.error) {
            console.error('Error loading graph:', data.error);
            document.getElementById('graph-info').textContent = 'Error loading graph';
            return;
        }

        cy = cytoscape({
            container: document.getElementById('cy'),
            style: getCytoscapeStyle(),
            elements: [...data.nodes, ...data.edges],
            wheelSensitivity: 0.1,
            autoungrabify: true
        });

        // Apply layout
        applyLayout();

        // Bind events
        bindCytoscapeEvents();

        // Update info
        document.getElementById('graph-info').textContent =
            `${data.num_nodes} nodes • ${data.num_edges} edges`;

        console.log('Cytoscape initialized');
    } catch (error) {
        console.error('Failed to initialize Cytoscape:', error);
        document.getElementById('graph-info').textContent = 'Failed to load graph';
    }
}

function getCytoscapeStyle() {
    return [
        {
            selector: 'node',
            style: {
                'background-color': '#FF6B6B',
                'label': 'data(label)',
                'width': 35,
                'height': 35,
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': 10,
                'color': '#fff',
                'text-outline-width': 2,
                'text-outline-color': '#1E1E1E',
                'border-width': 1,
                'border-color': 'rgba(0,0,0,0.3)',
                'transition-property': 'background-color, width, height',
                'transition-duration': '200ms'
            }
        },
        {
            selector: 'node:selected',
            style: {
                'background-color': '#FFD93D',
                'width': 50,
                'height': 50,
                'text-outline-color': '#FFD93D',
                'border-width': 2,
                'border-color': '#FFF'
            }
        },
        {
            selector: 'node.source',
            style: {
                'background-color': '#51CF66',
                'width': 50,
                'height': 50,
                'border-width': 2,
                'border-color': '#fff'
            }
        },
        {
            selector: 'node.target',
            style: {
                'background-color': '#FF6B6B',
                'width': 50,
                'height': 50,
                'border-width': 2,
                'border-color': '#fff'
            }
        },
        {
            selector: 'node.path',
            style: {
                'background-color': '#4ECDC4',
                'width': 45,
                'height': 45,
                'border-width': 2,
                'border-color': '#fff'
            }
        },
        {
            selector: 'edge',
            style: {
                'line-color': 'rgba(255,255,255,0.15)',
                'width': 1.5,
                'target-arrow-color': 'rgba(255,255,255,0.15)',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 0.8,
                'curve-style': 'bezier',
                'transition-property': 'line-color, width',
                'transition-duration': '200ms'
            }
        },
        {
            selector: 'edge:selected',
            style: {
                'line-color': '#FFD93D',
                'width': 2.5,
                'target-arrow-color': '#FFD93D'
            }
        },
        {
            selector: 'edge.path',
            style: {
                'line-color': '#4ECDC4',
                'width': 4,
                'target-arrow-color': '#4ECDC4',
                'target-arrow-fill': 'filled'
            }
        }
    ];
}

function applyLayout() {
    if (!cy) return;

    const layout = cy.layout({
        name: 'grid',
        directed: false,
        rows: Math.ceil(Math.sqrt(cy.nodes().length)),
        cols: Math.ceil(Math.sqrt(cy.nodes().length)),
        fit: true,
        padding: 50,
        animate: true,
        animationDuration: 500,
        animationEasing: 'ease-out',
        stop: function () {
            console.log('Layout complete - Grid layout applied');
        }
    });

    layout.run();
}

function bindCytoscapeEvents() {
    if (!cy) return;

    // Click on node to select
    cy.on('tap', 'node', function (evt) {
        evt.stopPropagation();
        const node = evt.target;
        const nodeId = node.id();
        console.log('Node clicked:', nodeId);
        selectNode(nodeId);
    });

    // Click on background to deselect
    cy.on('tap', function (evt) {
        if (evt.target === cy) {
            console.log('Background clicked');
            // Could add deselect here if needed
        }
    });
}

function selectNode(nodeId) {
    // Remove selection from all nodes
    cy.nodes().removeClass('source target');

    // If no source selected, set as source
    if (selectedSourceNode === null) {
        selectedSourceNode = nodeId;
        cy.$id(nodeId).addClass('source');
        updateNodeDisplay();
    }
    // If source selected but different, set as target
    else if (selectedSourceNode !== nodeId) {
        // Clear previous target
        if (selectedTargetNode !== null) {
            cy.$id(selectedTargetNode).removeClass('target');
        }
        selectedTargetNode = nodeId;
        cy.$id(selectedSourceNode).addClass('source');
        cy.$id(nodeId).addClass('target');
        updateNodeDisplay();

        // Enable find path button
        document.getElementById('find-path-btn').disabled = false;
    }
    // If clicking same node as source, toggle to target
    else {
        selectedSourceNode = null;
        updateNodeDisplay();
        document.getElementById('find-path-btn').disabled = true;
    }
}

function updateNodeDisplay() {
    const sourceEl = document.getElementById('source-node');
    const targetEl = document.getElementById('target-node');

    if (selectedSourceNode !== null) {
        sourceEl.textContent = selectedSourceNode;
        sourceEl.classList.remove('empty');
    } else {
        sourceEl.textContent = 'None';
        sourceEl.classList.add('empty');
    }

    if (selectedTargetNode !== null) {
        targetEl.textContent = selectedTargetNode;
        targetEl.classList.remove('empty');
    } else {
        targetEl.textContent = 'None';
        targetEl.classList.add('empty');
    }
}

function resetSelection() {
    selectedSourceNode = null;
    selectedTargetNode = null;
    cy.nodes().removeClass('source target path');
    cy.edges().removeClass('path');
    document.getElementById('find-path-btn').disabled = true;
    document.getElementById('result-box').style.display = 'none';
    updateNodeDisplay();
}

async function findPath() {
    if (selectedSourceNode === null || selectedTargetNode === null) {
        alert('Please select both source and target nodes');
        return;
    }

    try {
        document.getElementById('find-path-btn').disabled = true;
        document.getElementById('find-path-btn').textContent = '⏳ Finding...';

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source: parseInt(selectedSourceNode),
                target: parseInt(selectedTargetNode)
            })
        });

        const result = await response.json();

        if (result.error) {
            alert('Error: ' + result.error);
            return;
        }

        displayPathResult(result);
        updateMetrics();
        updateSamples();
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to find path');
    } finally {
        document.getElementById('find-path-btn').disabled = false;
        document.getElementById('find-path-btn').textContent = '⚡ Find Shortest Path';
    }
}

function displayPathResult(result) {
    // Highlight path on graph
    highlightPath(result.ml_path);

    // Display result
    const resultBox = document.getElementById('result-box');
    document.getElementById('ml-path').textContent = result.ml_path.join(' → ');
    document.getElementById('ml-cost').textContent = `Cost: ${result.ml_cost.toFixed(2)}`;
    document.getElementById('dij-path').textContent = result.dijkstra_path.join(' → ');
    document.getElementById('dij-cost').textContent = `Cost: ${result.dijkstra_cost.toFixed(2)}`;

    const statusBadge = document.getElementById('path-status');
    if (result.is_correct) {
        statusBadge.textContent = '✓ CORRECT';
        statusBadge.className = 'status-badge correct';
    } else {
        statusBadge.textContent = '✗ INCORRECT';
        statusBadge.className = 'status-badge incorrect';
    }

    document.getElementById('latency').textContent = result.latency_ms.toFixed(2) + ' ms';

    resultBox.style.display = 'block';
}

function highlightPath(path) {
    if (!cy) return;

    // Clear previous highlighting
    cy.nodes().removeClass('path');
    cy.edges().removeClass('path');

    // Highlight nodes
    for (let i = 0; i < path.length; i++) {
        cy.$id(String(path[i])).addClass('path');
    }

    // Highlight edges
    for (let i = 0; i < path.length - 1; i++) {
        const edgeId = `${path[i]}-${path[i + 1]}`;
        const edge1 = cy.$id(edgeId);
        const edge2 = cy.$id(`${path[i + 1]}-${path[i]}`);

        if (edge1.length > 0) {
            edge1.addClass('path');
        }
        if (edge2.length > 0) {
            edge2.addClass('path');
        }
    }
}

async function updateMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const metrics = await response.json();

        document.getElementById('metric-latency').textContent =
            metrics.avg_latency_ms.toFixed(2) + ' ms';
        document.getElementById('metric-accuracy').textContent =
            (metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('stat-predictions').textContent =
            metrics.total_predictions;
        document.getElementById('stat-accuracy').textContent =
            (metrics.accuracy * 100).toFixed(1) + '%';

        // Update progress bars
        const maxLatency = 100;
        const latencyPercent = Math.min((metrics.avg_latency_ms / maxLatency) * 100, 100);
        document.getElementById('latency-bar').style.width = latencyPercent + '%';

        document.getElementById('accuracy-bar').style.width = (metrics.accuracy * 100) + '%';
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

async function updateSamples() {
    try {
        const response = await fetch('/api/samples');
        const data = await response.json();

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
    } catch (error) {
        console.error('Error updating samples:', error);
    }
}

async function loadSummary() {
    try {
        const response = await fetch('/api/summary');
        const data = await response.json();

        document.getElementById('stat-nodes').textContent = data.graph_stats.num_nodes;
        document.getElementById('stat-edges').textContent = data.graph_stats.num_edges;
    } catch (error) {
        console.error('Error loading summary:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    initCytoscape();
    loadSummary();

    document.getElementById('find-path-btn').addEventListener('click', findPath);
    document.getElementById('reset-btn').addEventListener('click', resetSelection);

    // Auto-update metrics every 2 seconds
    setInterval(() => {
        updateMetrics();
    }, 2000);
});
