/**
 * ANN Visualizer. Frontend
 *
 * Responsibilities:
 *  1. Collect topology config from the user (inputs, hidden layers, output count)
 *  2. Render the network on an HTML5 Canvas
 *  3. Accept input values, POST them to /run-network, and animate the results
 */

'use strict';

const API_BASE = window.location.hostname === 'localhost' 
  ? 'http://localhost:1112'
  : 'https://ai-neuroncanvas.onrender.com'; // localhost or Render.com API







//  Canvas setup 
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');



//  Layout constants ─
const NEURON_RADIUS  = 22;    // px — circle radius
const X_SPACING      = 180;   // px — horizontal gap between layers
const Y_SPACING      = 90;    // px — vertical gap between neurons
const CANVAS_PADDING = 60;    // px — outer padding



//  Network state 
/** @type {Array<Array<{x:number, y:number, fired:boolean, value:number}>>} */
let neurons     = [];
let inputCount  = 0;
let hiddenLayers= [];
let outputCount = 2;
// Data from the most recent successful /run-network call.
// lastWeights[layerIdx][dstNeuronIdx][srcNeuronIdx] = connection weight.
// lastFiredMap[layerIdx][neuronIdx] = 1 if that neuron fired, 0 otherwise.
// (layerIdx here is the "gap" index — 0 = input→hidden1, 1 = hidden1→hidden2, …)
let lastWeights     = null;
let lastFiredMap    = null;
let lastRunData     = null;   // full response, kept so the explain popup can be reopened
let lastRunInputs   = null;
let hoveredConnection = null; // { layerIdx, srcIdx, dstIdx } | null



//  Preset configurations 
const PRESETS = {
  xor: {
    label: 'XOR Gate',
    inputCount: 2,
    hiddenLayers: [4],
    outputCount: 2,
    sampleInput: '0, 1',
    description: 'Classic XOR logic gate. Tests non-linear separability.',
  },
  classifier: {
    label: '3-Class Classifier',
    inputCount: 4,
    hiddenLayers: [8, 6],
    outputCount: 3,
    sampleInput: '0.5, 0.8, 0.2, 0.9',
    description: 'Multi-layer network for classifying 4-feature inputs into 3 classes.',
  },
  deep: {
    label: 'Deep Network',
    inputCount: 3,
    hiddenLayers: [6, 6, 4],
    outputCount: 2,
    sampleInput: '1, 0.5, 0.2',
    description: 'Deeper architecture: watch activations propagate through many layers.',
  },
  single: {
    label: 'Single Neuron',
    inputCount: 2,
    hiddenLayers: [1],
    outputCount: 1,
    sampleInput: '0.4, 0.7',
    description: 'Stripped-back network: one hidden neuron. Great for learning the basics.',
  },
};




//  DOM refs 
const inputCountEl   = document.getElementById('inputCount');
const hiddenLayerEl  = document.getElementById('hiddenLayerCount');
const outputCountEl  = document.getElementById('outputCount');
const renderBtn      = document.getElementById('renderbtn');
const inputForm      = document.getElementById('inputForm');
const networkInput   = document.getElementById('networkInput');
const resultDiv      = document.getElementById('result');
const resultValues   = document.getElementById('resultValues');
const warningDiv     = document.getElementById('outputWarning');
const gifDiv         = document.getElementById('gifDiv');
const toHideDiv      = document.querySelector('.toHide');
const inputHintEl    = document.getElementById('inputHint');
const legendEl       = document.getElementById('legend');

const weightTooltip  = document.getElementById('weightTooltip');
const explainOverlay = document.getElementById('explainOverlay');
const explainBody    = document.getElementById('explainBody');
const explainClose   = document.getElementById('explainClose');
const explainGotIt   = document.getElementById('explainGotIt');
const reopenExplainBtn = document.getElementById('reopenExplain');





//  Preset buttons 
document.querySelectorAll('[data-preset]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const key    = btn.dataset.preset;
    const preset = PRESETS[key];
    if (!preset) return;

    inputCountEl.value  = preset.inputCount;
    hiddenLayerEl.value = preset.hiddenLayers.join(', ');
    outputCountEl.value = preset.outputCount;

    document.getElementById('presetDescription').textContent = preset.description;
    networkInput.placeholder = `e.g. ${preset.sampleInput}`;
  });
});




//  Quick-fill templates (Step 3) 
function buildTemplateValues(kind) {
  switch (kind) {
    case 'uniform':
      return Array.from({ length: inputCount }, () => 0.5);
    case 'random':
      return Array.from({ length: inputCount }, () => Math.round(Math.random() * 100) / 100);
    case 'alt':
      return Array.from({ length: inputCount }, (_, i) => (i % 2 === 0 ? 0 : 1));
    default:
      return [];
  }
}

document.querySelectorAll('[data-fill]').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (!inputCount) return;
    networkInput.value = buildTemplateValues(btn.dataset.fill).join(', ');
    warningDiv.textContent = '';
  });
});




//  Render button 
renderBtn.addEventListener('click', () => {
  const parsedInput  = parseInt(inputCountEl.value, 10);
  const parsedHidden = hiddenLayerEl.value.split(',').map((s) => parseInt(s.trim(), 10));
  const parsedOutput = parseInt(outputCountEl.value, 10) || 2;

  // Validate topology inputs
  if (
    isNaN(parsedInput)  || parsedInput  < 1 ||
    parsedHidden.some((n) => isNaN(n) || n < 1) ||
    isNaN(parsedOutput) || parsedOutput < 1
  ) {
    warningDiv.textContent = 'Please enter valid numbers for all topology fields.';
    return;
  }

  warningDiv.textContent = '';
  inputCount   = parsedInput;
  hiddenLayers = parsedHidden;
  outputCount  = parsedOutput;

  lastWeights        = null;
  lastFiredMap       = null;
  lastRunData        = null;
  lastRunInputs      = null;
  hoveredConnection  = null;
  reopenExplainBtn.classList.remove('show');
  closeExplainModal();

  buildNeurons();
  resizeCanvas();
  drawNetwork();

  // Update the input hint
  inputHintEl.textContent =
    `Enter ${inputCount} comma-separated numbers between 0 and 1 (e.g. ${
      Array.from({ length: inputCount }, () => (Math.random()).toFixed(1)).join(', ')
    })`;
  networkInput.placeholder =
    Array.from({ length: inputCount }, () => (Math.random()).toFixed(1)).join(', ');

  // Show the interactive section
  toHideDiv.style.display = 'block';
  resultDiv.textContent   = '';
  resultValues.textContent= '';
  gifDiv.style.display    = 'none';
});

//  Build neuron data model ─
function buildNeurons() {
  neurons = [];
  const layerSizes = [inputCount, ...hiddenLayers, outputCount];
  const maxNeurons = Math.max(...layerSizes);

  layerSizes.forEach((count, layerIndex) => {
    const layerHeight  = (count - 1) * Y_SPACING;
    const totalHeight  = (maxNeurons - 1) * Y_SPACING;
    const topOffset    = CANVAS_PADDING + (totalHeight - layerHeight) / 2;
    const x            = CANVAS_PADDING + layerIndex * X_SPACING;

    const layer = Array.from({ length: count }, (_, neuronIndex) => ({
      x,
      y: topOffset + neuronIndex * Y_SPACING,
      fired: false,
      value: 0,
    }));

    neurons.push(layer);
  });
}

//  Canvas sizing 
function resizeCanvas() {
  const totalLayers  = 2 + hiddenLayers.length;
  const maxNeurons   = Math.max(inputCount, ...hiddenLayers, outputCount);

  canvas.width  = CANVAS_PADDING * 2 + (totalLayers - 1) * X_SPACING;
  canvas.height = CANVAS_PADDING * 2 + (maxNeurons - 1) * Y_SPACING;
}

//  Drawing helpers 
/** Returns the fill colour for a neuron given its layer position and fired state. */
function neuronColor(layerIndex, isFired, value) {
  const isInput  = layerIndex === 0;
  const isOutput = layerIndex === neurons.length - 1;

  if (isInput)  return '#dde8f0';
  if (isOutput) return '#b0bec5';

  if (isFired)  return `rgba(172, 255, 48, ${0.5 + value * 0.5})`;
  return `rgba(220, 80, 60, ${0.4 + (1 - value) * 0.4})`;
}



function drawConnections(firedMap) {
  for (let i = 0; i < neurons.length - 1; i++) {

    neurons[i].forEach((src, srcIdx) => {
      neurons[i + 1].forEach((dst, dstIdx) => {
        const hasFiredData = firedMap && firedMap[i];
        // fired state is per DESTINATION neuron, not source
        const fired = hasFiredData && firedMap[i][dstIdx];
        const isHovered =
          hoveredConnection &&
          hoveredConnection.layerIdx === i &&
          hoveredConnection.srcIdx === srcIdx &&
          hoveredConnection.dstIdx === dstIdx;

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(dst.x, dst.y);
        // ctx.strokeStyle = fired ? 'rgba(172,255,48,0.45)' : 'rgba(210,210,210,0.55)';
        // ctx.lineWidth   = fired ? 2 : 1;

        if (isHovered) {
          ctx.strokeStyle = 'rgba(108, 74, 220, 0.9)';
          ctx.lineWidth   = 3.5;
        } else {
          ctx.strokeStyle = fired ? 'rgba(172,255,48,0.45)' : 'rgba(210,210,210,0.55)';
          ctx.lineWidth   = fired ? 2 : 1;
        }

        ctx.stroke();
      });
    });
  }
}




function drawNeurons() {
  neurons.forEach((layer, layerIndex) => {
    layer.forEach((neuron) => {
      // Shadow
      ctx.shadowColor  = neuron.fired ? 'rgba(172,255,48,0.6)' : 'rgba(0,0,0,0.15)';
      ctx.shadowBlur   = neuron.fired ? 16 : 6;

      // Fill
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, NEURON_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = neuronColor(layerIndex, neuron.fired, neuron.value);
      ctx.fill();

      // Border
      ctx.strokeStyle = neuron.fired ? 'rgb(172,255,48)' : 'rgba(180,180,180,0.6)';
      ctx.lineWidth   = neuron.fired ? 2.5 : 1.5;
      ctx.stroke();
      ctx.shadowBlur  = 0;

      // Value label (shown after a run)
      if (neuron.value !== 0 || layerIndex === 0) {
        ctx.fillStyle = '#222';
        ctx.font      = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(neuron.value.toFixed(2), neuron.x, neuron.y + 4);
      }
    });
  });
}

function drawLayerLabels() {
  const labels = [
    'Input',
    ...hiddenLayers.map((_, i) => `Hidden ${i + 1}`),
    'Output',
  ];

  ctx.fillStyle = 'rgba(80,80,80,0.75)';
  ctx.font      = '12px "Montserrat", sans-serif';
  ctx.textAlign = 'center';

  neurons.forEach((layer, i) => {
    ctx.fillText(labels[i], layer[0].x, CANVAS_PADDING / 2);
  });
}



function drawNetwork(firedMap) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawConnections(firedMap);
  drawNeurons();
  drawLayerLabels();
}



//  Weight hover tooltip 
function distanceToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq !== 0 ? ((px - x1) * dx + (py - y1) * dy) / lenSq : -1;
  t = Math.max(0, Math.min(1, t));
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  return Math.hypot(px - projX, py - projY);
}

function findNearestConnection(mx, my) {
  const HOVER_THRESHOLD = 6;
  let best = null;
  let bestDist = HOVER_THRESHOLD;

  for (let i = 0; i < neurons.length - 1; i++) {
    neurons[i].forEach((src, srcIdx) => {
      neurons[i + 1].forEach((dst, dstIdx) => {
        const d = distanceToSegment(mx, my, src.x, src.y, dst.x, dst.y);
        if (d < bestDist) {
          bestDist = d;
          best = { layerIdx: i, srcIdx, dstIdx };
        }
      });
    });
  }
  return best;
}

canvas.addEventListener('mousemove', (e) => {
  if (neurons.length === 0) return;

  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top)  * scaleY;

  const found = findNearestConnection(mx, my);
  hoveredConnection = found;

  if (found) {
    canvas.style.cursor = 'pointer';
    if (lastWeights) {
      const w = lastWeights[found.layerIdx][found.dstIdx][found.srcIdx];
      weightTooltip.textContent = `weight: ${w.toFixed(3)}`;
    } else {
      weightTooltip.textContent = 'Run the network to see this weight';
    }
    weightTooltip.style.left    = `${e.clientX + 14}px`;
    weightTooltip.style.top     = `${e.clientY + 14}px`;
    weightTooltip.style.display = 'block';
  } else {
    canvas.style.cursor = 'default';
    weightTooltip.style.display = 'none';
  }

  drawNetwork(lastFiredMap);
});

canvas.addEventListener('mouseleave', () => {
  hoveredConnection = null;
  weightTooltip.style.display = 'none';
  canvas.style.cursor = 'default';
  drawNetwork(lastFiredMap);
});





//  Form submission - run forward pass
inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const raw        = networkInput.value;
  const inputArray = raw.split(',').map((s) => parseFloat(s.trim()));

  // Client-side validation
  if (
    inputArray.length !== inputCount ||
    inputArray.some((v) => isNaN(v))
  ) {
    warningDiv.textContent =
      `Please enter exactly ${inputCount} valid numbers separated by commas.`;
    return;
  }

  warningDiv.textContent = '';

  let data;
  try {
    const response = await fetch(`${API_BASE}/run-network`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        input:       inputArray,
        inputCount,
        hiddenLayers,
        outputCount,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error ?? 'Server error');
    }

    data = await response.json();
  } catch (err) {
    warningDiv.textContent = `Error: ${err.message}`;
    return;
  }

  //  Update visual model 
  // Input layer — show user values
  neurons[0].forEach((neuron, i) => {
    neuron.value = inputArray[i];
    neuron.fired = false;
  });

  // Hidden + output layers — update from server response
  data.firedNeurons.forEach((layerFired, li) => {
    layerFired.forEach((fired, ni) => {
      const neuron   = neurons[li + 1][ni];
      neuron.fired   = fired === 1;
      neuron.value   = data.eachLayerInputValues[li + 1][ni];
    });
  });

  lastWeights   = data.weights;
  lastFiredMap  = data.firedNeurons;
  lastRunData   = data;
  lastRunInputs = inputArray;

  drawNetwork(data.firedNeurons);

  //  Display outputs 
  resultDiv.textContent    = 'Output:';
  resultValues.textContent = data.finalOutputs.map((v) => v.toFixed(4)).join('   ');

  // Celebration GIF 🎉
  gifDiv.style.display = 'block';
  setTimeout(() => { gifDiv.style.display = 'none'; }, 7200);

  reopenExplainBtn.classList.add('show');
  showExplanation(inputArray, data);
});










//  "What just happened?" explanation popup 
function openExplainModal()  { explainOverlay.classList.add('active'); }
function closeExplainModal() { explainOverlay.classList.remove('active'); }

explainClose.addEventListener('click', closeExplainModal);
explainGotIt.addEventListener('click', closeExplainModal);
explainOverlay.addEventListener('click', (e) => {
  if (e.target === explainOverlay) closeExplainModal();
});
reopenExplainBtn.addEventListener('click', () => {
  if (!lastRunData) return;
  explainBody.innerHTML = buildExplanationHtml(lastRunInputs, lastRunData);
  openExplainModal();
});

const fmt2 = (n) => Number(n).toFixed(2);
const fmt3 = (n) => Number(n).toFixed(3);
console.log("fmt2: ", fmt2, "\n fmt3: ", fmt3)

function describeInputLayer(inputArray) {
  const list = inputArray.map(fmt2).join(', ');
  return `
    <div class="explain-section">
      <h3>Layer 1 · Input</h3>
      <p>You gave the network ${inputArray.length} number${inputArray.length === 1 ? '' : 's'}:
      <strong>${list}</strong>. No maths happens here — each value is just handed
      straight to the next layer as-is.</p>
    </div>
  `;
}

function describeComputedLayer(label, layerNumber, prevValues, weightsForLayer, zValues, values, firedFlags, isOutput) {
  const rows = values.map((value, ni) => {
    const w = weightsForLayer && weightsForLayer[ni];
    const z = zValues && zValues[ni];
    const fired = !!(firedFlags && firedFlags[ni] === 1);

    let sumExpr;
    if (w && prevValues && w.length <= 5) {
      const terms = w.map((wt, k) => `(${fmt2(prevValues[k])}×${fmt2(wt)})`);
      sumExpr = `${terms.join(' + ')} + 0.25 (bias)`;
    } else if (w) {
      sumExpr = `its ${w.length} weighted inputs + 0.25 (bias)`;
    } else {
      sumExpr = `its weighted inputs + bias`; // backend hasn't sent weight data
    }

    const rawScoreText = typeof z === 'number' ? `${fmt3(z)} raw score. ` : '';


    const verdict = isOutput
      ? `That's one of the network's final answers.`
      : fired
        ? `Since ${fmt3(value)} &gt; 0.6, it <strong class="fired-text">fired 🟢</strong>.`
        : `Since ${fmt3(value)} ≤ 0.6, it <strong class="unfired-text">did not fire 🔴</strong>.`;

    return `<li><strong>Neuron ${ni + 1}:</strong> ${sumExpr} = ${rawScoreText}
      Squashed by sigmoid, that becomes <strong>${fmt3(value)}</strong>. ${verdict}</li>`;
  }).join('');

  return `
    <div class="explain-section">
      <h3>Layer ${layerNumber} · ${label}</h3>
      <ul class="explain-list">${rows}</ul>
    </div>
  `;
}

function buildExplanationHtml(inputArray, data) {
  const layerLabels = ['Input', ...hiddenLayers.map((_, i) => `Hidden ${i + 1}`), 'Output'];
  const totalLayers = layerLabels.length;

  let explainIndices;
  let skippedCount = 0;

  if (totalLayers <= 5) {
    explainIndices = layerLabels.map((_, i) => i);
  } else {
    explainIndices = [0, 1, 2, 3, totalLayers - 1];
    skippedCount = totalLayers - 5;
  }

  const sections = explainIndices.map((layerIdx) => {
    if (layerIdx === 0) return describeInputLayer(inputArray);

    const compIdx = layerIdx - 1;

    // Check if the required data exists for this specific index before calling the function
    // Ensure all required data arrays exist for the specific component index
    const hasData = data.weights[compIdx] && 
                    data.eachLayerInputValues[layerIdx] &&
                    data.eachLayerZValues[compIdx] && 
                    data.firedNeurons[compIdx];

    // console.log("layerLabels[layerIdx]", layerLabels[layerIdx], "\n data.eachLayerInputValues[layerIdx - 1]: ", data.eachLayerInputValues[layerIdx - 1], "\ndata.weights[compIdx]: ", data.weights[compIdx], "\ndata.eachLayerZValues[compIdx]: ", data.eachLayerZValues[compIdx], "\ndata.eachLayerInputValues[layerIdx]", data.eachLayerInputValues[layerIdx], "\ndata.firedNeurons[compIdx]", data.firedNeurons[compIdx]);

    if (!hasData) {
      console.warn(`Missing data for layer index: ${layerIdx}`);
      return ''; 
    }


    return describeComputedLayer(
      layerLabels[layerIdx],
      layerIdx + 1,
      data.eachLayerInputValues[layerIdx - 1],
      data.weights[compIdx],
      data.eachLayerZValues[compIdx],
      data.eachLayerInputValues[layerIdx],
      data.firedNeurons[compIdx],
      layerIdx === totalLayers - 1
    );
  });

  let skipNote = '';
  if (skippedCount > 0) {
    skipNote = `
      <div class="explain-skip-note">
        This network has ${totalLayers} layers. To keep this readable, we walked through
        <strong>Input, Hidden 1, Hidden 2, Hidden 3</strong> and the <strong>Output</strong> layer —
        the ${skippedCount} hidden layer${skippedCount === 1 ? '' : 's'} in between
        ${skippedCount === 1 ? 'was' : 'were'} skipped from this explanation for brevity.
      </div>
    `;
  }

  return sections.join('') + skipNote;
}

function showExplanation(inputArray, data) {
  explainBody.innerHTML = buildExplanationHtml(inputArray, data);
  openExplainModal();
}