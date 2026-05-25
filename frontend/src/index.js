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
  ? 'http://localhost:3000'
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
    for (const src of neurons[i]) {
      for (const dst of neurons[i + 1]) {
        const hasFiredData = firedMap && firedMap[i];
        const fired = hasFiredData && firedMap[i][neurons[i].indexOf(src)];

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(dst.x, dst.y);
        ctx.strokeStyle = fired ? 'rgba(172,255,48,0.45)' : 'rgba(210,210,210,0.55)';
        ctx.lineWidth   = fired ? 2 : 1;
        ctx.stroke();
      }
    }
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

  drawNetwork(data.firedNeurons);

  //  Display outputs 
  resultDiv.textContent    = 'Output:';
  resultValues.textContent = data.finalOutputs.map((v) => v.toFixed(4)).join('   ');

  // Celebration GIF 🎉
  gifDiv.style.display = 'block';
  setTimeout(() => { gifDiv.style.display = 'none'; }, 1200);
});
