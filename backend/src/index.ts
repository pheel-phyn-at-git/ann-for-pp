import path from 'path';
import Express, { Request, Response } from 'express';
import cors from 'cors';

import { EActivationFunction } from './lib/neuronal-net/activation-functions';
import NeuronalNet from './lib/neuronal-net/neuronal-net';

const app = Express();
const PORT = process.env.PORT ?? 3000;

//  Middleware 
app.use(cors()); // allows your Vercel frontend to call this
app.use(Express.json()); // built-in body parser (Express ≥ 4.16)
app.use(Express.static(path.join(__dirname))); // serves index.html, styles.css, index.js …

//  Routes 
app.get('/', (_req: Request, res: Response) => {
  // res.sendFile(path.join(__dirname, 'index.html')); /* vercel now handles frontend */
  res.status(404).json({ message: 'Backend serves as API only. Frontend is on Vercel' })
});
  
// For Uptimerobot's interval ping check
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' })
})

/**
 * POST /run-network
 *
 * Receives the user's inputs and the network topology, builds a fresh network,
 * runs a forward pass, and returns the results.
 *
 * Body shape:
 * {
 *   input:        number[]   — values for each input neuron
 *   inputCount:   number     — number of input neurons
 *   hiddenLayers: number[]   — neurons per hidden layer, e.g. [4, 3]
 *   outputCount:  number     — number of output neurons
 * }
 */
app.post('/run-network', (req: Request, res: Response) => {
  const { input, inputCount, hiddenLayers, outputCount } = req.body as {
    input: number[];
    inputCount: number;
    hiddenLayers: number[];
    outputCount: number;
  };

  //  Basic validation 
  if (
    !Array.isArray(input) ||
    input.some((v) => typeof v !== 'number' || isNaN(v)) ||
    input.length !== inputCount
  ) {
    res.status(400).json({ error: `Expected ${inputCount} numeric inputs.` });
    return;
  }

  try {
    const nn = new NeuronalNet({
      activationFunction: EActivationFunction.SIGMOID,
      inputCount,
      hiddenLayers,
      outputCount,
    });

    const output = nn.send(input);
    res.json(output);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[NeuronalNet error]', message);
    res.status(400).json({ error: message });
  }
});

//  Start 
app.listen(PORT, () => {
  // console.log(`✓ ANN Visualizer running at http://localhost:${PORT}`);
  console.log(`✓ ANN API running on port ${PORT}`);
});
