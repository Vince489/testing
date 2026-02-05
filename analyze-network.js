import fs from 'fs';
import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

// Load the trained network using the binary format
const network = await OptimizedNeuralNetwork.load('goals_embedding_network.json');

// Analyze the network configuration
console.log('=== NETWORK CONFIGURATION ===');
console.log('Layer Configs:', network.layerConfigs);
console.log('Learning Rate:', network.learningRate);
console.log('Output Activation Function:', network.outputActivation);
console.log('Loss Function:', network.lossFunction);
console.log('Use Adam:', network.useAdam);
console.log('Gradient Clipping:', network.gradientClipping);

// Analyze the weights
console.log('\n=== WEIGHTS ANALYSIS ===');
for (let i = 0; i < network.layers.length; i++) {
  const layer = network.layers[i];
  console.log(`\nLayer ${i} to ${i+1} weights:`);

  // Check for extreme values
  let minVal = Infinity;
  let maxVal = -Infinity;
  let zeroCount = 0;
  let nearOneCount = 0; // Values close to 1 or -1
  let totalCount = layer.weights.length;

  for (let j = 0; j < layer.weights.length; j++) {
    const val = layer.weights[j];
    minVal = Math.min(minVal, val);
    maxVal = Math.max(maxVal, val);

    if (Math.abs(val) < 0.0001) {
      zeroCount++;
    }

    if (Math.abs(Math.abs(val) - 1) < 0.0001) {
      nearOneCount++;
    }
  }

  console.log(`  Min value: ${minVal.toFixed(6)}`);
  console.log(`  Max value: ${maxVal.toFixed(6)}`);
  console.log(`  Zero or near-zero values: ${zeroCount} (${totalCount > 0 ? (zeroCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
  console.log(`  Values near 1 or -1: ${nearOneCount} (${totalCount > 0 ? (nearOneCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
}

// Analyze the biases
console.log('\n=== BIASES ANALYSIS ===');
for (let i = 0; i < network.layers.length; i++) {
  const layer = network.layers[i];
  if (layer.biases) {
    console.log(`\nLayer ${i+1} biases:`);

    let minVal = Infinity;
    let maxVal = -Infinity;
    let zeroCount = 0;
    let nearOneCount = 0;
    let totalCount = layer.biases.length;

    for (let j = 0; j < layer.biases.length; j++) {
      const val = layer.biases[j];
      minVal = Math.min(minVal, val);
      maxVal = Math.max(maxVal, val);

      if (Math.abs(val) < 0.0001) {
        zeroCount++;
      }

      if (Math.abs(Math.abs(val) - 1) < 0.0001) {
        nearOneCount++;
      }
    }

    console.log(`  Min value: ${minVal.toFixed(6)}`);
    console.log(`  Max value: ${maxVal.toFixed(6)}`);
    console.log(`  Zero or near-zero values: ${zeroCount} (${totalCount > 0 ? (zeroCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
    console.log(`  Values near 1 or -1: ${nearOneCount} (${totalCount > 0 ? (nearOneCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
  }
}

console.log('\nAnalysis complete!');