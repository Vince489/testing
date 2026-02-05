/**
 * Test to verify save/load fix - compare weights before and after
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

// Test data for XOR problem
const xorData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

async function testSaveLoadFix() {
  console.log('🧪 TESTING SAVE/LOAD FIX');
  console.log('========================');
  
  try {
    // Create and train network
    console.log('\n1. Creating and training network...');
    const originalNetwork = new OptimizedNeuralNetwork({
      layers: [2, 4, 1],
      learningRate: 0.1,
      outputActivation: 'sigmoid',
      lossFunction: 'mse'
    });
    
    originalNetwork.train(xorData, 50, { verbose: false });
    
    // Get original predictions
    console.log('\n2. Getting original predictions...');
    const originalPredictions = xorData.map(data => originalNetwork.predict(data.input));
    
    // Get original weights for comparison
    console.log('\n3. Extracting original weights...');
    const originalWeights = originalNetwork.getAllWeightsAsFloat32();
    const originalBiases = originalNetwork.getAllBiasesAsFloat32();
    
    // Save network
    console.log('\n4. Saving network...');
    originalNetwork.save('test-fix.bin');
    
    // Load network
    console.log('\n5. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('test-fix.bin');
    
    // Get loaded predictions
    console.log('\n6. Getting loaded predictions...');
    const loadedPredictions = xorData.map(data => loadedNetwork.predict(data.input));
    
    // Get loaded weights for comparison
    console.log('\n7. Extracting loaded weights...');
    const loadedWeights = loadedNetwork.getAllWeightsAsFloat32();
    const loadedBiases = loadedNetwork.getAllBiasesAsFloat32();
    
    // Compare predictions
    console.log('\n8. COMPARING PREDICTIONS:');
    console.log('Original vs Loaded');
    let predictionsMatch = true;
    for (let i = 0; i < xorData.length; i++) {
      const orig = originalPredictions[i][0];
      const loaded = loadedPredictions[i][0];
      const diff = Math.abs(orig - loaded);
      const match = diff < 1e-6;
      predictionsMatch = predictionsMatch && match;
      
      console.log(`Example ${i + 1}: ${orig.toFixed(6)} vs ${loaded.toFixed(6)} (diff: ${diff.toFixed(8)}) ${match ? '✓' : '✗'}`);
    }
    
    // Compare weights layer by layer
    console.log('\n9. COMPARING WEIGHTS LAYER BY LAYER:');
    let weightsMatch = true;
    let weightOffset = 0;
    for (let layerIdx = 0; layerIdx < originalNetwork.layers.length; layerIdx++) {
      const layer = originalNetwork.layers[layerIdx];
      const loadedLayer = loadedNetwork.layers[layerIdx];
      console.log(`Layer ${layerIdx} (${layer.inputSize} -> ${layer.outputSize}):`);
      
      for (let i = 0; i < layer.weights.length; i++) {
        const origWeight = layer.weights[i];
        const loadedWeight = loadedLayer.weights[i];
        const diff = Math.abs(origWeight - loadedWeight);
        if (diff > 1e-6) {
          weightsMatch = false;
          if (weightOffset + i < 10) { // Show first few mismatches
            console.log(`  Weight[${weightOffset + i}]: ${origWeight.toFixed(6)} vs ${loadedWeight.toFixed(6)} (diff: ${diff.toFixed(8)}) ✗`);
          }
        }
      }
      weightOffset += layer.weights.length;
    }
    console.log(`Weights match: ${weightsMatch ? '✓' : '✗'}`);
    
    // Compare biases layer by layer
    console.log('\n10. COMPARING BIASES LAYER BY LAYER:');
    let biasesMatch = true;
    for (let layerIdx = 0; layerIdx < originalNetwork.layers.length; layerIdx++) {
      const layer = originalNetwork.layers[layerIdx];
      const loadedLayer = loadedNetwork.layers[layerIdx];
      if (layer.biases) {
        console.log(`Layer ${layerIdx} biases:`);
        for (let i = 0; i < layer.biases.length; i++) {
          const origBias = layer.biases[i];
          const loadedBias = loadedLayer.biases[i];
          const diff = Math.abs(origBias - loadedBias);
          if (diff > 1e-6) {
            biasesMatch = false;
            console.log(`  Bias[${i}]: ${origBias.toFixed(6)} vs ${loadedBias.toFixed(6)} (diff: ${diff.toFixed(8)}) ✗`);
          }
        }
      }
    }
    console.log(`Biases match: ${biasesMatch ? '✓' : '✗'}`);
    
    // Final result
    console.log('\n11. FINAL RESULT:');
    const allMatch = predictionsMatch && weightsMatch && biasesMatch;
    console.log(allMatch ? '✅ SAVE/LOAD FIX SUCCESSFUL!' : '❌ SAVE/LOAD FIX FAILED!');
    
    // Clean up
    try {
      const fs = await import('fs');
      fs.unlinkSync('test-fix.bin');
    } catch (e) {
      // Ignore cleanup errors
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
  }
}

// Run test
testSaveLoadFix();