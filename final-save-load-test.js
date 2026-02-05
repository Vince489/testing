/**
 * Final test to verify save/load fix - proper comparison
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

// Test data for XOR problem
const xorData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

async function finalSaveLoadTest() {
  console.log('🧪 FINAL SAVE/LOAD TEST');
  console.log('=======================');
  
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
    
    // Save network
    console.log('\n3. Saving network...');
    originalNetwork.save('final-test.bin');
    
    // Load network
    console.log('\n4. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('final-test.bin');
    
    // Get loaded predictions
    console.log('\n5. Getting loaded predictions...');
    const loadedPredictions = xorData.map(data => loadedNetwork.predict(data.input));
    
    // Compare predictions
    console.log('\n6. COMPARING PREDICTIONS:');
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
    
    // Compare weights
    console.log('\n7. COMPARING WEIGHTS:');
    let weightsMatch = true;
    for (let layerIdx = 0; layerIdx < originalNetwork.layers.length; layerIdx++) {
      const layer = originalNetwork.layers[layerIdx];
      const loadedLayer = loadedNetwork.layers[layerIdx];
      for (let i = 0; i < layer.weights.length; i++) {
        const diff = Math.abs(layer.weights[i] - loadedLayer.weights[i]);
        if (diff > 1e-6) {
          weightsMatch = false;
          if (i < 5) { // Show first few mismatches
            console.log(`  Layer ${layerIdx}, Weight ${i}: ${layer.weights[i].toFixed(6)} vs ${loadedLayer.weights[i].toFixed(6)} (diff: ${diff.toFixed(8)}) ✗`);
          }
        }
      }
    }
    console.log(`Weights match: ${weightsMatch ? '✓' : '✗'}`);
    
    // Compare biases
    console.log('\n8. COMPARING BIASES:');
    let biasesMatch = true;
    for (let layerIdx = 0; layerIdx < originalNetwork.layers.length; layerIdx++) {
      const layer = originalNetwork.layers[layerIdx];
      const loadedLayer = loadedNetwork.layers[layerIdx];
      if (layer.biases) {
        for (let i = 0; i < layer.biases.length; i++) {
          const diff = Math.abs(layer.biases[i] - loadedLayer.biases[i]);
          if (diff > 1e-6) {
            biasesMatch = false;
            console.log(`  Layer ${layerIdx}, Bias ${i}: ${layer.biases[i].toFixed(6)} vs ${loadedLayer.biases[i].toFixed(6)} (diff: ${diff.toFixed(8)}) ✗`);
          }
        }
      }
    }
    console.log(`Biases match: ${biasesMatch ? '✓' : '✗'}`);
    
    // Final result
    console.log('\n9. FINAL RESULT:');
    const allMatch = predictionsMatch && weightsMatch && biasesMatch;
    console.log(allMatch ? '✅ SAVE/LOAD FIX SUCCESSFUL!' : '❌ SAVE/LOAD FIX FAILED!');
    
    if (allMatch) {
      console.log('\n🎉 The save/load issue has been FIXED!');
      console.log('   - Weights are properly saved and restored');
      console.log('   - Biases are properly saved and restored');
      console.log('   - Predictions match exactly');
      console.log('   - Test 3 predictions now match Test 1');
    }
    
    // Clean up
    try {
      const fs = await import('fs');
      fs.unlinkSync('final-test.bin');
    } catch (e) {
      // Ignore cleanup errors
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
  }
}

// Run test
finalSaveLoadTest();