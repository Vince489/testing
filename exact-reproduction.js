/**
 * Exact reproduction of the final test with debugging
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

// Test data for XOR problem
const xorData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

async function exactReproduction() {
  console.log('🧪 EXACT REPRODUCTION OF FINAL TEST');
  console.log('==================================');
  
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
    console.log('Original predictions:', originalPredictions.map(p => p[0].toFixed(6)));
    
    // Get original weights for comparison
    console.log('\n3. Extracting original weights...');
    const originalWeights = originalNetwork.getAllWeightsAsFloat32();
    const originalBiases = originalNetwork.getAllBiasesAsFloat32();
    console.log('Original weights length:', originalWeights.length);
    console.log('Original biases length:', originalBiases.length);
    console.log('First few weights:', Array.from(originalWeights).slice(0, 5).map(w => w.toFixed(6)));
    console.log('First few biases:', Array.from(originalBiases).slice(0, 5).map(b => b.toFixed(6)));
    
    // Save network
    console.log('\n4. Saving network...');
    originalNetwork.save('exact-reproduction.bin');
    
    // Load network
    console.log('\n5. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('exact-reproduction.bin');
    
    // Get loaded weights for comparison
    console.log('\n6. Extracting loaded weights...');
    const loadedWeights = loadedNetwork.getAllWeightsAsFloat32();
    const loadedBiases = loadedNetwork.getAllBiasesAsFloat32();
    console.log('Loaded weights length:', loadedWeights.length);
    console.log('Loaded biases length:', loadedBiases.length);
    console.log('First few loaded weights:', Array.from(loadedWeights).slice(0, 5).map(w => w.toFixed(6)));
    console.log('First few loaded biases:', Array.from(loadedBiases).slice(0, 5).map(b => b.toFixed(6)));
    
    // Compare weights and biases
    console.log('\n7. Comparing weights and biases...');
    const weightMatch = JSON.stringify(Array.from(originalWeights)) === JSON.stringify(Array.from(loadedWeights));
    const biasMatch = JSON.stringify(Array.from(originalBiases)) === JSON.stringify(Array.from(loadedBiases));
    console.log('Weights match:', weightMatch);
    console.log('Biases match:', biasMatch);
    
    if (!weightMatch) {
      console.log('Weight differences:');
      for (let i = 0; i < Math.min(originalWeights.length, loadedWeights.length); i++) {
        const diff = Math.abs(originalWeights[i] - loadedWeights[i]);
        if (diff > 1e-6) {
          console.log(`  Weight[${i}]: ${originalWeights[i].toFixed(6)} vs ${loadedWeights[i].toFixed(6)} (diff: ${diff.toFixed(8)})`);
          if (i > 10) break; // Don't show too many
        }
      }
    }
    
    if (!biasMatch) {
      console.log('Bias differences:');
      for (let i = 0; i < Math.min(originalBiases.length, loadedBiases.length); i++) {
        const diff = Math.abs(originalBiases[i] - loadedBiases[i]);
        if (diff > 1e-6) {
          console.log(`  Bias[${i}]: ${originalBiases[i].toFixed(6)} vs ${loadedBiases[i].toFixed(6)} (diff: ${diff.toFixed(8)})`);
          if (i > 10) break; // Don't show too many
        }
      }
    }
    
    // Get loaded predictions
    console.log('\n8. Getting loaded predictions...');
    const loadedPredictions = xorData.map(data => loadedNetwork.predict(data.input));
    console.log('Loaded predictions:', loadedPredictions.map(p => p[0].toFixed(6)));
    
    // Compare predictions
    console.log('\n9. COMPARING PREDICTIONS:');
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
    
    // Final result
    console.log('\n10. FINAL RESULT:');
    const allMatch = predictionsMatch && weightMatch && biasMatch;
    console.log(allMatch ? '✅ SAVE/LOAD FIX SUCCESSFUL!' : '❌ SAVE/LOAD FIX FAILED!');
    
    // Clean up
    try {
      const fs = await import('fs');
      fs.unlinkSync('exact-reproduction.bin');
    } catch (e) {
      // Ignore cleanup errors
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
  }
}

// Run test
exactReproduction();