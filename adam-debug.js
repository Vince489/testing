/**
 * Debug Adam optimizer state preservation
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function adamDebug() {
  console.log('🔍 DEBUGGING ADAM STATE PRESERVATION');
  console.log('=====================================');
  
  try {
    // Create a simple network
    console.log('\n1. Creating simple network...');
    const network = new OptimizedNeuralNetwork({
      layers: [2, 2, 1],
      learningRate: 0.1,
      outputActivation: 'sigmoid',
      lossFunction: 'mse'
    });
    
    // Set some known weights and Adam state for debugging
    console.log('\n2. Setting known weights and Adam state...');
    // Layer 0: 2 inputs -> 2 outputs (4 weights total)
    network.layers[0].weights[0] = 1.0;  // w00
    network.layers[0].weights[1] = 2.0;  // w01  
    network.layers[0].weights[2] = 3.0;  // w10
    network.layers[0].weights[3] = 4.0;  // w11
    
    // Layer 1: 2 inputs -> 1 output (2 weights total)
    network.layers[1].weights[0] = 5.0;  // w0
    network.layers[1].weights[1] = 6.0;  // w1
    
    network.layers[0].biases[0] = 0.1;   // b0
    network.layers[0].biases[1] = 0.2;   // b1
    network.layers[1].biases[0] = 0.3;   // b
    
    // Set some Adam state
    network.layers[0].mWeights[0] = 0.1;  // m00
    network.layers[0].mWeights[1] = 0.2;  // m01
    network.layers[0].mWeights[2] = 0.3;  // m10
    network.layers[0].mWeights[3] = 0.4;  // m11
    network.layers[1].mWeights[0] = 0.5;  // m0
    network.layers[1].mWeights[1] = 0.6;  // m1
    
    network.layers[0].vWeights[0] = 0.01; // v00
    network.layers[0].vWeights[1] = 0.02; // v01
    network.layers[0].vWeights[2] = 0.03; // v10
    network.layers[0].vWeights[3] = 0.04; // v11
    network.layers[1].vWeights[0] = 0.05; // v0
    network.layers[1].vWeights[1] = 0.06; // v1
    
    console.log('Original Layer 0 weights:', Array.from(network.layers[0].weights));
    console.log('Original Layer 0 mWeights:', Array.from(network.layers[0].mWeights));
    console.log('Original Layer 0 vWeights:', Array.from(network.layers[0].vWeights));
    console.log('Original Layer 1 weights:', Array.from(network.layers[1].weights));
    console.log('Original Layer 1 mWeights:', Array.from(network.layers[1].mWeights));
    console.log('Original Layer 1 vWeights:', Array.from(network.layers[1].vWeights));
    
    // Test forward pass
    console.log('\n3. Testing forward pass...');
    const input = [1.0, 0.0];
    const output = network.predict(input);
    console.log('Input:', input);
    console.log('Output:', output);
    
    // Save network
    console.log('\n4. Saving network...');
    network.save('adam-debug.bin');
    
    // Load network
    console.log('\n5. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('adam-debug.bin');
    
    // Check loaded weights and Adam state
    console.log('\n6. Checking loaded weights and Adam state...');
    console.log('Loaded Layer 0 weights:', Array.from(loadedNetwork.layers[0].weights));
    console.log('Loaded Layer 0 mWeights:', Array.from(loadedNetwork.layers[0].mWeights));
    console.log('Loaded Layer 0 vWeights:', Array.from(loadedNetwork.layers[0].vWeights));
    console.log('Loaded Layer 1 weights:', Array.from(loadedNetwork.layers[1].weights));
    console.log('Loaded Layer 1 mWeights:', Array.from(loadedNetwork.layers[1].mWeights));
    console.log('Loaded Layer 1 vWeights:', Array.from(loadedNetwork.layers[1].vWeights));
    
    // Test loaded forward pass
    console.log('\n7. Testing loaded forward pass...');
    const loadedOutput = loadedNetwork.predict(input);
    console.log('Input:', input);
    console.log('Loaded Output:', loadedOutput);
    
    // Compare
    console.log('\n8. COMPARISON:');
    const weightMatch = JSON.stringify(Array.from(network.layers[0].weights)) === JSON.stringify(Array.from(loadedNetwork.layers[0].weights)) &&
                       JSON.stringify(Array.from(network.layers[1].weights)) === JSON.stringify(Array.from(loadedNetwork.layers[1].weights));
    const mWeightMatch = JSON.stringify(Array.from(network.layers[0].mWeights)) === JSON.stringify(Array.from(loadedNetwork.layers[0].mWeights)) &&
                        JSON.stringify(Array.from(network.layers[1].mWeights)) === JSON.stringify(Array.from(loadedNetwork.layers[1].mWeights));
    const vWeightMatch = JSON.stringify(Array.from(network.layers[0].vWeights)) === JSON.stringify(Array.from(loadedNetwork.layers[0].vWeights)) &&
                        JSON.stringify(Array.from(network.layers[1].vWeights)) === JSON.stringify(Array.from(loadedNetwork.layers[1].vWeights));
    const outputMatch = Math.abs(output[0] - loadedOutput[0]) < 1e-6;
    
    console.log('Weights match:', weightMatch);
    console.log('mWeights match:', mWeightMatch);
    console.log('vWeights match:', vWeightMatch);
    console.log('Outputs match:', outputMatch);
    
    // Clean up
    try {
      const fs = await import('fs');
      fs.unlinkSync('adam-debug.bin');
    } catch (e) {
      // Ignore cleanup errors
    }
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    console.error(error.stack);
  }
}

// Run debug
adamDebug();