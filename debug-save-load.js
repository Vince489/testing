/**
 * Debug save/load issue step by step
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function debugSaveLoad() {
  console.log('🔍 DEBUGGING SAVE/LOAD ISSUE');
  console.log('============================');
  
  try {
    // Create a simple network
    console.log('\n1. Creating simple network...');
    const network = new OptimizedNeuralNetwork({
      layers: [2, 2, 1],
      learningRate: 0.1,
      outputActivation: 'sigmoid',
      lossFunction: 'mse'
    });
    
    // Set some known weights for debugging
    console.log('\n2. Setting known weights...');
    network.layers[0].weights[0] = 1.0;  // w00
    network.layers[0].weights[1] = 2.0;  // w01  
    network.layers[0].weights[2] = 3.0;  // w10
    network.layers[0].weights[3] = 4.0;  // w11
    network.layers[1].weights[0] = 5.0;  // w0
    network.layers[1].weights[1] = 6.0;  // w1
    
    network.layers[0].biases[0] = 0.1;   // b0
    network.layers[0].biases[1] = 0.2;   // b1
    network.layers[1].biases[0] = 0.3;   // b
    
    console.log('Layer 0 weights:', Array.from(network.layers[0].weights));
    console.log('Layer 0 biases:', Array.from(network.layers[0].biases));
    console.log('Layer 1 weights:', Array.from(network.layers[1].weights));
    console.log('Layer 1 biases:', Array.from(network.layers[1].biases));
    
    // Test forward pass
    console.log('\n3. Testing forward pass...');
    const input = [1.0, 0.0];
    const output = network.predict(input);
    console.log('Input:', input);
    console.log('Output:', output);
    
    // Save network
    console.log('\n4. Saving network...');
    network.save('debug.bin');
    
    // Load network
    console.log('\n5. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('debug.bin');
    
    // Check loaded weights
    console.log('\n6. Checking loaded weights...');
    console.log('Loaded Layer 0 weights:', Array.from(loadedNetwork.layers[0].weights));
    console.log('Loaded Layer 0 biases:', Array.from(loadedNetwork.layers[0].biases));
    console.log('Loaded Layer 1 weights:', Array.from(loadedNetwork.layers[1].weights));
    console.log('Loaded Layer 1 biases:', Array.from(loadedNetwork.layers[1].biases));
    
    // Test loaded forward pass
    console.log('\n7. Testing loaded forward pass...');
    const loadedOutput = loadedNetwork.predict(input);
    console.log('Input:', input);
    console.log('Loaded Output:', loadedOutput);
    
    // Compare
    console.log('\n8. COMPARISON:');
    const weightMatch = JSON.stringify(Array.from(network.layers[0].weights)) === JSON.stringify(Array.from(loadedNetwork.layers[0].weights)) &&
                       JSON.stringify(Array.from(network.layers[1].weights)) === JSON.stringify(Array.from(loadedNetwork.layers[1].weights));
    const biasMatch = JSON.stringify(Array.from(network.layers[0].biases)) === JSON.stringify(Array.from(loadedNetwork.layers[0].biases)) &&
                     JSON.stringify(Array.from(network.layers[1].biases)) === JSON.stringify(Array.from(loadedNetwork.layers[1].biases));
    const outputMatch = Math.abs(output[0] - loadedOutput[0]) < 1e-6;
    
    console.log('Weights match:', weightMatch);
    console.log('Biases match:', biasMatch);
    console.log('Outputs match:', outputMatch);
    
    // Clean up
    try {
      const fs = await import('fs');
      fs.unlinkSync('debug.bin');
    } catch (e) {
      // Ignore cleanup errors
    }
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    console.error(error.stack);
  }
}

// Run debug
debugSaveLoad();