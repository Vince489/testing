/**
 * Detailed debug to see exactly what's happening with weights
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function detailedDebug() {
  console.log('🔍 DETAILED DEBUGGING SAVE/LOAD ISSUE');
  console.log('====================================');
  
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
    // Layer 0: 2 inputs -> 2 outputs (4 weights total)
    network.layers[0].weights[0] = 1.0;  // w00 (input 0 -> output 0)
    network.layers[0].weights[1] = 2.0;  // w01 (input 0 -> output 1)  
    network.layers[0].weights[2] = 3.0;  // w10 (input 1 -> output 0)
    network.layers[0].weights[3] = 4.0;  // w11 (input 1 -> output 1)
    
    // Layer 1: 2 inputs -> 1 output (2 weights total)
    network.layers[1].weights[0] = 5.0;  // w0 (input 0 -> output 0)
    network.layers[1].weights[1] = 6.0;  // w1 (input 1 -> output 0)
    
    network.layers[0].biases[0] = 0.1;   // b0
    network.layers[0].biases[1] = 0.2;   // b1
    network.layers[1].biases[0] = 0.3;   // b
    
    console.log('Original Layer 0 weights:', Array.from(network.layers[0].weights));
    console.log('Original Layer 0 biases:', Array.from(network.layers[0].biases));
    console.log('Original Layer 1 weights:', Array.from(network.layers[1].weights));
    console.log('Original Layer 1 biases:', Array.from(network.layers[1].biases));
    
    // Test forward pass
    console.log('\n3. Testing forward pass...');
    const input = [1.0, 0.0];
    const output = network.predict(input);
    console.log('Input:', input);
    console.log('Output:', output);
    
    // Extract all weights as arrays
    console.log('\n4. Extracting all weights...');
    const allWeights = network.getAllWeightsAsFloat32();
    const allBiases = network.getAllBiasesAsFloat32();
    console.log('All weights array:', Array.from(allWeights));
    console.log('All biases array:', Array.from(allBiases));
    
    // Save network
    console.log('\n5. Saving network...');
    network.save('detailed-debug.bin');
    
    // Load network
    console.log('\n6. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('detailed-debug.bin');
    
    // Check loaded weights
    console.log('\n7. Checking loaded weights...');
    console.log('Loaded Layer 0 weights:', Array.from(loadedNetwork.layers[0].weights));
    console.log('Loaded Layer 0 biases:', Array.from(loadedNetwork.layers[0].biases));
    console.log('Loaded Layer 1 weights:', Array.from(loadedNetwork.layers[1].weights));
    console.log('Loaded Layer 1 biases:', Array.from(loadedNetwork.layers[1].biases));
    
    // Extract loaded weights as arrays
    console.log('\n8. Extracting loaded weights...');
    const loadedAllWeights = loadedNetwork.getAllWeightsAsFloat32();
    const loadedAllBiases = loadedNetwork.getAllBiasesAsFloat32();
    console.log('Loaded all weights array:', Array.from(loadedAllWeights));
    console.log('Loaded all biases array:', Array.from(loadedAllBiases));
    
    // Test loaded forward pass
    console.log('\n9. Testing loaded forward pass...');
    const loadedOutput = loadedNetwork.predict(input);
    console.log('Input:', input);
    console.log('Loaded Output:', loadedOutput);
    
    // Compare
    console.log('\n10. COMPARISON:');
    const weightMatch = JSON.stringify(Array.from(allWeights)) === JSON.stringify(Array.from(loadedAllWeights));
    const biasMatch = JSON.stringify(Array.from(allBiases)) === JSON.stringify(Array.from(loadedAllBiases));
    const outputMatch = Math.abs(output[0] - loadedOutput[0]) < 1e-6;
    
    console.log('All weights match:', weightMatch);
    console.log('All biases match:', biasMatch);
    console.log('Outputs match:', outputMatch);
    
    // Clean up
    try {
      const fs = await import('fs');
      fs.unlinkSync('detailed-debug.bin');
    } catch (e) {
      // Ignore cleanup errors
    }
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    console.error(error.stack);
  }
}

// Run debug
detailedDebug();