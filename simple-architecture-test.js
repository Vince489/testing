/**
 * Simple test to check architecture preservation
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function simpleArchitectureTest() {
  console.log('🔍 SIMPLE ARCHITECTURE TEST');
  console.log('===========================');
  
  try {
    // Create network with [2, 4, 1] architecture
    console.log('\n1. Creating network with [2, 4, 1] architecture...');
    const originalNetwork = new OptimizedNeuralNetwork({
      layers: [2, 4, 1],
      learningRate: 0.1,
      outputActivation: 'sigmoid',
      lossFunction: 'mse'
    });
    
    console.log('Original network layers:', originalNetwork.layers.length);
    console.log('Original layer configs:', originalNetwork.layerConfigs);
    console.log('Original layers sizes:', originalNetwork.layers.map(l => `${l.inputSize} -> ${l.outputSize}`));
    
    // Save network
    console.log('\n2. Saving network...');
    originalNetwork.save('simple-test.bin');
    
    // Load network
    console.log('\n3. Loading network...');
    const loadedNetwork = await OptimizedNeuralNetwork.load('simple-test.bin');
    
    console.log('Loaded network layers:', loadedNetwork.layers.length);
    console.log('Loaded layer configs:', loadedNetwork.layerConfigs);
    console.log('Loaded layers sizes:', loadedNetwork.layers.map(l => `${l.inputSize} -> ${l.outputSize}`));
    
    // Test with simple input
    console.log('\n4. Testing with simple input...');
    const input = [1.0, 0.0];
    const originalOutput = originalNetwork.predict(input);
    const loadedOutput = loadedNetwork.predict(input);
    
    console.log('Input:', input);
    console.log('Original output:', originalOutput);
    console.log('Loaded output:', loadedOutput);
    console.log('Outputs match:', Math.abs(originalOutput[0] - loadedOutput[0]) < 1e-6);
    
    // Clean up
    const fs = await import('fs');
    fs.unlinkSync('simple-test.bin');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
  }
}

// Run test
simpleArchitectureTest();