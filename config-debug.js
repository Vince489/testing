/**
 * Debug configuration loading
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function configDebug() {
  console.log('🔍 DEBUGGING CONFIGURATION LOADING');
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
    
    originalNetwork.train([{ input: [0, 0], output: [0] }], 1, { verbose: false });
    
    // Save network
    console.log('\n2. Saving network...');
    originalNetwork.save('config-debug.bin');
    
    // Load network and inspect config
    console.log('\n3. Loading network and inspecting config...');
    const fs = await import('fs');
    const buffer = fs.readFileSync('config-debug.bin');
    let offset = 0;
    
    // Read header
    const configSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('Config size:', configSize);
    
    // Read config data
    const configData = buffer.toString('utf8', offset, offset + configSize);
    console.log('Raw config data:', configData);
    
    const config = JSON.parse(configData);
    console.log('Parsed config:', config);
    console.log('Layer configs:', config.layerConfigs);
    
    // Now load the actual network
    const loadedNetwork = await OptimizedNeuralNetwork.load('config-debug.bin');
    console.log('Loaded network layer configs:', loadedNetwork.layerConfigs);
    console.log('Loaded network layers count:', loadedNetwork.layers.length);
    console.log('Loaded network layers sizes:', loadedNetwork.layers.map(l => `${l.inputSize} -> ${l.outputSize}`));
    
    // Clean up
    fs.unlinkSync('config-debug.bin');
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    console.error(error.stack);
  }
}

// Run debug
configDebug();