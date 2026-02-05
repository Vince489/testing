/**
 * Debug JSON parsing in the load method
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function jsonDebug() {
  console.log('🔍 DEBUGGING JSON PARSING IN LOAD METHOD');
  console.log('=========================================');
  
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
    originalNetwork.save('json-debug.bin');
    
    // Manually load and parse the config
    console.log('\n3. Manually loading and parsing config...');
    const fs = await import('fs');
    const buffer = fs.readFileSync('json-debug.bin');
    let offset = 32; // Skip header
    
    const configSize = buffer.readUInt32LE(0);
    console.log('Config size from header:', configSize);
    
    const configData = buffer.toString('utf8', offset, offset + configSize);
    console.log('Raw config data length:', configData.length);
    console.log('Raw config data:', configData);
    
    try {
      const config = JSON.parse(configData);
      console.log('Parsed config:', config);
      console.log('Layer configs from parsed config:', config.layerConfigs);
      
      // Now test creating a network with this config
      console.log('\n4. Creating network with parsed config...');
      const testNetwork = new OptimizedNeuralNetwork(config);
      console.log('Test network layer configs:', testNetwork.layerConfigs);
      console.log('Test network layers count:', testNetwork.layers.length);
      console.log('Test network layers sizes:', testNetwork.layers.map(l => `${l.inputSize} -> ${l.outputSize}`));
      
    } catch (e) {
      console.log('JSON parse error:', e.message);
    }
    
    // Clean up
    fs.unlinkSync('json-debug.bin');
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    console.error(error.stack);
  }
}

// Run debug
jsonDebug();