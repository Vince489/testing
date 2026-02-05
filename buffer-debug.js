/**
 * Debug buffer reading in detail
 */

import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

async function bufferDebug() {
  console.log('🔍 DEBUGGING BUFFER READING IN DETAIL');
  console.log('=====================================');
  
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
    originalNetwork.save('buffer-debug.bin');
    
    // Load network and inspect buffer in detail
    console.log('\n3. Loading network and inspecting buffer...');
    const fs = await import('fs');
    const buffer = fs.readFileSync('buffer-debug.bin');
    console.log('Total buffer length:', buffer.length);
    
    let offset = 0;
    console.log('\n--- HEADER ---');
    console.log('Offset:', offset);
    const configSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('Config size:', configSize);
    
    console.log('Offset:', offset);
    const weightSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('Weight size:', weightSize);
    
    console.log('Offset:', offset);
    const biasSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('Bias size:', biasSize);
    
    console.log('Offset:', offset);
    const mWeightSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('mWeight size:', mWeightSize);
    
    console.log('Offset:', offset);
    const vWeightSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('vWeight size:', vWeightSize);
    
    console.log('Offset:', offset);
    const mBiasSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('mBias size:', mBiasSize);
    
    console.log('Offset:', offset);
    const vBiasSize = buffer.readUInt32LE(offset); offset += 4;
    console.log('vBias size:', vBiasSize);
    
    console.log('Offset:', offset);
    const layerCount = buffer.readUInt32LE(offset); offset += 4;
    console.log('Layer count:', layerCount);
    
    console.log('\n--- CONFIG DATA ---');
    console.log('Offset:', offset);
    console.log('Config size:', configSize);
    console.log('Remaining buffer length:', buffer.length - offset);
    
    if (configSize > buffer.length - offset) {
      console.log('ERROR: Config size exceeds remaining buffer!');
      console.log('Config size:', configSize);
      console.log('Remaining buffer:', buffer.length - offset);
    } else {
      const configData = buffer.toString('utf8', offset, offset + configSize);
      console.log('Config data length:', configData.length);
      console.log('Config data:', configData);
      
      try {
        const config = JSON.parse(configData);
        console.log('Parsed config:', config);
        console.log('Layer configs:', config.layerConfigs);
      } catch (e) {
        console.log('JSON parse error:', e.message);
      }
    }
    
    // Clean up
    fs.unlinkSync('buffer-debug.bin');
    
  } catch (error) {
    console.error('❌ Debug failed:', error.message);
    console.error(error.stack);
  }
}

// Run debug
bufferDebug();