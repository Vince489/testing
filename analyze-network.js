import fs from 'fs';

// Load the trained network data
const networkData = JSON.parse(fs.readFileSync('goals_embedding_network.json', 'utf8'));

// Analyze the network configuration
console.log('=== NETWORK CONFIGURATION ===');
console.log('Layers:', networkData.config.layers);
console.log('Learning Rate:', networkData.config.learningRate);
console.log('Activation Function:', networkData.config.activation);
console.log('Output Activation Function:', networkData.config.outputActivation);
console.log('Momentum:', networkData.config.momentum);
console.log('Regularization:', networkData.config.regularization);
console.log('Regularization Rate:', networkData.config.regularizationRate);
console.log('Dropout Rate:', networkData.config.dropoutRate);

// Analyze the weights
console.log('\n=== WEIGHTS ANALYSIS ===');
for (let i = 0; i < networkData.weights.length; i++) {
  console.log(`\nLayer ${i} to ${i+1} weights:`);

  // Check for extreme values
  let minVal = Infinity;
  let maxVal = -Infinity;
  let zeroCount = 0;
  let nearOneCount = 0; // Values close to 1 or -1
  let totalCount = 0;

  // Handle object-based structure
  for (let j = 0; j < networkData.weights[i].length; j++) {
    const weightObject = networkData.weights[i][j];
    for (const key in weightObject) {
      const val = weightObject[key];
      minVal = Math.min(minVal, val);
      maxVal = Math.max(maxVal, val);
      totalCount++;

      if (Math.abs(val) < 0.0001) {
        zeroCount++;
      }

      if (Math.abs(Math.abs(val) - 1) < 0.0001) {
        nearOneCount++;
      }
    }
  }

  console.log(`  Min value: ${minVal.toFixed(6)}`);
  console.log(`  Max value: ${maxVal.toFixed(6)}`);
  console.log(`  Zero or near-zero values: ${zeroCount} (${totalCount > 0 ? (zeroCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
  console.log(`  Values near 1 or -1: ${nearOneCount} (${totalCount > 0 ? (nearOneCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
}

// Analyze the biases
console.log('\n=== BIASES ANALYSIS ===');
for (let i = 0; i < networkData.biases.length; i++) {
  console.log(`\nLayer ${i+1} biases:`);

  let minVal = Infinity;
  let maxVal = -Infinity;
  let zeroCount = 0;
  let nearOneCount = 0;
  let totalCount = 0;

  // Handle object-based structure
  const biasObject = networkData.biases[i];
  for (const key in biasObject) {
    const val = biasObject[key];
    minVal = Math.min(minVal, val);
    maxVal = Math.max(maxVal, val);
    totalCount++;

    if (Math.abs(val) < 0.0001) {
      zeroCount++;
    }

    if (Math.abs(Math.abs(val) - 1) < 0.0001) {
      nearOneCount++;
    }
  }

  console.log(`  Min value: ${minVal.toFixed(6)}`);
  console.log(`  Max value: ${maxVal.toFixed(6)}`);
  console.log(`  Zero or near-zero values: ${zeroCount} (${totalCount > 0 ? (zeroCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
  console.log(`  Values near 1 or -1: ${nearOneCount} (${totalCount > 0 ? (nearOneCount/totalCount*100).toFixed(2) : 'N/A'}%)`);
}

console.log('\nAnalysis complete!');