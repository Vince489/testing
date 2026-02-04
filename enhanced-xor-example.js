import { NeuralNetwork } from './neural-network-2.js';

// Example usage for XOR problem with enhanced features
console.log("=== ENHANCED NEURAL NETWORK EXAMPLE ===");

const enhancedConfig = {
  layers: [2, 4, 1],           // More hidden neurons
  learningRate: 0.1,          // Lower learning rate
  activation: 'tanh',         // Better activation function
  outputActivation: 'sigmoid',
  momentum: 0.9,              // Add momentum
  regularization: 'l2',       // Add regularization
  regularizationRate: 0.01,
  dropoutRate: 0.1,           // Add dropout
  batchSize: 1,               // Online learning
  verbose: true
};

const enhancedNN = new NeuralNetwork(enhancedConfig);

const xorTrainingData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

console.log("Training enhanced XOR neural network...");
enhancedNN.train(xorTrainingData, 5000, {
  validationData: xorTrainingData,
  earlyStopping: true,
  patience: 20
});

console.log("\nEnhanced XOR Predictions:");
enhancedNN.test(xorTrainingData);

// Save the trained network
enhancedNN.save('enhanced_xor_network.json');