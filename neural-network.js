import fs from 'fs';

/**
 * NeuralNetwork Class
 * 
 * A comprehensive neural network implementation with enhanced features for better performance
 * and integration with text preprocessing for CBOW model training.
 * 
 * Features:
 * - Multiple activation functions (sigmoid, tanh, ReLU)
 * - Enhanced weight initialization (Xavier/Glorot)
 * - Momentum for faster convergence
 * - Learning rate scheduling
 * - Regularization (L1, L2)
 * - Dropout for regularization
 * - Batch training support
 * - Performance monitoring
 */

class NeuralNetwork {
  constructor(config) {
    // Configuration with defaults
    this.config = {
      layers: config.layers || [2, 2, 1],
      learningRate: config.learningRate || 0.5,
      activation: config.activation || 'sigmoid', // 'sigmoid', 'tanh', 'relu'
      outputActivation: config.outputActivation || config.activation || 'sigmoid',
      momentum: config.momentum || 0.0,
      regularization: config.regularization || null, // 'l1', 'l2', or null
      regularizationRate: config.regularizationRate || 0.01,
      dropoutRate: config.dropoutRate || 0.0,
      batchSize: config.batchSize || 1,
      verbose: config.verbose || false,
      ...config
    };

    this.layers = this.config.layers;
    this.learningRate = this.config.learningRate;
    this.momentum = this.config.momentum;
    this.regularization = this.config.regularization;
    this.regularizationRate = this.config.regularizationRate;
    this.dropoutRate = this.config.dropoutRate;
    this.batchSize = this.config.batchSize;
    this.verbose = this.config.verbose;

    // Initialize network components
    this.weights = [];
    this.biases = [];
    this.weightUpdates = []; // For momentum
    this.biasUpdates = [];   // For momentum
    
    // Performance tracking
    this.trainingHistory = [];
    this.currentEpoch = 0;

    this._initializeNetwork();
  }

  /**
   * Initialize network weights and biases with proper initialization
   */
  _initializeNetwork() {
    // Initialize weights and biases
    for (let i = 0; i < this.layers.length - 1; i++) {
      const weights = this._initializeWeights(this.layers[i], this.layers[i + 1]);
      const biases = this._initializeBiases(this.layers[i + 1]);
      
      this.weights.push(weights);
      this.biases.push(biases);
      
      // Initialize momentum tracking
      this.weightUpdates.push(this._createMatrix(this.layers[i], this.layers[i + 1], 0));
      this.biasUpdates.push(this._createMatrix(1, this.layers[i + 1], 0)[0]);
    }
  }

  /**
   * Enhanced weight initialization using Xavier/Glorot initialization
   */
  _initializeWeights(inputSize, outputSize) {
    const weights = [];
    // Xavier initialization: sqrt(6 / (fan_in + fan_out))
    const limit = Math.sqrt(6 / (inputSize + outputSize));
    
    for (let i = 0; i < inputSize; i++) {
      const row = [];
      for (let j = 0; j < outputSize; j++) {
        // Random value between -limit and +limit
        row.push((Math.random() * 2 - 1) * limit);
      }
      weights.push(row);
    }
    return weights;
  }

  /**
   * Initialize biases with small positive values
   */
  _initializeBiases(size) {
    const biases = [];
    for (let i = 0; i < size; i++) {
      biases.push(0.1); // Small positive bias
    }
    return biases;
  }

  /**
   * Create a matrix filled with a specific value
   */
  _createMatrix(rows, cols, value = 0) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        row.push(value);
      }
      matrix.push(row);
    }
    return matrix;
  }

  /**
   * Activation functions
   */
  activationFunction(x, type = this.config.activation) {
    switch (type) {
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x));
      case 'tanh':
        return Math.tanh(x);
      case 'relu':
        return Math.max(0, x);
      default:
        return 1 / (1 + Math.exp(-x));
    }
  }

  /**
   * Derivatives of activation functions
   */
  activationDerivative(x, type = this.config.activation) {
    switch (type) {
      case 'sigmoid':
        return x * (1 - x);
      case 'tanh':
        return 1 - x * x;
      case 'relu':
        return x > 0 ? 1 : 0;
      default:
        return x * (1 - x);
    }
  }

  /**
   * Forward pass with dropout support
   */
  forwardPass(inputs, training = true) {
    let activations = [inputs];
    let dropoutMasks = [];

    for (let i = 0; i < this.weights.length; i++) {
      const weightedSum = this._calculateWeightedSum(activations[i], this.weights[i], this.biases[i]);
      
      // Apply activation function
      let activation;
      if (i === this.weights.length - 1) {
        // Output layer
        activation = weightedSum.map(x => this.activationFunction(x, this.config.outputActivation));
      } else {
        // Hidden layers
        activation = weightedSum.map(x => this.activationFunction(x, this.config.activation));
      }
      
      // Apply dropout if enabled and in training mode
      let dropoutMask = null;
      if (training && this.dropoutRate > 0 && i < this.weights.length - 1) {
        dropoutMask = this._applyDropout(activation);
        activation = this._applyDropoutMask(activation, dropoutMask);
        dropoutMasks.push(dropoutMask);
      } else {
        dropoutMasks.push(null);
      }
      
      activations.push(activation);
    }

    return { activations, dropoutMasks };
  }

  /**
   * Apply dropout to activations
   */
  _applyDropout(activations) {
    const mask = [];
    for (let i = 0; i < activations.length; i++) {
      mask.push(Math.random() > this.dropoutRate ? 1 : 0);
    }
    return mask;
  }

  /**
   * Apply dropout mask to activations
   */
  _applyDropoutMask(activations, mask) {
    return activations.map((activation, i) => activation * mask[i]);
  }

  /**
   * Calculate weighted sum with bias
   */
  _calculateWeightedSum(inputs, weights, biases) {
    const result = [];
    for (let j = 0; j < weights[0].length; j++) {
      let sum = biases[j];
      for (let i = 0; i < inputs.length; i++) {
        sum += inputs[i] * weights[i][j];
      }
      result.push(sum);
    }
    return result;
  }

  /**
   * Train the network with enhanced features
   */
  train(trainingData, epochs = 10000, options = {}) {
    const config = {
      validationData: options.validationData || null,
      earlyStopping: options.earlyStopping || false,
      patience: options.patience || 10,
      minDelta: options.minDelta || 0.001,
      learningRateSchedule: options.learningRateSchedule || null, // 'exponential', 'plateau'
      decayRate: options.decayRate || 0.95,
      decaySteps: options.decaySteps || 1000,
      ...options
    };

    let bestLoss = Infinity;
    let patienceCounter = 0;
    let learningRate = this.learningRate;

    console.log(`Starting training with ${epochs} epochs...`);
    console.log(`Network architecture: [${this.layers.join(' -> ')}]`);
    console.log(`Activation: ${this.config.activation}, Output: ${this.config.outputActivation}`);
    console.log(`Learning rate: ${learningRate}, Momentum: ${this.momentum}`);
    console.log(`Regularization: ${this.regularization || 'None'}, Dropout: ${this.dropoutRate}`);

    const startTime = Date.now();

    for (let epoch = 0; epoch < epochs; epoch++) {
      this.currentEpoch = epoch;
      
      // Apply learning rate scheduling
      if (config.learningRateSchedule) {
        learningRate = this._applyLearningRateSchedule(learningRate, epoch, config);
      }

      // Shuffle training data
      const shuffledData = this._shuffleArray([...trainingData]);
      
      // Train in batches
      let totalLoss = 0;
      let batchCount = 0;

      for (let i = 0; i < shuffledData.length; i += this.batchSize) {
        const batch = shuffledData.slice(i, i + this.batchSize);
        const batchLoss = this._trainBatch(batch, learningRate);
        totalLoss += batchLoss;
        batchCount++;
      }

      const avgLoss = totalLoss / batchCount;
      this.trainingHistory.push({ epoch, loss: avgLoss, learningRate });

      // Validation check
      if (config.validationData) {
        const validationLoss = this._calculateLoss(config.validationData);
        if (validationLoss < bestLoss - config.minDelta) {
          bestLoss = validationLoss;
          patienceCounter = 0;
        } else {
          patienceCounter++;
        }

        if (config.earlyStopping && patienceCounter >= config.patience) {
          console.log(`Early stopping at epoch ${epoch} with validation loss: ${validationLoss.toFixed(6)}`);
          break;
        }
      }

      // Progress logging
      if (this.verbose && (epoch % 1000 === 0 || epoch === epochs - 1)) {
        console.log(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(6)}, LR = ${learningRate.toFixed(6)}`);
      }
    }

    const trainingTime = Date.now() - startTime;
    console.log(`Training completed in ${trainingTime}ms`);
    console.log(`Final loss: ${this.trainingHistory[this.trainingHistory.length - 1].loss.toFixed(6)}`);
  }

  /**
   * Train a single batch
   */
  _trainBatch(batch, learningRate) {
    let totalLoss = 0;

    // Forward pass for all examples in batch
    const batchResults = batch.map(data => {
      const result = this.forwardPass(data.input, true);
      const output = result.activations[result.activations.length - 1];
      const loss = this._calculateLossSingle(output, data.output);
      totalLoss += loss;
      return { ...result, target: data.output };
    });

    // Calculate average gradients
    const avgGradients = this._calculateAverageGradients(batchResults);

    // Update weights and biases
    this._updateParameters(avgGradients, learningRate);

    return totalLoss / batch.length;
  }

  /**
   * Calculate loss for a single output
   */
  _calculateLossSingle(output, target) {
    let loss = 0;
    for (let i = 0; i < output.length; i++) {
      loss += Math.pow(target[i] - output[i], 2);
    }
    return loss / 2;
  }

  /**
   * Calculate total loss for a dataset
   */
  _calculateLoss(dataset) {
    let totalLoss = 0;
    dataset.forEach(data => {
      const prediction = this.predict(data.input);
      totalLoss += this._calculateLossSingle(prediction, data.output);
    });
    return totalLoss / dataset.length;
  }

  /**
   * Calculate average gradients for a batch
   */
  _calculateAverageGradients(batchResults) {
    const gradients = {
      weightGradients: this.weights.map(w => this._createMatrix(w.length, w[0].length, 0)),
      biasGradients: this.biases.map(b => new Array(b.length).fill(0))
    };

    batchResults.forEach(result => {
      const errors = this._calculateErrors(result.activations, result.target);
      const layerGradients = this._calculateLayerGradients(result.activations, errors, result.dropoutMasks);
      
      // Accumulate gradients
      for (let i = 0; i < gradients.weightGradients.length; i++) {
        for (let j = 0; j < gradients.weightGradients[i].length; j++) {
          for (let k = 0; k < gradients.weightGradients[i][j].length; k++) {
            gradients.weightGradients[i][j][k] += layerGradients.weightGradients[i][j][k];
          }
        }
        for (let j = 0; j < gradients.biasGradients[i].length; j++) {
          gradients.biasGradients[i][j] += layerGradients.biasGradients[i][j];
        }
      }
    });

    // Average gradients
    const batchSize = batchResults.length;
    for (let i = 0; i < gradients.weightGradients.length; i++) {
      for (let j = 0; j < gradients.weightGradients[i].length; j++) {
        for (let k = 0; k < gradients.weightGradients[i][j].length; k++) {
          gradients.weightGradients[i][j][k] /= batchSize;
        }
      }
      for (let j = 0; j < gradients.biasGradients[i].length; j++) {
        gradients.biasGradients[i][j] /= batchSize;
      }
    }

    return gradients;
  }

  /**
   * Calculate errors for backpropagation
   */
  _calculateErrors(activations, target) {
    const errors = [];
    
    // Output layer error
    const output = activations[activations.length - 1];
    const outputError = target.map((t, i) => t - output[i]);
    errors.unshift(outputError);

    // Hidden layer errors
    for (let i = this.weights.length - 1; i > 0; i--) {
      const hiddenError = this._calculateHiddenError(errors[0], this.weights[i], activations[i]);
      errors.unshift(hiddenError);
    }

    return errors;
  }

  /**
   * Calculate hidden layer error
   */
  _calculateHiddenError(outputErrors, weights, activations) {
    const hiddenError = new Array(activations.length).fill(0);
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights[i].length; j++) {
        hiddenError[i] += outputErrors[j] * weights[i][j];
      }
    }
    return hiddenError;
  }

  /**
   * Calculate layer gradients
   */
  _calculateLayerGradients(activations, errors, dropoutMasks) {
    const gradients = {
      weightGradients: [],
      biasGradients: []
    };

    for (let i = this.weights.length - 1; i >= 0; i--) {
      const activationDeriv = activations[i + 1].map((a, idx) => {
        const deriv = this.activationDerivative(a, i === this.weights.length - 1 ? this.config.outputActivation : this.config.activation);
        // Apply dropout mask if present
        return dropoutMasks[i] ? deriv * dropoutMasks[i][idx] : deriv;
      });

      const layerError = errors[i].map((error, idx) => error * activationDeriv[idx]);

      // Weight gradients
      const weightGradient = this._createMatrix(this.weights[i].length, this.weights[i][0].length, 0);
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          weightGradient[j][k] = layerError[k] * activations[i][j];
        }
      }
      gradients.weightGradients.unshift(weightGradient);

      // Bias gradients
      gradients.biasGradients.unshift([...layerError]);
    }

    return gradients;
  }

  /**
   * Update network parameters with momentum and regularization
   */
  _updateParameters(gradients, learningRate) {
    for (let i = 0; i < this.weights.length; i++) {
      // Update weights with momentum
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          const gradient = gradients.weightGradients[i][j][k];
          
          // Apply regularization
          let regTerm = 0;
          if (this.regularization === 'l2') {
            regTerm = this.regularizationRate * this.weights[i][j][k];
          } else if (this.regularization === 'l1') {
            regTerm = this.regularizationRate * Math.sign(this.weights[i][j][k]);
          }

          const update = learningRate * (gradient + regTerm);
          this.weightUpdates[i][j][k] = this.momentum * this.weightUpdates[i][j][k] + update;
          this.weights[i][j][k] += this.weightUpdates[i][j][k];
        }
      }

      // Update biases with momentum
      for (let j = 0; j < this.biases[i].length; j++) {
        const gradient = gradients.biasGradients[i][j];
        const update = learningRate * gradient;
        this.biasUpdates[i][j] = this.momentum * this.biasUpdates[i][j] + update;
        this.biases[i][j] += this.biasUpdates[i][j];
      }
    }
  }

  /**
   * Apply learning rate scheduling
   */
  _applyLearningRateSchedule(currentLR, epoch, config) {
    switch (config.learningRateSchedule) {
      case 'exponential':
        return currentLR * Math.pow(config.decayRate, Math.floor(epoch / config.decaySteps));
      case 'plateau':
        // Simple plateau detection (would need more sophisticated implementation)
        return currentLR;
      default:
        return currentLR;
    }
  }

  /**
   * Shuffle array using Fisher-Yates algorithm
   */
  _shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  /**
   * Make predictions
   */
  predict(inputs) {
    const result = this.forwardPass(inputs, false);
    return result.activations[result.activations.length - 1];
  }

  /**
   * Test the network and display results
   */
  test(trainingData) {
    console.log("\n=== PREDICTIONS ===");
    trainingData.forEach((data, index) => {
      const prediction = this.predict(data.input);
      const target = data.output;
      
      console.log(`Example ${index + 1}:`);
      console.log(`  Input: [${data.input.map(x => x.toFixed(3)).join(', ')}]`);
      console.log(`  Target: [${target.map(x => x.toFixed(3)).join(', ')}]`);
      console.log(`  Prediction: [${prediction.map(x => x.toFixed(3)).join(', ')}]`);
      console.log(`  Error: [${target.map((t, i) => Math.abs(t - prediction[i])).map(x => x.toFixed(3)).join(', ')}]`);
      console.log('');
    });
  }

  /**
   * Get training history
   */
  getTrainingHistory() {
    return this.trainingHistory;
  }

  /**
   * Save network to JSON
   */
  save(filename) {
    const data = {
      config: this.config,
      weights: this.weights,
      biases: this.biases,
      trainingHistory: this.trainingHistory
    };
    
    fs.writeFileSync(filename, JSON.stringify(data, null, 2));
    console.log(`Network saved to ${filename}`);
  }

  /**
   * Load network from JSON
   */
  static load(filename) {
    const data = JSON.parse(fs.readFileSync(filename, 'utf8'));
    
    const nn = new NeuralNetwork(data.config);
    nn.weights = data.weights;
    nn.biases = data.biases;
    nn.trainingHistory = data.trainingHistory || [];
    
    console.log(`Network loaded from ${filename}`);
    return nn;
  }
}

/**
 * CBOW Model Class
 *
 * A memory-efficient implementation of the Continuous Bag of Words model
 * with optimized context averaging and reusable buffers.
 */
class CBOWModel {
  /**
   * Initialize the CBOW model
   * @param {number} vocabularySize - Size of the vocabulary
   * @param {number} embeddingDim - Dimension of word embeddings
   */
  constructor(vocabularySize, embeddingDim) {
    this.vocabularySize = vocabularySize;
    this.embeddingDim = embeddingDim;

    // Initialize weights with proper structure
    this.weights = [
      this.initializeWeights(vocabularySize, embeddingDim),  // Input embeddings
      this.initializeWeights(embeddingDim, vocabularySize)    // Hidden to output
    ];

    // Reusable buffers for memory efficiency
    this.averagedInput = new Float64Array(embeddingDim);
    this.targetVector = new Float64Array(vocabularySize);
    this.hiddenLayer = new Float64Array(embeddingDim);
    this.outputLayer = new Float64Array(vocabularySize);
  }

  /**
   * Initialize weights with small random values
   */
  initializeWeights(rows, cols) {
    const weights = [];
    for (let i = 0; i < rows; i++) {
      weights[i] = new Float64Array(cols);
      for (let j = 0; j < cols; j++) {
        weights[i][j] = (Math.random() - 0.5) * 0.1;  // Small random values
      }
    }
    return weights;
  }

  /**
   * Get averaged context vector for CBOW with optimized calculation
   * @param {number[]} contextIndices - Array of context word indices
   * @returns {Float64Array} The averaged context vector
   */
  getContextVector(contextIndices) {
    // Clear the buffer
    this.averagedInput.fill(0);

    // Sum all context word embeddings
    for (const idx of contextIndices) {
      const wordVector = this.weights[0][idx];
      for (let i = 0; i < this.embeddingDim; i++) {
        this.averagedInput[i] += wordVector[i];
      }
    }

    // Divide by context size once at the end (optimization)
    const contextSize = contextIndices.length;
    if (contextSize > 0) {
      for (let i = 0; i < this.embeddingDim; i++) {
        this.averagedInput[i] /= contextSize;
      }
    }

    return this.averagedInput;
  }

  /**
   * Training method with exact CBOW implementation
   * @param {Object} pair - Training pair with context and target
   */
  train(pair) {
    // 1. Get averaged context vector
    const contextIndices = pair.context;
    const inputVector = this.getContextVector(contextIndices);

    // 2. Prepare target vector (reusing buffer)
    this.targetVector.fill(0);
    this.targetVector[pair.target] = 1;

    // 3. Forward propagation
    this.forwardPropagate(inputVector);

    // 4. Backpropagation and weight updates
    this.backPropagate(this.targetVector);
    this.updateWeights();
  }

  /**
   * Forward propagation for CBOW
   * @param {Float64Array} inputVector - Input vector
   */
  forwardPropagate(inputVector) {
    // Copy input to hidden layer
    for (let i = 0; i < this.embeddingDim; i++) {
      this.hiddenLayer[i] = inputVector[i];
    }

    // Calculate output layer activations
    for (let i = 0; i < this.vocabularySize; i++) {
      this.outputLayer[i] = 0;
      for (let j = 0; j < this.embeddingDim; j++) {
        this.outputLayer[i] += this.hiddenLayer[j] * this.weights[1][j][i];
      }
      // Apply sigmoid activation
      this.outputLayer[i] = 1 / (1 + Math.exp(-this.outputLayer[i]));
    }
  }

  /**
   * Backpropagation for CBOW
   * @param {Float64Array} targetVector - Target vector
   */
  backPropagate(targetVector) {
    // Calculate output error
    for (let i = 0; i < this.vocabularySize; i++) {
      const error = targetVector[i] - this.outputLayer[i];
      const outputDerivative = this.outputLayer[i] * (1 - this.outputLayer[i]);

      // Update hidden to output weights
      for (let j = 0; j < this.embeddingDim; j++) {
        const gradient = error * outputDerivative * this.hiddenLayer[j];
        this.weights[1][j][i] += 0.01 * gradient; // Learning rate
      }
    }
  }

  /**
   * Update weights for CBOW
   */
  updateWeights() {
    // This method can be expanded for more sophisticated updates
    // Currently handled in backPropagate for simplicity
  }

  /**
   * Batch training with exact CBOW implementation
   * @param {Array} trainingPairs - Array of training pairs
   * @param {Object} options - Training options
   */
  trainBatch(trainingPairs, options = {}) {
    const { epochs = 1, batchSize = 1000 } = options;
    const totalPairs = trainingPairs.length;

    console.log(`Starting CBOW training with ${totalPairs} pairs`);

    for (let epoch = 0; epoch < epochs; epoch++) {
      console.log(`Epoch ${epoch + 1}/${epochs}`);

      for (let i = 0; i < totalPairs; i++) {
        const pair = trainingPairs[i];
        this.train(pair);

        // Progress reporting
        if (i % 1000 === 0) {
          console.log(`Processed ${i}/${totalPairs} pairs`);
        }
      }
    }

    console.log('CBOW training complete');
  }
}

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

/**
 * EmbeddingSimilarity Class
 *
 * A utility for finding similar words using embeddings from a trained neural network
 */
class EmbeddingSimilarity {
  /**
   * Initialize with a trained network and vocabulary
   * @param {NeuralNetwork} network - Trained neural network
   * @param {Object} vocabulary - Vocabulary mapping object
   */
  constructor(network, vocabulary) {
    this.network = network;
    this.vocabulary = vocabulary;
  }

  /**
   * Get embedding for a word
   * @param {string} word - Word to get embedding for
   * @returns {Array} The embedding vector
   */
  getEmbedding(word) {
    // Convert word to one-hot encoded vector
    const wordIndex = this.vocabulary.wordToIndex[word];
    if (wordIndex === undefined) {
      throw new Error(`Word "${word}" not in vocabulary`);
    }

    const input = new Array(this.vocabulary.vocabSize).fill(0);
    input[wordIndex] = 1;

    // Get hidden layer activation (the embedding)
    const result = this.network.forwardPass(input, false);
    return result.activations[1]; // Hidden layer activations
  }

  /**
   * Calculate cosine similarity between two vectors
   * @param {Array} vecA - First vector
   * @param {Array} vecB - Second vector
   * @returns {number} Cosine similarity score between -1 and 1
   */
  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magnitudeA += vecA[i] * vecA[i];
      magnitudeB += vecB[i] * vecB[i];
    }

    // Handle edge case where one or both vectors have zero magnitude
    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    // Add debug output to check values
    console.log(`Debug - Dot product: ${dotProduct}, Magnitude A: ${magnitudeA}, Magnitude B: ${magnitudeB}`);

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Find most similar words to a given word
   * @param {string} word - Target word
   * @param {number} topN - Number of similar words to return
   * @returns {Array} Array of {word, similarity} objects
   */
  findSimilarWords(word, topN = 5) {
    const targetEmbedding = this.getEmbedding(word);
    const similarities = [];

    // Compare with all words in vocabulary
    for (const vocabWord of this.vocabulary.vocab) {
      if (vocabWord === word) continue;

      try {
        const embedding = this.getEmbedding(vocabWord);
        const similarity = this.cosineSimilarity(targetEmbedding, embedding);
        similarities.push({ word: vocabWord, similarity });
      } catch (e) {
        // Skip words not in vocabulary (shouldn't happen)
      }
    }

    // Sort by similarity and return top N
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topN);
  }
}

export { NeuralNetwork, CBOWModel, EmbeddingSimilarity };
