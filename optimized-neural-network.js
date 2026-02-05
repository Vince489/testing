/**
 * Optimized Neural Network Implementation
 * 
 * A production-grade neural network with:
 * - Layer-based architecture using DenseLayer classes
 * - Float32Array for memory efficiency and performance
 * - Softmax activation and cross-entropy loss for classification
 * - Sparse input optimization for CBOW models
 * - Adam optimizer for better convergence
 * - Binary weight saving/loading
 */

/**
 * DenseLayer Class
 * 
 * A single dense (fully connected) layer with optimized operations
 */
class DenseLayer {
  constructor(inputSize, outputSize, config = {}) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = config.activation || 'sigmoid';
    this.useBias = config.useBias !== false;
    
    // Performance: Use Float32Array for all calculations
    this.weights = new Float32Array(inputSize * outputSize);
    this.biases = this.useBias ? new Float32Array(outputSize) : null;
    
    // Adam optimizer state
    this.mWeights = new Float32Array(inputSize * outputSize); // 1st moment
    this.vWeights = new Float32Array(inputSize * outputSize); // 2nd moment
    this.mBiases = this.useBias ? new Float32Array(outputSize) : null;
    this.vBiases = this.useBias ? new Float32Array(outputSize) : null;
    
    // For backpropagation
    this.lastInput = null;
    this.lastOutput = null;
    
    this._initWeights();
  }

  /**
   * Initialize weights using Xavier/Glorot initialization
   */
  _initWeights() {
    const limit = Math.sqrt(6 / (this.inputSize + this.outputSize));
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() * 2 - 1) * limit;
    }
    
    if (this.biases) {
      for (let i = 0; i < this.biases.length; i++) {
        this.biases[i] = 0.1; // Small positive bias
      }
    }
  }

  /**
   * Forward pass with optimized matrix multiplication
   */
  forward(input) {
    this.lastInput = input;
    const output = new Float32Array(this.outputSize);
    
    // Optimized dot product: output = input * weights + biases
    for (let j = 0; j < this.outputSize; j++) {
      let sum = this.useBias ? this.biases[j] : 0;
      for (let i = 0; i < this.inputSize; i++) {
        sum += input[i] * this.weights[i * this.outputSize + j];
      }
      output[j] = sum;
    }
    
    // Apply activation function
    this.lastOutput = this._applyActivation(output);
    return this.lastOutput;
  }

  /**
   * Sparse forward pass for one-hot inputs (CBOW optimization)
   */
  forwardSparse(wordIndex) {
    if (wordIndex < 0 || wordIndex >= this.inputSize) {
      throw new Error(`Word index ${wordIndex} out of range [0, ${this.inputSize})`);
    }
    
    // Create the one-hot input vector for backpropagation
    const input = new Float32Array(this.inputSize);
    input[wordIndex] = 1;
    this.lastInput = input;
    
    const output = new Float32Array(this.outputSize);
    
    // Optimization: Just copy the relevant weight row
    const startIndex = wordIndex * this.outputSize;
    for (let j = 0; j < this.outputSize; j++) {
      output[j] = this.weights[startIndex + j];
      if (this.useBias && this.biases) {
        output[j] += this.biases[j];
      }
    }
    
    this.lastOutput = this._applyActivation(output);
    return this.lastOutput;
  }

  /**
   * Apply activation function
   */
  _applyActivation(x) {
    const result = new Float32Array(x.length);
    
    switch (this.activation) {
      case 'sigmoid':
        for (let i = 0; i < x.length; i++) {
          result[i] = 1 / (1 + Math.exp(-x[i]));
        }
        break;
      case 'tanh':
        for (let i = 0; i < x.length; i++) {
          result[i] = Math.tanh(x[i]);
        }
        break;
      case 'relu':
        for (let i = 0; i < x.length; i++) {
          result[i] = Math.max(0, x[i]);
        }
        break;
      case 'softmax':
        result.set(this._softmax(x));
        break;
      default:
        for (let i = 0; i < x.length; i++) {
          result[i] = 1 / (1 + Math.exp(-x[i]));
        }
    }
    
    return result;
  }

  /**
   * Numerically stable softmax implementation
   */
  _softmax(x) {
    const maxVal = Math.max(...x);
    const exps = new Float32Array(x.length);
    let sum = 0;
    
    for (let i = 0; i < x.length; i++) {
      exps[i] = Math.exp(x[i] - maxVal);
      sum += exps[i];
    }
    
    const result = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      result[i] = exps[i] / sum;
    }
    
    return result;
  }

  /**
   * Backward pass for gradient calculation
   */
  backward(inputError, learningRate, epoch) {
    // Calculate activation derivative
    const activationDeriv = this._activationDerivative(this.lastOutput);
    
    // Element-wise multiplication: inputError * activationDerivative
    const layerError = new Float32Array(inputError.length);
    for (let i = 0; i < inputError.length; i++) {
      layerError[i] = inputError[i] * activationDeriv[i];
    }
    
    // Calculate weight gradients: layerError * lastInput^T
    const weightGradients = new Float32Array(this.weights.length);
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        weightGradients[i * this.outputSize + j] = this.lastInput[i] * layerError[j];
      }
    }
    
    // Calculate bias gradients
    let biasGradients = null;
    if (this.useBias && this.biases && this.biases.length > 0) {
      biasGradients = new Float32Array(this.biases.length);
      for (let j = 0; j < this.outputSize; j++) {
        biasGradients[j] = layerError[j];
      }
    }
    
    // Update weights using Adam optimizer
    this._updateWeightsAdam(weightGradients, biasGradients, learningRate, epoch);
    
    // Calculate input gradients for previous layer
    const inputGradients = new Float32Array(this.inputSize);
    for (let i = 0; i < this.inputSize; i++) {
      let sum = 0;
      for (let j = 0; j < this.outputSize; j++) {
        sum += layerError[j] * this.weights[i * this.outputSize + j];
      }
      inputGradients[i] = sum;
    }
    
    return inputGradients;
  }

  /**
   * Activation function derivatives
   */
  _activationDerivative(x) {
    const result = new Float32Array(x.length);
    
    switch (this.activation) {
      case 'sigmoid':
        for (let i = 0; i < x.length; i++) {
          result[i] = x[i] * (1 - x[i]);
        }
        break;
      case 'tanh':
        for (let i = 0; i < x.length; i++) {
          result[i] = 1 - x[i] * x[i];
        }
        break;
      case 'relu':
        for (let i = 0; i < x.length; i++) {
          result[i] = x[i] > 0 ? 1 : 0;
        }
        break;
      case 'softmax':
        // For softmax, we need the full Jacobian, but this is handled
        // in the cross-entropy loss calculation
        for (let i = 0; i < x.length; i++) {
          result[i] = 1; // Placeholder, actual derivative handled in loss
        }
        break;
      default:
        for (let i = 0; i < x.length; i++) {
          result[i] = x[i] * (1 - x[i]);
        }
    }
    
    return result;
  }

  /**
   * Adam optimizer update
   */
  _updateWeightsAdam(weightGradients, biasGradients, learningRate, epoch) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;
    const biasCorrection = true;
    
    // Update weights
    for (let i = 0; i < this.weights.length; i++) {
      // Update moments
      this.mWeights[i] = beta1 * this.mWeights[i] + (1 - beta1) * weightGradients[i];
      this.vWeights[i] = beta2 * this.vWeights[i] + (1 - beta2) * (weightGradients[i] * weightGradients[i]);
      
      let mHat = this.mWeights[i];
      let vHat = this.vWeights[i];
      
      if (biasCorrection) {
        const t = epoch + 1;
        mHat = this.mWeights[i] / (1 - Math.pow(beta1, t));
        vHat = this.vWeights[i] / (1 - Math.pow(beta2, t));
      }
      
      // Update weight
      this.weights[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
    }
    
    // Update biases
    if (this.useBias && this.biases && biasGradients) {
      for (let i = 0; i < this.biases.length; i++) {
        this.mBiases[i] = beta1 * this.mBiases[i] + (1 - beta1) * biasGradients[i];
        this.vBiases[i] = beta2 * this.vBiases[i] + (1 - beta2) * (biasGradients[i] * biasGradients[i]);
        
        let mHat = this.mBiases[i];
        let vHat = this.vBiases[i];
        
        if (biasCorrection) {
          const t = epoch + 1;
          mHat = this.mBiases[i] / (1 - Math.pow(beta1, t));
          vHat = this.vBiases[i] / (1 - Math.pow(beta2, t));
        }
        
        this.biases[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
      }
    }
  }

  /**
   * Get weight matrix as 2D array (for compatibility)
   */
  getWeightMatrix() {
    const matrix = [];
    for (let i = 0; i < this.inputSize; i++) {
      const row = [];
      for (let j = 0; j < this.outputSize; j++) {
        row.push(this.weights[i * this.outputSize + j]);
      }
      matrix.push(row);
    }
    return matrix;
  }

  /**
   * Get bias vector (for compatibility)
   */
  getBiasVector() {
    if (!this.biases) return null;
    return Array.from(this.biases);
  }
}

/**
 * Optimized Neural Network Class
 */
class OptimizedNeuralNetwork {
  constructor(config) {
    this.layers = [];
    this.layerConfigs = config.layerConfigs || [2, 2, 1];
    this.learningRate = config.learningRate || 0.01; // Lower default for Adam
    this.outputActivation = config.outputActivation || 'sigmoid';
    this.lossFunction = config.lossFunction || 'mse'; // 'mse' or 'cross_entropy'
    this.useAdam = config.useAdam !== false;
    this.gradientClipping = config.gradientClipping || 5.0;
    
    // Restore training state if available
    this.currentEpoch = config.currentEpoch || 0;
    this.trainingHistory = config.trainingHistory || [];
    
    this._buildNetwork();
  }

  /**
   * Build the network from layer configurations
   */
  _buildNetwork() {
    for (let i = 0; i < this.layerConfigs.length - 1; i++) {
      const inputSize = this.layerConfigs[i];
      const outputSize = this.layerConfigs[i + 1];
      const activation = i === this.layerConfigs.length - 2 ? this.outputActivation : 'relu';
      
      const layer = new DenseLayer(inputSize, outputSize, {
        activation: activation,
        useBias: true // Always use bias for all layers
      });
      
      this.layers.push(layer);
    }
  }

  /**
   * Forward pass through all layers
   */
  forwardPass(inputs, training = true) {
    let currentInput = new Float32Array(inputs);
    
    for (let i = 0; i < this.layers.length; i++) {
      // For CBOW, use sparse input if it's a one-hot vector
      if (training && i === 0 && this._isOneHot(currentInput)) {
        const wordIndex = this._getOneHotIndex(currentInput);
        currentInput = this.layers[i].forwardSparse(wordIndex);
      } else {
        currentInput = this.layers[i].forward(currentInput);
      }
    }
    
    return Array.from(currentInput);
  }

  /**
   * Check if input is a one-hot vector
   */
  _isOneHot(input) {
    let ones = 0;
    for (let i = 0; i < input.length; i++) {
      if (input[i] === 1) ones++;
      else if (input[i] !== 0) return false;
    }
    return ones === 1;
  }

  /**
   * Get index of the 1 in a one-hot vector
   */
  _getOneHotIndex(input) {
    for (let i = 0; i < input.length; i++) {
      if (input[i] === 1) return i;
    }
    return -1;
  }

  /**
   * Train the network
   */
  train(trainingData, epochs = 1000, options = {}) {
    const config = {
      validationData: options.validationData || null,
      earlyStopping: options.earlyStopping || false,
      patience: options.patience || 10,
      minDelta: options.minDelta || 0.001,
      verbose: options.verbose || false,
      ...options
    };

    let bestLoss = Infinity;
    let patienceCounter = 0;

    console.log(`Starting training with ${epochs} epochs...`);
    console.log(`Network architecture: [${this.layerConfigs.join(' -> ')}]`);
    console.log(`Output activation: ${this.outputActivation}, Loss: ${this.lossFunction}`);
    console.log(`Learning rate: ${this.learningRate}, Adam: ${this.useAdam}`);

    const startTime = Date.now();

    for (let epoch = 0; epoch < epochs; epoch++) {
      this.currentEpoch = epoch;
      
      // Shuffle training data
      const shuffledData = this._shuffleArray([...trainingData]);
      
      // Train on all data
      let totalLoss = 0;
      for (const data of shuffledData) {
        const prediction = this.forwardPass(data.input, true);
        const loss = this._calculateLoss(prediction, data.output);
        totalLoss += loss;
        
        // Backpropagation
        const outputError = this._calculateOutputError(prediction, data.output);
        this._backpropagate(outputError);
      }

      const avgLoss = totalLoss / shuffledData.length;
      this.trainingHistory.push({ epoch, loss: avgLoss });

      // Validation check
      if (config.validationData) {
        const validationLoss = this._calculateLossDataset(config.validationData);
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
      if (config.verbose && (epoch % 100 === 0 || epoch === epochs - 1)) {
        console.log(`Epoch ${epoch}: Loss = ${avgLoss.toFixed(6)}`);
      }
    }

    const trainingTime = Date.now() - startTime;
    console.log(`Training completed in ${trainingTime}ms`);
    console.log(`Final loss: ${this.trainingHistory[this.trainingHistory.length - 1].loss.toFixed(6)}`);
  }

  /**
   * Backpropagation through all layers
   */
  _backpropagate(outputError) {
    let currentError = new Float32Array(outputError);
    
    // Backpropagate through layers in reverse order
    for (let i = this.layers.length - 1; i >= 0; i--) {
      currentError = this.layers[i].backward(currentError, this.learningRate, this.currentEpoch);
    }
  }

  /**
   * Calculate output error based on loss function
   */
  _calculateOutputError(prediction, target) {
    const error = new Float32Array(prediction.length);
    
    if (this.lossFunction === 'cross_entropy' && this.outputActivation === 'softmax') {
      // Simplified gradient for softmax + cross-entropy
      for (let i = 0; i < prediction.length; i++) {
        error[i] = prediction[i] - target[i];
      }
    } else {
      // Standard MSE gradient
      for (let i = 0; i < prediction.length; i++) {
        error[i] = prediction[i] - target[i];
      }
    }
    
    return error;
  }

  /**
   * Calculate loss for a single prediction
   */
  _calculateLoss(prediction, target) {
    let loss = 0;
    
    if (this.lossFunction === 'cross_entropy') {
      for (let i = 0; i < prediction.length; i++) {
        // Add small epsilon to prevent log(0)
        const p = Math.max(1e-15, Math.min(1 - 1e-15, prediction[i]));
        loss -= target[i] * Math.log(p);
      }
    } else {
      // MSE
      for (let i = 0; i < prediction.length; i++) {
        loss += Math.pow(target[i] - prediction[i], 2);
      }
      loss /= 2;
    }
    
    return loss;
  }

  /**
   * Calculate loss for a dataset
   */
  _calculateLossDataset(dataset) {
    let totalLoss = 0;
    for (const data of dataset) {
      const prediction = this.predict(data.input);
      totalLoss += this._calculateLoss(prediction, data.output);
    }
    return totalLoss / dataset.length;
  }

  /**
   * Make predictions
   */
  predict(inputs) {
    return this.forwardPass(inputs, false);
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
   * Get training history
   */
  getTrainingHistory() {
    return this.trainingHistory;
  }

  /**
   * Helper methods for save/load - extract all weights as single Float32Array
   */
  getAllWeightsAsFloat32() {
    const totalLength = this.layers.reduce((sum, layer) => sum + layer.weights.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const layer of this.layers) {
      result.set(layer.weights, offset);
      offset += layer.weights.length;
    }
    
    return result;
  }

  /**
   * Extract all biases as single Float32Array
   */
  getAllBiasesAsFloat32() {
    const totalLength = this.layers.reduce((sum, layer) => sum + (layer.biases ? layer.biases.length : 0), 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const layer of this.layers) {
      if (layer.biases) {
        result.set(layer.biases, offset);
        offset += layer.biases.length;
      }
    }
    
    return result;
  }

  /**
   * Extract all Adam mWeights as single Float32Array
   */
  getAllMWeightsAsFloat32() {
    const totalLength = this.layers.reduce((sum, layer) => sum + layer.mWeights.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const layer of this.layers) {
      result.set(layer.mWeights, offset);
      offset += layer.mWeights.length;
    }
    
    return result;
  }

  /**
   * Extract all Adam vWeights as single Float32Array
   */
  getAllVWeightsAsFloat32() {
    const totalLength = this.layers.reduce((sum, layer) => sum + layer.vWeights.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const layer of this.layers) {
      result.set(layer.vWeights, offset);
      offset += layer.vWeights.length;
    }
    
    return result;
  }

  /**
   * Extract all Adam mBiases as single Float32Array
   */
  getAllMBiasesAsFloat32() {
    const totalLength = this.layers.reduce((sum, layer) => sum + (layer.mBiases ? layer.mBiases.length : 0), 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const layer of this.layers) {
      if (layer.mBiases) {
        result.set(layer.mBiases, offset);
        offset += layer.mBiases.length;
      }
    }
    
    return result;
  }

  /**
   * Extract all Adam vBiases as single Float32Array
   */
  getAllVBiasesAsFloat32() {
    const totalLength = this.layers.reduce((sum, layer) => sum + (layer.vBiases ? layer.vBiases.length : 0), 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    
    for (const layer of this.layers) {
      if (layer.vBiases) {
        result.set(layer.vBiases, offset);
        offset += layer.vBiases.length;
      }
    }
    
    return result;
  }

  /**
   * Set all weights from single Float32Array
   */
  setAllWeightsFromFloat32(weightsArray) {
    let offset = 0;
    
    for (const layer of this.layers) {
      // Copy weights slice to layer weights
      for (let i = 0; i < layer.weights.length; i++) {
        layer.weights[i] = weightsArray[offset + i];
      }
      offset += layer.weights.length;
    }
  }

  /**
   * Set all biases from single Float32Array
   */
  setAllBiasesFromFloat32(biasesArray) {
    let offset = 0;
    
    for (const layer of this.layers) {
      if (layer.biases) {
        layer.biases.set(biasesArray.slice(offset, offset + layer.biases.length));
        offset += layer.biases.length;
      }
    }
  }

  /**
   * Set all Adam mWeights from single Float32Array
   */
  setAllMWeightsFromFloat32(mWeightsArray) {
    let offset = 0;
    
    for (const layer of this.layers) {
      layer.mWeights.set(mWeightsArray.slice(offset, offset + layer.mWeights.length));
      offset += layer.mWeights.length;
    }
  }

  /**
   * Set all Adam vWeights from single Float32Array
   */
  setAllVWeightsFromFloat32(vWeightsArray) {
    let offset = 0;
    
    for (const layer of this.layers) {
      layer.vWeights.set(vWeightsArray.slice(offset, offset + layer.vWeights.length));
      offset += layer.vWeights.length;
    }
  }

  /**
   * Set all Adam mBiases from single Float32Array
   */
  setAllMBiasesFromFloat32(mBiasesArray) {
    let offset = 0;
    
    for (const layer of this.layers) {
      if (layer.mBiases) {
        layer.mBiases.set(mBiasesArray.slice(offset, offset + layer.mBiases.length));
        offset += layer.mBiases.length;
      }
    }
  }

  /**
   * Set all Adam vBiases from single Float32Array
   */
  setAllVBiasesFromFloat32(vBiasesArray) {
    let offset = 0;
    
    for (const layer of this.layers) {
      if (layer.vBiases) {
        layer.vBiases.set(vBiasesArray.slice(offset, offset + layer.vBiases.length));
        offset += layer.vBiases.length;
      }
    }
  }

  /**
   * Save network to binary file with robust state preservation
   */
  save(filename) {
    const data = {
      layerConfigs: this.layerConfigs,
      learningRate: this.learningRate,
      outputActivation: this.outputActivation,
      lossFunction: this.lossFunction,
      useAdam: this.useAdam,
      currentEpoch: this.currentEpoch,
      trainingHistory: this.trainingHistory
    };
    
    // Save configuration as JSON
    const configData = JSON.stringify(data);
    const configBuffer = Buffer.from(configData, 'utf8');
    
    // Combine all weights into one contiguous Float32Array
    const totalWeights = this.getAllWeightsAsFloat32();
    const totalBiases = this.getAllBiasesAsFloat32();
    
    // Save Adam optimizer state
    const totalMWeights = this.getAllMWeightsAsFloat32();
    const totalVWeights = this.getAllVWeightsAsFloat32();
    const totalMBiases = this.getAllMBiasesAsFloat32();
    const totalVBiases = this.getAllVBiasesAsFloat32();
    
    // Create header with sizes
    const header = Buffer.alloc(32); // 8 uint32 fields
    let headerOffset = 0;
    header.writeUInt32LE(configBuffer.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(totalWeights.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(totalBiases.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(totalMWeights.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(totalVWeights.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(totalMBiases.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(totalVBiases.length, headerOffset); headerOffset += 4;
    header.writeUInt32LE(this.layers.length, headerOffset); headerOffset += 4;
    
    // Convert Float32Arrays to Buffers with proper slicing
    const weightsBuffer = Buffer.from(totalWeights.buffer, totalWeights.byteOffset, totalWeights.byteLength);
    const biasesBuffer = Buffer.from(totalBiases.buffer, totalBiases.byteOffset, totalBiases.byteLength);
    const mWeightsBuffer = Buffer.from(totalMWeights.buffer, totalMWeights.byteOffset, totalMWeights.byteLength);
    const vWeightsBuffer = Buffer.from(totalVWeights.buffer, totalVWeights.byteOffset, totalVWeights.byteLength);
    const mBiasesBuffer = Buffer.from(totalMBiases.buffer, totalMBiases.byteOffset, totalMBiases.byteLength);
    const vBiasesBuffer = Buffer.from(totalVBiases.buffer, totalVBiases.byteOffset, totalVBiases.byteLength);
    
    // Combine all buffers
    const finalBuffer = Buffer.concat([
      header,
      configBuffer,
      weightsBuffer,
      biasesBuffer,
      mWeightsBuffer,
      vWeightsBuffer,
      mBiasesBuffer,
      vBiasesBuffer
    ]);
    
    // Use dynamic import for fs to work with ES modules
    import('fs').then(fs => {
      fs.writeFileSync(filename, finalBuffer);
      console.log(`Network saved to ${filename} (${finalBuffer.length} bytes)`);
    });
  }

  /**
   * Load network from binary file with robust state restoration
   */
  static async load(filename) {
    // Use dynamic import for fs to work with ES modules
    const fs = await import('fs');
    const buffer = fs.readFileSync(filename);
    let offset = 0;
    
    // Read header with all sizes
    const configSize = buffer.readUInt32LE(offset); offset += 4;
    const weightSize = buffer.readUInt32LE(offset); offset += 4;
    const biasSize = buffer.readUInt32LE(offset); offset += 4;
    const mWeightSize = buffer.readUInt32LE(offset); offset += 4;
    const vWeightSize = buffer.readUInt32LE(offset); offset += 4;
    const mBiasSize = buffer.readUInt32LE(offset); offset += 4;
    const vBiasSize = buffer.readUInt32LE(offset); offset += 4;
    const layerCount = buffer.readUInt32LE(offset); offset += 4;
    
    // Read config
    const configData = buffer.toString('utf8', offset, offset + configSize);
    offset += configSize;
    const config = JSON.parse(configData);
    
    // Create network with exact same architecture
    const nn = new OptimizedNeuralNetwork(config);
    
    // Validate layer count matches
    if (nn.layers.length !== layerCount) {
      throw new Error(`Layer count mismatch: expected ${layerCount}, got ${nn.layers.length}`);
    }
    
    // Read all binary data sections - fix buffer reading logic
    const weightsArray = new Float32Array(weightSize);
    const biasesArray = new Float32Array(biasSize);
    const mWeightsArray = new Float32Array(mWeightSize);
    const vWeightsArray = new Float32Array(vWeightSize);
    const mBiasesArray = new Float32Array(mBiasSize);
    const vBiasesArray = new Float32Array(vBiasSize);
    
    // Read float values directly from the original buffer at correct offsets
    // This fixes the buffer reading issue where sliced buffers were not properly aligned
    for (let i = 0; i < weightSize; i++) {
      weightsArray[i] = buffer.readFloatLE(offset + i * 4);
    }
    offset += weightSize * 4;
    
    for (let i = 0; i < biasSize; i++) {
      biasesArray[i] = buffer.readFloatLE(offset + i * 4);
    }
    offset += biasSize * 4;
    
    for (let i = 0; i < mWeightSize; i++) {
      mWeightsArray[i] = buffer.readFloatLE(offset + i * 4);
    }
    offset += mWeightSize * 4;
    
    for (let i = 0; i < vWeightSize; i++) {
      vWeightsArray[i] = buffer.readFloatLE(offset + i * 4);
    }
    offset += vWeightSize * 4;
    
    for (let i = 0; i < mBiasSize; i++) {
      mBiasesArray[i] = buffer.readFloatLE(offset + i * 4);
    }
    offset += mBiasSize * 4;
    
    for (let i = 0; i < vBiasSize; i++) {
      vBiasesArray[i] = buffer.readFloatLE(offset + i * 4);
    }
    offset += vBiasSize * 4;
    
    // Restore all state to layers
    nn.setAllWeightsFromFloat32(weightsArray);
    nn.setAllBiasesFromFloat32(biasesArray);
    nn.setAllMWeightsFromFloat32(mWeightsArray);
    nn.setAllVWeightsFromFloat32(vWeightsArray);
    nn.setAllMBiasesFromFloat32(mBiasesArray);
    nn.setAllVBiasesFromFloat32(vBiasesArray);
    
    console.log(`Network loaded from ${filename} (${buffer.length} bytes)`);
    return nn;
  }
}

export { DenseLayer, OptimizedNeuralNetwork };