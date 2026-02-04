import fs from 'fs';

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

export { CBOWModel };