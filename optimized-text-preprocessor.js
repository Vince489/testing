import fs from 'fs';

/**
 * OptimizedTextPreprocessor Class
 * 
 * A memory-optimized text preprocessing module for CBOW (Continuous Bag of Words) model training.
 * This class addresses the memory bottleneck by using sparse indices instead of dense one-hot vectors.
 * 
 * Key Optimizations:
 * - Sparse training data: Returns indices instead of dense arrays
 * - Embedding layer: Efficient index-based lookups
 * - Memory monitoring: Tracks and reports memory usage
 * - Proper CBOW averaging: Implements correct context averaging
 */

class EmbeddingLayer {
  constructor(vocabSize, embeddingDim) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.weights = this._initializeWeights(vocabSize, embeddingDim);
  }

  /**
   * Initialize embedding weights with small random values
   */
  _initializeWeights(vocabSize, embeddingDim) {
    const weights = [];
    for (let i = 0; i < vocabSize; i++) {
      const row = [];
      for (let j = 0; j < embeddingDim; j++) {
        // Initialize with small random values between -0.5 and 0.5
        row.push((Math.random() - 0.5) / embeddingDim);
      }
      weights.push(row);
    }
    return weights;
  }

  /**
   * Lookup embeddings for given indices
   * @param {number[]} indices - Array of word indices
   * @returns {number[][]} Array of embedding vectors
   */
  lookup(indices) {
    return indices.map(idx => this.weights[idx]);
  }

  /**
   * Average multiple embedding vectors (for CBOW context)
   * @param {number[][]} embeddings - Array of embedding vectors
   * @returns {number[]} Averaged embedding vector
   */
  averageEmbeddings(embeddings) {
    if (embeddings.length === 0) {
      return new Array(this.embeddingDim).fill(0);
    }

    const result = new Array(this.embeddingDim).fill(0);
    
    // Sum all embeddings
    for (const embedding of embeddings) {
      for (let i = 0; i < this.embeddingDim; i++) {
        result[i] += embedding[i];
      }
    }

    // Average by dividing by number of embeddings
    for (let i = 0; i < this.embeddingDim; i++) {
      result[i] /= embeddings.length;
    }

    return result;
  }

  /**
   * Get embedding for a single word index
   * @param {number} index - Word index
   * @returns {number[]} Embedding vector
   */
  getEmbedding(index) {
    return this.weights[index];
  }

  /**
   * Update embedding weights (for training)
   * @param {number} index - Word index
   * @param {number[]} gradients - Gradient vector
   * @param {number} learningRate - Learning rate
   */
  updateEmbedding(index, gradients, learningRate) {
    const embedding = this.weights[index];
    for (let i = 0; i < this.embeddingDim; i++) {
      embedding[i] -= learningRate * gradients[i];
    }
  }
}

class OptimizedTextPreprocessor {
  constructor() {
    this.tokens = [];
    this.vocab = [];
    this.wordToIndex = {};
    this.indexToWord = {};
    this.vocabSize = 0;
    this.minCount = 2;
    this.windowSize = 5;
    this.embeddingLayer = null;
    
    // Define stop words to exclude from vocabulary
    this.stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
      'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
      'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'then', 'here', 'there', 'up', 'down', 'out',
      'over', 'under', 'again', 'further', 'then', 'once', 'here', 'when', 'where', 'how', 'why', 'what', 'which', 'who', 'whom', 'whose'
    ]);
  }

  /**
   * Load and clean text from a file
   * @param {string} filePath - Path to the text file
   * @returns {string[]} Array of cleaned tokens
   */
  cleanText(filePath) {
    try {
      const rawText = fs.readFileSync(filePath, 'utf8');
      
      // Convert to lowercase
      const lowerText = rawText.toLowerCase();
      
      // Remove punctuation and special characters, but preserve numbers and hyphens
      // Keep letters, numbers, hyphens, and whitespace
      const cleanedText = lowerText.replace(/[^a-z0-9\s-]/g, '');
      
      // Split by whitespace and filter out empty strings
      const tokens = cleanedText.split(/\s+/).filter(word => word.length > 0);
      
      this.tokens = tokens;
      console.log(`Text cleaned successfully. Total tokens: ${tokens.length}`);
      return tokens;
    } catch (error) {
      console.error(`Error reading file ${filePath}:`, error.message);
      throw error;
    }
  }

  /**
   * Calculate word frequencies from tokens
   * @param {string[]} tokens - Array of tokens
   * @returns {Object} Object with word frequencies
   */
  calculateFrequencies(tokens) {
    const freqCounts = {};
    tokens.forEach(word => {
      freqCounts[word] = (freqCounts[word] || 0) + 1;
    });
    return freqCounts;
  }

  /**
   * Subsample frequent words using Mikolov formula
   * This provides the "Speed Multiplier" by reducing training data size by 30-60%
   * @param {string[]} tokens - Array of tokens
   * @param {number} threshold - Subsampling threshold (default 1e-5)
   * @returns {string[]} Subsampled tokens
   */
  subsampleFrequentWords(tokens, threshold = 1e-5) {
    if (!tokens || tokens.length === 0) {
      throw new Error('No tokens available for subsampling.');
    }

    console.log(`Starting subsampling with threshold ${threshold}...`);
    
    // Calculate word frequencies
    const freqCounts = this.calculateFrequencies(tokens);
    const totalTokens = tokens.length;
    
    // Apply Mikolov formula: P(wi) = 1 - sqrt(t/f(wi))
    const subsampledTokens = [];
    let discardedCount = 0;
    
    for (const token of tokens) {
      const freq = freqCounts[token] / totalTokens;
      const prob = 1 - Math.sqrt(threshold / freq);
      
      // Keep word if random number > probability (inverse of discard probability)
      if (Math.random() > prob) {
        subsampledTokens.push(token);
      } else {
        discardedCount++;
      }
    }
    
    const reductionPercent = ((discardedCount / tokens.length) * 100);
    console.log(`Subsampling completed:`);
    console.log(`- Original tokens: ${tokens.length}`);
    console.log(`- Subsampled tokens: ${subsampledTokens.length}`);
    console.log(`- Reduction: ${reductionPercent.toFixed(1)}%`);
    console.log(`- Free speed boost: Training data reduced by ${subsampledTokens.length / tokens.length * 100}%`);
    
    return subsampledTokens;
  }

  /**
   * Build vocabulary from tokens with frequency filtering
   * @param {string[]} tokens - Array of tokens
   * @param {number} minCount - Minimum frequency count for words to be included
   * @returns {Object} Vocabulary information
   */
  buildVocab(tokens = this.tokens, minCount = this.minCount) {
    if (!tokens || tokens.length === 0) {
      throw new Error('No tokens available. Please run cleanText() first.');
    }

    // Count word frequencies
    const counts = {};
    tokens.forEach(word => {
      counts[word] = (counts[word] || 0) + 1;
    });

    // Filter words by minimum count and exclude stop words
    this.vocab = Object.keys(counts)
      .filter(word => counts[word] >= minCount)
      .filter(word => !this.stopWords.has(word));
    
    // Sort vocabulary by frequency (descending)
    this.vocab.sort((a, b) => counts[b] - counts[a]);

    // Create lookup tables
    this.wordToIndex = {};
    this.indexToWord = {};

    this.vocab.forEach((word, index) => {
      this.wordToIndex[word] = index;
      this.indexToWord[index] = word;
    });

    this.vocabSize = this.vocab.length;
    this.minCount = minCount;

    console.log(`Vocabulary built successfully:`);
    console.log(`- Total unique words: ${Object.keys(counts).length}`);
    console.log(`- Words after filtering (minCount >= ${minCount}): ${this.vocabSize}`);
    console.log(`- Vocabulary coverage: ${((this.vocabSize / Object.keys(counts).length) * 100).toFixed(2)}%`);

    return {
      vocabSize: this.vocabSize,
      wordToIndex: this.wordToIndex,
      indexToWord: this.indexToWord,
      vocab: this.vocab
    };
  }

  /**
   * Initialize embedding layer
   * @param {number} embeddingDim - Dimension of embedding vectors
   */
  initEmbeddingLayer(embeddingDim = 100) {
    this.embeddingLayer = new EmbeddingLayer(this.vocabSize, embeddingDim);
    console.log(`Embedding layer initialized with dimension ${embeddingDim}`);
  }

  /**
   * Build negative sampling table using unigram distribution with 0.75 power
   * This provides O(1) lookup for negative samples and boosts rare word selection
   * @param {number} tableSize - Size of the unigram table (default 10 million)
   */
  buildNegativeSamplingTable(tableSize = 1e7) {
    if (this.vocabSize === 0) {
      throw new Error('Vocabulary not built. Please run buildVocab() first.');
    }

    console.log(`Building negative sampling table with ${tableSize} entries...`);
    
    // Calculate frequency^0.75 for all words
    const freqCounts = this.calculateFrequencies(this.tokens);
    const freqPower = this.vocab.map(word => Math.pow(freqCounts[word] || 1, 0.75));
    const totalPower = freqPower.reduce((sum, val) => sum + val, 0);
    
    // Normalize and create cumulative distribution
    const normalizedFreq = freqPower.map(freq => freq / totalPower);
    
    // Build unigram table for O(1) lookup
    this.negSamplingTable = new Int32Array(tableSize);
    let tableIndex = 0;
    
    for (let i = 0; i < this.vocabSize; i++) {
      const count = Math.floor(normalizedFreq[i] * tableSize);
      for (let j = 0; j < count; j++) {
        if (tableIndex < tableSize) {
          this.negSamplingTable[tableIndex++] = i;
        }
      }
    }
    
    // Fill remaining slots with random indices to ensure full coverage
    while (tableIndex < tableSize) {
      this.negSamplingTable[tableIndex++] = Math.floor(Math.random() * this.vocabSize);
    }
    
    console.log(`Negative sampling table generated with ${tableSize} entries for O(1) lookup.`);
    console.log(`- Rare words get boosted probability via 0.75 power`);
    console.log(`- Table ready for efficient negative sampling`);
  }

  /**
   * Get random negative indices using the unigram table
   * @param {number} count - Number of negative samples to get
   * @param {number} excludeIndex - Index to exclude from sampling
   * @returns {number[]} Array of negative sample indices
   */
  getRandomNegativeIndices(count, excludeIndex) {
    if (!this.negSamplingTable) {
      throw new Error('Negative sampling table not built. Please run buildNegativeSamplingTable() first.');
    }

    const negatives = [];
    let attempts = 0;
    const maxAttempts = count * 10; // Prevent infinite loops
    
    while (negatives.length < count && attempts < maxAttempts) {
      const idx = this.negSamplingTable[Math.floor(Math.random() * this.negSamplingTable.length)];
      if (idx !== excludeIndex && !negatives.includes(idx)) {
        negatives.push(idx);
      }
      attempts++;
    }
    
    // If we couldn't find enough unique negatives, fill with random indices
    while (negatives.length < count) {
      const randomIdx = Math.floor(Math.random() * this.vocabSize);
      if (randomIdx !== excludeIndex && !negatives.includes(randomIdx)) {
        negatives.push(randomIdx);
      }
    }
    
    return negatives;
  }

  /**
   * Generate CBOW training pairs from tokens with optional subsampling (SPARSE VERSION)
   * @param {string[]} tokens - Array of tokens
   * @param {number} windowSize - Size of the context window
   * @param {boolean} useSubsampling - Whether to apply subsampling (default true)
   * @param {number} subsampleThreshold - Subsampling threshold (default 1e-5)
   * @returns {Array} Array of training pairs [context_indices, target_index]
   */
  generateTrainingPairs(tokens = this.tokens, windowSize = this.windowSize, useSubsampling = true, subsampleThreshold = 1e-5) {
    if (!this.wordToIndex || Object.keys(this.wordToIndex).length === 0) {
      throw new Error('Vocabulary not built. Please run buildVocab() first.');
    }

    if (!tokens || tokens.length === 0) {
      throw new Error('No tokens available. Please run cleanText() first.');
    }

    // Apply subsampling if requested (provides the "Speed Multiplier")
    let processedTokens = tokens;
    if (useSubsampling) {
      processedTokens = this.subsampleFrequentWords(tokens, subsampleThreshold);
      console.log(`Using subsampled tokens for training pair generation.`);
    } else {
      console.log(`Using original tokens for training pair generation.`);
    }

    const trainingPairs = [];
    const vocabSize = this.vocabSize;

    for (let i = 0; i < processedTokens.length; i++) {
      const targetWord = processedTokens[i];
      
      // Skip if target word is not in vocabulary
      if (!(targetWord in this.wordToIndex)) {
        continue;
      }

      const targetIndex = this.wordToIndex[targetWord];
      const contextIndices = [];

      // Collect context words from left side
      for (let j = Math.max(0, i - windowSize); j < i; j++) {
        const contextWord = processedTokens[j];
        if (contextWord in this.wordToIndex) {
          contextIndices.push(this.wordToIndex[contextWord]);
        }
      }

      // Collect context words from right side
      for (let j = i + 1; j <= Math.min(processedTokens.length - 1, i + windowSize); j++) {
        const contextWord = processedTokens[j];
        if (contextWord in this.wordToIndex) {
          contextIndices.push(this.wordToIndex[contextWord]);
        }
      }

      // Only add pair if we have context words
      if (contextIndices.length > 0) {
        trainingPairs.push({
          context: contextIndices,
          target: targetIndex
        });
      }
    }

    this.windowSize = windowSize;
    console.log(`Training pairs generated successfully:`);
    console.log(`- Total pairs: ${trainingPairs.length}`);
    console.log(`- Window size: ${windowSize}`);
    console.log(`- Vocabulary size: ${vocabSize}`);
    console.log(`- Subsampling used: ${useSubsampling}`);

    return trainingPairs;
  }

  /**
   * Prepare SPARSE training data (MEMORY OPTIMIZED)
   * @param {Array} trainingPairs - Array of training pairs from generateTrainingPairs
   * @returns {Array} Training data in format [{context_indices: [...], target_index: number}]
   */
  prepareSparseTrainingData(trainingPairs) {
    if (!trainingPairs || trainingPairs.length === 0) {
      throw new Error('No training pairs available. Please run generateTrainingPairs() first.');
    }

    // Calculate memory usage for sparse vs dense representation
    const estimatedDenseMemory = trainingPairs.length * this.vocabSize * 8; // 8 bytes per float64
    const estimatedSparseMemory = trainingPairs.reduce((total, pair) => {
      return total + pair.context.length + 1; // context indices + target index
    }, 0) * 8; // 8 bytes per float64

    console.log(`Memory optimization analysis:`);
    console.log(`- Dense representation would use: ${(estimatedDenseMemory / (1024 * 1024)).toFixed(2)} MB`);
    console.log(`- Sparse representation uses: ${(estimatedSparseMemory / (1024 * 1024)).toFixed(2)} MB`);
    console.log(`- Memory savings: ${((1 - estimatedSparseMemory / estimatedDenseMemory) * 100).toFixed(2)}%`);

    console.log(`Sparse training data prepared:`);
    console.log(`- Total training examples: ${trainingPairs.length}`);
    console.log(`- Average context size: ${(trainingPairs.reduce((sum, pair) => sum + pair.context.length, 0) / trainingPairs.length).toFixed(2)}`);

    return trainingPairs; // Return sparse format directly
  }

  /**
   * Convert sparse training data to dense format for neural network (ON DEMAND)
   * @param {Array} sparseData - Sparse training data
   * @returns {Array} Dense training data in format [{input: [...], output: [...]}]
   */
  convertToDenseFormat(sparseData) {
    if (!this.embeddingLayer) {
      throw new Error('Embedding layer not initialized. Please run initEmbeddingLayer() first.');
    }

    console.log(`Converting sparse data to dense format for neural network...`);
    
    const denseData = sparseData.map(pair => {
      // Get context embeddings and average them
      const contextEmbeddings = this.embeddingLayer.lookup(pair.context);
      const averagedContext = this.embeddingLayer.averageEmbeddings(contextEmbeddings);
      
      // Create one-hot output vector
      const outputVector = new Array(this.vocabSize).fill(0);
      outputVector[pair.target] = 1;

      return {
        input: averagedContext,
        output: outputVector
      };
    });

    console.log(`Conversion complete. Dense data ready for neural network training.`);
    return denseData;
  }

  /**
   * Create input vector using embedding lookup and averaging (PROPER CBOW)
   * @param {number[]} contextIndices - Array of context word indices
   * @returns {number[]} Averaged context embedding vector
   */
  createInputVector(contextIndices) {
    if (!this.embeddingLayer) {
      throw new Error('Embedding layer not initialized. Please run initEmbeddingLayer() first.');
    }

    const contextEmbeddings = this.embeddingLayer.lookup(contextIndices);
    return this.embeddingLayer.averageEmbeddings(contextEmbeddings);
  }

  /**
   * Create one-hot encoded output vector for neural network
   * @param {number} targetIndex - Target word index
   * @returns {number[]} One-hot encoded vector
   */
  createOutputVector(targetIndex) {
    const outputVector = new Array(this.vocabSize).fill(0);
    outputVector[targetIndex] = 1;
    return outputVector;
  }

  /**
   * Extract word embeddings from trained neural network
   * @param {NeuralNetwork} neuralNetwork - Trained neural network instance
   * @returns {Object} Word embeddings object
   */
  extractEmbeddings(neuralNetwork) {
    if (!neuralNetwork || !neuralNetwork.weights || neuralNetwork.weights.length === 0) {
      throw new Error('Invalid neural network provided.');
    }

    if (this.vocabSize === 0) {
      throw new Error('Vocabulary not built. Cannot extract embeddings.');
    }

    const embeddings = {};
    
    // The embeddings are stored in the first weight matrix (input to hidden)
    const embeddingMatrix = neuralNetwork.weights[0];

    Object.keys(this.wordToIndex).forEach(word => {
      const index = this.wordToIndex[word];
      embeddings[word] = embeddingMatrix[index];
    });

    console.log(`Word embeddings extracted successfully:`);
    console.log(`- Vocabulary size: ${this.vocabSize}`);
    console.log(`- Embedding dimension: ${embeddingMatrix[0].length}`);

    return embeddings;
  }

  /**
   * Get word similarity using cosine similarity
   * @param {Object} embeddings - Word embeddings object
   * @param {string} word1 - First word
   * @param {string} word2 - Second word
   * @returns {number} Cosine similarity score
   */
  getWordSimilarity(embeddings, word1, word2) {
    if (!(word1 in embeddings) || !(word2 in embeddings)) {
      throw new Error(`One or both words not found in embeddings: ${word1}, ${word2}`);
    }

    const vec1 = embeddings[word1];
    const vec2 = embeddings[word2];

    // Calculate dot product
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    return similarity;
  }

  /**
   * Find most similar words to a given word
   * @param {Object} embeddings - Word embeddings object
   * @param {string} targetWord - Target word
   * @param {number} topN - Number of similar words to return
   * @returns {Array} Array of {word, similarity} objects
   */
  findSimilarWords(embeddings, targetWord, topN = 10) {
    if (!(targetWord in embeddings)) {
      throw new Error(`Target word "${targetWord}" not found in embeddings.`);
    }

    const targetEmbedding = embeddings[targetWord];
    const similarities = [];

    Object.keys(embeddings).forEach(word => {
      if (word !== targetWord) {
        try {
          const similarity = this.getWordSimilarity(embeddings, targetWord, word);
          similarities.push({ word, similarity });
        } catch (error) {
          // Skip words that can't be compared
        }
      }
    });

    // Sort by similarity (descending) and return top N
    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, topN);
  }

  /**
   * Save preprocessed data to files
   * @param {string} basePath - Base path for saving files
   */
  savePreprocessedData(basePath) {
    // Save vocabulary
    const vocabData = {
      vocab: this.vocab,
      wordToIndex: this.wordToIndex,
      indexToWord: this.indexToWord,
      vocabSize: this.vocabSize,
      minCount: this.minCount,
      windowSize: this.windowSize
    };

    fs.writeFileSync(`${basePath}_vocab.json`, JSON.stringify(vocabData, null, 2));
    console.log(`Vocabulary saved to ${basePath}_vocab.json`);

    // Save tokens
    fs.writeFileSync(`${basePath}_tokens.json`, JSON.stringify(this.tokens, null, 2));
    console.log(`Tokens saved to ${basePath}_tokens.json`);
  }

  /**
   * Load preprocessed data from files
   * @param {string} basePath - Base path where files were saved
   */
  loadPreprocessedData(basePath) {
    try {
      // Load vocabulary
      const vocabData = JSON.parse(fs.readFileSync(`${basePath}_vocab.json`, 'utf8'));
      this.vocab = vocabData.vocab;
      this.wordToIndex = vocabData.wordToIndex;
      this.indexToWord = vocabData.indexToWord;
      this.vocabSize = vocabData.vocabSize;
      this.minCount = vocabData.minCount;
      this.windowSize = vocabData.windowSize;

      // Load tokens
      this.tokens = JSON.parse(fs.readFileSync(`${basePath}_tokens.json`, 'utf8'));

      console.log(`Preprocessed data loaded successfully:`);
      console.log(`- Vocabulary size: ${this.vocabSize}`);
      console.log(`- Total tokens: ${this.tokens.length}`);
      console.log(`- Min count: ${this.minCount}`);
      console.log(`- Window size: ${this.windowSize}`);

    } catch (error) {
      console.error('Error loading preprocessed data:', error.message);
      throw error;
    }
  }

  /**
   * Get vocabulary statistics
   * @returns {Object} Vocabulary statistics
   */
  getVocabStats() {
    if (this.vocabSize === 0) {
      return { message: 'No vocabulary built yet. Please run buildVocab() first.' };
    }

    const totalTokens = this.tokens.length;
    const uniqueTokens = new Set(this.tokens).size;
    const tokensInVocab = this.tokens.filter(token => token in this.wordToIndex).length;
    const coverage = (tokensInVocab / totalTokens) * 100;

    return {
      totalTokens,
      uniqueTokens,
      vocabSize: this.vocabSize,
      tokensInVocab,
      coverage: coverage.toFixed(2) + '%',
      minCount: this.minCount,
      windowSize: this.windowSize
    };
  }

  /**
   * Display vocabulary information
   */
  displayVocabInfo() {
    const stats = this.getVocabStats();
    
    console.log('\n=== VOCABULARY INFORMATION ===');
    if (typeof stats === 'string') {
      console.log(stats);
      return;
    }

    console.log(`Total tokens: ${stats.totalTokens}`);
    console.log(`Unique tokens: ${stats.uniqueTokens}`);
    console.log(`Vocabulary size: ${stats.vocabSize}`);
    console.log(`Tokens in vocabulary: ${stats.tokensInVocab}`);
    console.log(`Vocabulary coverage: ${stats.coverage}`);
    console.log(`Minimum count threshold: ${stats.minCount}`);
    console.log(`Window size: ${stats.windowSize}`);

    console.log('\n=== TOP 20 MOST FREQUENT WORDS ===');
    this.vocab.slice(0, 20).forEach((word, index) => {
      console.log(`${index + 1}. "${word}"`);
    });
  }
}

export { OptimizedTextPreprocessor, EmbeddingLayer };