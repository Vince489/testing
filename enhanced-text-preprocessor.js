import fs from 'fs';

/**
 * EnhancedTextPreprocessor Class
 * 
 * An enhanced version of the original TextPreprocessor with additional features:
 * - Progress indicators for long operations
 * - Memory usage monitoring
 * - Configurable preprocessing options
 * - Enhanced error handling and logging
 */

class EnhancedTextPreprocessor {
  constructor(options = {}) {
    this.tokens = [];
    this.vocab = [];
    this.wordToIndex = {};
    this.indexToWord = {};
    this.vocabSize = 0;
    this.minCount = options.minCount || 2;
    this.windowSize = options.windowSize || 5;
    this.embeddingDim = options.embeddingDim || 100;
    this.progressCallback = options.progressCallback || null;
    
    // Define stop words to exclude from vocabulary
    this.stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
      'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
      'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'then', 'here', 'there', 'up', 'down', 'out',
      'over', 'under', 'again', 'further', 'then', 'once', 'here', 'when', 'where', 'how', 'why', 'what', 'which', 'who', 'whom', 'whose'
    ]);

    // Text cleaning options
    this.cleaningOptions = {
      preserveNumbers: options.preserveNumbers !== false, // Default true
      preserveHyphens: options.preserveHyphens !== false, // Default true
      preserveApostrophes: options.preserveApostrophes || false, // Default false
      minWordLength: options.minWordLength || 1, // Default 1
      maxWordLength: options.maxWordLength || Infinity, // Default no limit
      customStopWords: options.customStopWords || []
    };

    // Merge custom stop words
    this.stopWords = new Set([...this.stopWords, ...this.cleaningOptions.customStopWords]);

    // Memory monitoring
    this.memoryUsage = {
      peakMemory: 0,
      currentMemory: 0,
      operations: []
    };
  }

  /**
   * Update progress callback
   * @param {string} operation - Current operation
   * @param {number} progress - Progress percentage (0-100)
   * @param {string} message - Additional message
   */
  updateProgress(operation, progress, message = '') {
    if (this.progressCallback) {
      this.progressCallback({
        operation,
        progress,
        message,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Monitor memory usage
   * @param {string} operation - Operation name
   */
  monitorMemory(operation) {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      const memUsage = process.memoryUsage();
      this.memoryUsage.currentMemory = memUsage.heapUsed;
      this.memoryUsage.peakMemory = Math.max(this.memoryUsage.peakMemory, memUsage.heapUsed);
      
      this.memoryUsage.operations.push({
        operation,
        memory: memUsage.heapUsed,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Get memory usage statistics
   * @returns {Object} Memory usage statistics
   */
  getMemoryStats() {
    return {
      peakMemoryMB: (this.memoryUsage.peakMemory / (1024 * 1024)).toFixed(2),
      currentMemoryMB: (this.memoryUsage.currentMemory / (1024 * 1024)).toFixed(2),
      operations: this.memoryUsage.operations
    };
  }

  /**
   * Load and clean text from a file with progress monitoring
   * @param {string} filePath - Path to the text file
   * @returns {string[]} Array of cleaned tokens
   */
  cleanText(filePath) {
    try {
      this.updateProgress('Loading file', 0, `Reading ${filePath}`);
      const rawText = fs.readFileSync(filePath, 'utf8');
      
      this.updateProgress('Cleaning text', 10, 'Converting to lowercase...');
      const lowerText = rawText.toLowerCase();
      
      this.updateProgress('Cleaning text', 30, 'Removing unwanted characters...');
      
      // Build regex based on cleaning options
      let pattern = '[^a-z';
      if (this.cleaningOptions.preserveNumbers) pattern += '0-9';
      if (this.cleaningOptions.preserveHyphens) pattern += '-';
      if (this.cleaningOptions.preserveApostrophes) pattern += "'";
      pattern += '\\s]';
      
      const cleanedText = lowerText.replace(new RegExp(pattern, 'g'), '');
      
      this.updateProgress('Cleaning text', 70, 'Tokenizing...');
      const tokens = cleanedText.split(/\s+/).filter(word => 
        word.length >= this.cleaningOptions.minWordLength && 
        word.length <= this.cleaningOptions.maxWordLength &&
        word.length > 0
      );
      
      this.tokens = tokens;
      
      this.updateProgress('Cleaning text', 100, `Completed: ${tokens.length} tokens`);
      this.monitorMemory('cleanText');
      
      console.log(`Text cleaned successfully. Total tokens: ${tokens.length}`);
      return tokens;
    } catch (error) {
      console.error(`Error reading file ${filePath}:`, error.message);
      throw error;
    }
  }

  /**
   * Build vocabulary from tokens with frequency filtering and progress monitoring
   * @param {string[]} tokens - Array of tokens
   * @param {number} minCount - Minimum frequency count for words to be included
   * @returns {Object} Vocabulary information
   */
  buildVocab(tokens = this.tokens, minCount = this.minCount) {
    if (!tokens || tokens.length === 0) {
      throw new Error('No tokens available. Please run cleanText() first.');
    }

    this.updateProgress('Building vocabulary', 0, 'Counting word frequencies...');
    
    // Count word frequencies
    const counts = {};
    const totalTokens = tokens.length;
    
    tokens.forEach((word, index) => {
      counts[word] = (counts[word] || 0) + 1;
      
      // Update progress every 10%
      if (index % Math.floor(totalTokens / 10) === 0) {
        this.updateProgress('Building vocabulary', Math.floor((index / totalTokens) * 50), `Processing token ${index}/${totalTokens}`);
      }
    });

    this.updateProgress('Building vocabulary', 50, 'Filtering words...');
    
    // Filter words by minimum count and exclude stop words
    this.vocab = Object.keys(counts)
      .filter(word => counts[word] >= minCount)
      .filter(word => !this.stopWords.has(word));
    
    // Sort vocabulary by frequency (descending)
    this.vocab.sort((a, b) => counts[b] - counts[a]);

    this.updateProgress('Building vocabulary', 75, 'Creating lookup tables...');
    
    // Create lookup tables
    this.wordToIndex = {};
    this.indexToWord = {};

    this.vocab.forEach((word, index) => {
      this.wordToIndex[word] = index;
      this.indexToWord[index] = word;
    });

    this.vocabSize = this.vocab.length;
    this.minCount = minCount;

    this.updateProgress('Building vocabulary', 100, `Completed: ${this.vocabSize} words`);
    this.monitorMemory('buildVocab');

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
   * Generate CBOW training pairs from tokens with progress monitoring
   * @param {string[]} tokens - Array of tokens
   * @param {number} windowSize - Size of the context window
   * @returns {Array} Array of training pairs [context_indices, target_index]
   */
  generateTrainingPairs(tokens = this.tokens, windowSize = this.windowSize) {
    if (!this.wordToIndex || Object.keys(this.wordToIndex).length === 0) {
      throw new Error('Vocabulary not built. Please run buildVocab() first.');
    }

    if (!tokens || tokens.length === 0) {
      throw new Error('No tokens available. Please run cleanText() first.');
    }

    this.updateProgress('Generating training pairs', 0, 'Processing tokens...');
    
    const trainingPairs = [];
    const vocabSize = this.vocabSize;
    const totalTokens = tokens.length;

    for (let i = 0; i < tokens.length; i++) {
      const targetWord = tokens[i];
      
      // Skip if target word is not in vocabulary
      if (!(targetWord in this.wordToIndex)) {
        continue;
      }

      const targetIndex = this.wordToIndex[targetWord];
      const contextIndices = [];

      // Collect context words from left side
      for (let j = Math.max(0, i - windowSize); j < i; j++) {
        const contextWord = tokens[j];
        if (contextWord in this.wordToIndex) {
          contextIndices.push(this.wordToIndex[contextWord]);
        }
      }

      // Collect context words from right side
      for (let j = i + 1; j <= Math.min(tokens.length - 1, i + windowSize); j++) {
        const contextWord = tokens[j];
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

      // Update progress every 10%
      if (i % Math.floor(totalTokens / 10) === 0) {
        this.updateProgress('Generating training pairs', Math.floor((i / totalTokens) * 100), `Processing token ${i}/${totalTokens}`);
      }
    }

    this.windowSize = windowSize;
    this.updateProgress('Generating training pairs', 100, `Completed: ${trainingPairs.length} pairs`);
    this.monitorMemory('generateTrainingPairs');

    console.log(`Training pairs generated successfully:`);
    console.log(`- Total pairs: ${trainingPairs.length}`);
    console.log(`- Window size: ${windowSize}`);
    console.log(`- Vocabulary size: ${vocabSize}`);

    return trainingPairs;
  }

  /**
   * Prepare training data with memory optimization option
   * @param {Array} trainingPairs - Array of training pairs from generateTrainingPairs
   * @param {boolean} useSparse - Whether to use sparse representation
   * @returns {Array} Training data
   */
  prepareTrainingData(trainingPairs, useSparse = false) {
    if (!trainingPairs || trainingPairs.length === 0) {
      throw new Error('No training pairs available. Please run generateTrainingPairs() first.');
    }

    if (useSparse) {
      return this.prepareSparseTrainingData(trainingPairs);
    } else {
      return this.prepareDenseTrainingData(trainingPairs);
    }
  }

  /**
   * Prepare SPARSE training data (MEMORY OPTIMIZED)
   * @param {Array} trainingPairs - Array of training pairs
   * @returns {Array} Sparse training data
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

    this.monitorMemory('prepareSparseTrainingData');
    return trainingPairs;
  }

  /**
   * Prepare DENSE training data (ORIGINAL APPROACH)
   * @param {Array} trainingPairs - Array of training pairs
   * @returns {Array} Dense training data
   */
  prepareDenseTrainingData(trainingPairs) {
    if (!trainingPairs || trainingPairs.length === 0) {
      throw new Error('No training pairs available. Please run generateTrainingPairs() first.');
    }

    this.updateProgress('Preparing training data', 0, 'Converting to dense format...');
    
    const trainingData = trainingPairs.map((pair, index) => {
      const input = this.createInputVector(pair.context);
      const output = this.createOutputVector(pair.target);
      
      // Update progress every 10%
      if (index % Math.floor(trainingPairs.length / 10) === 0) {
        this.updateProgress('Preparing training data', Math.floor((index / trainingPairs.length) * 100), `Processing pair ${index}/${trainingPairs.length}`);
      }
      
      return {
        input: input,
        output: output
      };
    });

    this.updateProgress('Preparing training data', 100, `Completed: ${trainingData.length} examples`);
    this.monitorMemory('prepareDenseTrainingData');

    console.log(`Training data prepared:`);
    console.log(`- Input vector size: ${trainingData[0].input.length}`);
    console.log(`- Output vector size: ${trainingData[0].output.length}`);
    console.log(`- Total training examples: ${trainingData.length}`);

    return trainingData;
  }

  /**
   * Create one-hot encoded input vector for neural network
   * @param {number[]} contextIndices - Array of context word indices
   * @returns {number[]} One-hot encoded vector
   */
  createInputVector(contextIndices) {
    const inputVector = new Array(this.vocabSize).fill(0);
    
    // For CBOW, we average the context word vectors
    // But for simplicity with the existing NN, we'll create a sparse representation
    contextIndices.forEach(index => {
      inputVector[index] = 1;
    });

    return inputVector;
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

    this.updateProgress('Extracting embeddings', 0, 'Processing neural network weights...');
    
    const embeddings = {};
    
    // The embeddings are stored in the first weight matrix (input to hidden)
    const embeddingMatrix = neuralNetwork.weights[0];

    Object.keys(this.wordToIndex).forEach(word => {
      const index = this.wordToIndex[word];
      embeddings[word] = embeddingMatrix[index];
    });

    this.updateProgress('Extracting embeddings', 100, `Completed: ${Object.keys(embeddings).length} embeddings`);
    this.monitorMemory('extractEmbeddings');

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

    this.updateProgress('Finding similar words', 0, `Searching for words similar to "${targetWord}"...`);
    
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
    const result = similarities.slice(0, topN);

    this.updateProgress('Finding similar words', 100, `Found ${result.length} similar words`);
    this.monitorMemory('findSimilarWords');

    return result;
  }

  /**
   * Save preprocessed data to files with enhanced metadata
   * @param {string} basePath - Base path for saving files
   */
  savePreprocessedData(basePath) {
    // Save vocabulary with enhanced metadata
    const vocabData = {
      vocab: this.vocab,
      wordToIndex: this.wordToIndex,
      indexToWord: this.indexToWord,
      vocabSize: this.vocabSize,
      minCount: this.minCount,
      windowSize: this.windowSize,
      embeddingDim: this.embeddingDim,
      cleaningOptions: this.cleaningOptions,
      memoryStats: this.getMemoryStats(),
      timestamp: new Date().toISOString(),
      version: '2.0'
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
      this.embeddingDim = vocabData.embeddingDim || 100;
      this.cleaningOptions = vocabData.cleaningOptions || this.cleaningOptions;

      // Load tokens
      this.tokens = JSON.parse(fs.readFileSync(`${basePath}_tokens.json`, 'utf8'));

      console.log(`Preprocessed data loaded successfully:`);
      console.log(`- Vocabulary size: ${this.vocabSize}`);
      console.log(`- Total tokens: ${this.tokens.length}`);
      console.log(`- Min count: ${this.minCount}`);
      console.log(`- Window size: ${this.windowSize}`);
      console.log(`- Embedding dimension: ${this.embeddingDim}`);

      if (vocabData.memoryStats) {
        console.log(`- Peak memory usage: ${vocabData.memoryStats.peakMemoryMB} MB`);
      }

    } catch (error) {
      console.error('Error loading preprocessed data:', error.message);
      throw error;
    }
  }

  /**
   * Get vocabulary statistics with enhanced information
   * @returns {Object} Enhanced vocabulary statistics
   */
  getVocabStats() {
    if (this.vocabSize === 0) {
      return { message: 'No vocabulary built yet. Please run buildVocab() first.' };
    }

    const totalTokens = this.tokens.length;
    const uniqueTokens = new Set(this.tokens).size;
    const tokensInVocab = this.tokens.filter(token => token in this.wordToIndex).length;
    const coverage = (tokensInVocab / totalTokens) * 100;

    // Calculate frequency distribution
    const freqCounts = {};
    Object.values(this.wordToIndex).forEach(index => {
      const word = this.indexToWord[index];
      const freq = this.tokens.filter(t => t === word).length;
      freqCounts[freq] = (freqCounts[freq] || 0) + 1;
    });

    return {
      totalTokens,
      uniqueTokens,
      vocabSize: this.vocabSize,
      tokensInVocab,
      coverage: coverage.toFixed(2) + '%',
      minCount: this.minCount,
      windowSize: this.windowSize,
      embeddingDim: this.embeddingDim,
      cleaningOptions: this.cleaningOptions,
      frequencyDistribution: freqCounts,
      memoryStats: this.getMemoryStats()
    };
  }

  /**
   * Display enhanced vocabulary information
   */
  displayVocabInfo() {
    const stats = this.getVocabStats();
    
    console.log('\n=== ENHANCED VOCABULARY INFORMATION ===');
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
    console.log(`Embedding dimension: ${stats.embeddingDim}`);
    console.log(`Peak memory usage: ${stats.memoryStats.peakMemoryMB} MB`);
    
    console.log('\n=== CLEANING OPTIONS ===');
    console.log(`Preserve numbers: ${stats.cleaningOptions.preserveNumbers}`);
    console.log(`Preserve hyphens: ${stats.cleaningOptions.preserveHyphens}`);
    console.log(`Preserve apostrophes: ${stats.cleaningOptions.preserveApostrophes}`);
    console.log(`Min word length: ${stats.cleaningOptions.minWordLength}`);
    console.log(`Max word length: ${stats.cleaningOptions.maxWordLength}`);
    
    console.log('\n=== FREQUENCY DISTRIBUTION ===');
    Object.keys(stats.frequencyDistribution).sort((a, b) => parseInt(a) - parseInt(b)).forEach(freq => {
      console.log(`Words with frequency ${freq}: ${stats.frequencyDistribution[freq]}`);
    });

    console.log('\n=== TOP 20 MOST FREQUENT WORDS ===');
    this.vocab.slice(0, 20).forEach((word, index) => {
      const freq = this.tokens.filter(t => t === word).length;
      console.log(`${index + 1}. "${word}" (freq: ${freq})`);
    });
  }
}

export default EnhancedTextPreprocessor;