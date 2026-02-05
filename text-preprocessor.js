import fs from 'fs';

/**
 * TextPreprocessor Class
 * 
 * A comprehensive text preprocessing module for CBOW (Continuous Bag of Words) model training.
 * This class handles text cleaning, vocabulary building, and training pair generation
 * specifically designed to work with the existing NeuralNetwork class.
 * 
 * Features:
 * - Text cleaning (lowercasing, noise removal, tokenization)
 * - Vocabulary building with frequency counting and filtering
 * - Lookup tables for word-to-index and index-to-word mappings
 * - CBOW training pair generation with configurable window size
 * - Integration with existing NeuralNetwork class for Word2Vec training
 */

class TextPreprocessor {
  constructor() {
    this.tokens = [];
    this.vocab = [];
    this.wordToIndex = {};
    this.indexToWord = {};
    this.vocabSize = 0;
    this.minCount = 2;
    this.windowSize = 5;
    
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
   * Generate CBOW training pairs from tokens
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

    const trainingPairs = [];
    const vocabSize = this.vocabSize;

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
    }

    this.windowSize = windowSize;
    console.log(`Training pairs generated successfully:`);
    console.log(`- Total pairs: ${trainingPairs.length}`);
    console.log(`- Window size: ${windowSize}`);
    console.log(`- Vocabulary size: ${vocabSize}`);

    return trainingPairs;
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
   * Prepare training data in format ready for NeuralNetwork
   * @param {Array} trainingPairs - Array of training pairs from generateTrainingPairs
   * @returns {Array} Training data in format [{input: [...], output: [...]}]
   */
  prepareTrainingData(trainingPairs) {
    if (!trainingPairs || trainingPairs.length === 0) {
      throw new Error('No training pairs available. Please run generateTrainingPairs() first.');
    }

    const trainingData = trainingPairs.map(pair => {
      const input = this.createInputVector(pair.context);
      const output = this.createOutputVector(pair.target);
      
      return {
        input: input,
        output: output
      };
    });

    console.log(`Training data prepared:`);
    console.log(`- Input vector size: ${trainingData[0].input.length}`);
    console.log(`- Output vector size: ${trainingData[0].output.length}`);
    console.log(`- Total training examples: ${trainingData.length}`);

    return trainingData;
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

export default TextPreprocessor;
