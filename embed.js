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
   * @param {Object} options - Configuration options
   * @param {boolean} options.debug - Enable debug output
   */
  constructor(network, vocabulary, options = {}) {
    this.network = network;
    this.vocabulary = vocabulary;
    this.debug = options.debug || false;
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

    if (this.debug) {
      console.log(`Embedding for "${word}": [${result.activations[1].map(v => v.toFixed(6)).join(', ')}]`);
      const magnitude = Math.sqrt(result.activations[1].reduce((sum, val) => sum + val * val, 0));
      console.log(`Embedding magnitude: ${magnitude.toFixed(6)}`);
    }

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

    // Handle edge cases where one or both vectors have zero magnitude
    if (magnitudeA === 0 || magnitudeB === 0) {
      if (this.debug) {
        console.log('Warning: Zero magnitude vector detected');
      }
      return 0;
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    // Handle potential division by zero or very small numbers
    const denominator = magnitudeA * magnitudeB;
    if (denominator < 1e-10) {
      if (this.debug) {
        console.log('Warning: Denominator too small, returning 0');
      }
      return 0;
    }

    const similarity = dotProduct / denominator;

    // Clamp the result to the valid range [-1, 1] to handle floating point errors
    return Math.max(-1, Math.min(1, similarity));
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

        if (this.debug) {
          console.log(`Similarity between "${word}" and "${vocabWord}": ${similarity.toFixed(6)}`);
        }

        similarities.push({ word: vocabWord, similarity });
      } catch (e) {
        // Skip words not in vocabulary (shouldn't happen)
        if (this.debug) {
          console.log(`Error comparing "${word}" and "${vocabWord}": ${e.message}`);
        }
      }
    }

    // Sort by similarity and return top N
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topN);
  }

  /**
   * Normalize an embedding vector to unit length
   * @param {Array} vector - Vector to normalize
   * @returns {Array} Normalized vector
   */
  normalize(vector) {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude === 0) return vector; // Avoid division by zero

    return vector.map(val => val / magnitude);
  }
}

export { EmbeddingSimilarity };
