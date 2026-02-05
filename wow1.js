import fs from 'fs';
import { NeuralNetwork } from './neural-network-2.js';

/**
 * Word2VecSimilarity Class
 *
 * A utility for finding similar words using embeddings extracted directly from
 * the input-to-hidden weights matrix of a trained CBOW model
 */
export class Word2VecSimilarity {

  /**
   * Initialize with a trained network and vocabulary
   * @param {NeuralNetwork} network - Trained neural network
   * @param {Object} vocabulary - Vocabulary mapping object
   */
  constructor(network, vocabulary) {
    this.network = network;
    this.vocabulary = vocabulary;

    // Extract embeddings directly from the input-to-hidden weights matrix
    // In CBOW, the input-to-hidden weights matrix contains the word embeddings
    this.embeddings = this._extractEmbeddings();
  }

  /**
   * Extract embeddings from the input-to-hidden weights matrix
   * @returns {Array} Array of embedding vectors
   */
  _extractEmbeddings() {
    // The input-to-hidden weights matrix is the first weights matrix
    // Each row corresponds to a word in the vocabulary
    return this.network.weights[0];
  }

  /**
   * Get embedding for a word
   * @param {string} word - Word to get embedding for
   * @returns {Array} The embedding vector
   */
  getEmbedding(word) {
    const wordIndex = this.vocabulary.wordToIndex[word];
    if (wordIndex === undefined) {
      throw new Error(`Word "${word}" not in vocabulary`);
    }

    // Get the embedding directly from the weights matrix
    return this.embeddings[wordIndex];
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

    // Handle edge cases
    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    const denominator = magnitudeA * magnitudeB;
    if (denominator < 1e-10) {
      return 0;
    }

    const similarity = dotProduct / denominator;
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

// Load the trained network and vocabulary
const networkData = JSON.parse(fs.readFileSync('goals_embedding_network.json', 'utf8'));
const vocabulary = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));

// Create neural network instance and load the trained weights
const network = new NeuralNetwork(networkData.config);
network.weights = networkData.weights;
network.biases = networkData.biases;

// Create the similarity checker
const similarityChecker = new Word2VecSimilarity(network, vocabulary);

// Test words to find similarities for
const testWords = ['success', 'goals', 'achieve', 'money', 'happiness'];

// Find and display similar words for each test word
testWords.forEach(word => {
  try {
    console.log(`\nWords similar to "${word}":`);
    const similarWords = similarityChecker.findSimilarWords(word, 10);

    similarWords.forEach((similarWord, i) => {
      console.log(`${i + 1}. ${similarWord.word} (similarity: ${similarWord.similarity.toFixed(4)})`);
    });
  } catch (e) {
    console.log(`\nCould not find similarities for "${word}": ${e.message}`);
  }
});

console.log('\nDemo complete!');