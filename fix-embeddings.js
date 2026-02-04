import fs from 'fs';
import { NeuralNetwork } from './neural-network-2.js';
import { EmbeddingSimilarity } from './embed.js';

// Load the trained network and vocabulary
const networkData = JSON.parse(fs.readFileSync('goals_embedding_network.json', 'utf8'));
const vocabulary = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));

// Create neural network instance and load the trained weights
const network = new NeuralNetwork(networkData.config);
network.weights = networkData.weights;
network.biases = networkData.biases;

// Create a modified version of the EmbeddingSimilarity class that normalizes the embeddings
class FixedEmbeddingSimilarity extends EmbeddingSimilarity {
  /**
   * Get normalized embedding for a word
   * @param {string} word - Word to get embedding for
   * @returns {Array} The normalized embedding vector
   */
  getNormalizedEmbedding(word) {
    const embedding = this.getEmbedding(word);
    return this.normalize(embedding);
  }

  /**
   * Find most similar words to a given word using normalized embeddings
   * @param {string} word - Target word
   * @param {number} topN - Number of similar words to return
   * @returns {Array} Array of {word, similarity} objects
   */
  findSimilarWords(word, topN = 5) {
    const targetEmbedding = this.getNormalizedEmbedding(word);
    const similarities = [];

    // Compare with all words in vocabulary
    for (const vocabWord of this.vocabulary.vocab) {
      if (vocabWord === word) continue;

      try {
        const embedding = this.getNormalizedEmbedding(vocabWord);
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

// Create the similarity checker with our fixed class
const similarityChecker = new FixedEmbeddingSimilarity(network, vocabulary);

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