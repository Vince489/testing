import fs from 'fs';
import readline from 'readline';
import { NeuralNetwork } from './neural-network-3.js';

// Load the trained network and vocabulary
const networkData = JSON.parse(fs.readFileSync('goals_embedding_network.json', 'utf8'));
const vocabulary = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));

// Create neural network instance and load the trained weights
const network = new NeuralNetwork(networkData.config);
network.weights = networkData.weights;
network.biases = networkData.biases;

// Word2VecSimilarity class (simplified version)
class Word2VecSimilarity {
  constructor(network, vocabulary) {
    this.network = network;
    this.vocabulary = vocabulary;
    this.embeddings = this._extractEmbeddings();
    
    // Define stop words to filter out of the results
    this.stopWords = new Set([
      'the', 'you', 'and', 'to', 'of', 'your', 'in', 'a', 'that', 'is', 
      'it', 'are', 'or', 'be', 'what', 'for', 'will', 'have', 'on', 'as',
      'this', 'with', 'they', 'do', 'i', 'more', 'can', 'if', 'at',
      'all', 'would', 'every', 'by', 'not', 'he', 'about', 'was', 'when', 'from',
      'an', 'how', 'than', 'them', 'their', 'there', 'then',
      'one', 'into', 'any', 'could', 'who', 'had', 'no', 'like', 'my', 'his', 'but', 'we', 'these', 'other',
      'each', 'up', 'out', 'were', 'over', 'before', 'very', 'down'
    ]);
  }

  _extractEmbeddings() {
    return this.network.weights[0];
  }

  getEmbedding(word) {
    const wordIndex = this.vocabulary.wordToIndex[word];
    if (wordIndex === undefined) {
      throw new Error(`Word "${word}" not in vocabulary`);
    }
    return this.embeddings[wordIndex];
  }

  cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magnitudeA += vecA[i] * vecA[i];
      magnitudeB += vecB[i] * vecB[i];
    }

    if (magnitudeA === 0 || magnitudeB === 0) return 0;

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    const similarity = dotProduct / (magnitudeA * magnitudeB);
    return Math.max(-1, Math.min(1, similarity));
  }

  findTopSimilar(word, limit = 5) {
    const targetEmbedding = this.getEmbedding(word);
    let results = [];

    for (const vocabWord of this.vocabulary.vocab) {
      // Skip the word itself AND skip stop words
      if (vocabWord === word || this.stopWords.has(vocabWord)) continue;

      try {
        const embedding = this.getEmbedding(vocabWord);
        const similarity = this.cosineSimilarity(targetEmbedding, embedding);
        results.push({ word: vocabWord, similarity });
      } catch (e) {
        continue;
      }
    }

    // Sort by highest similarity first and take the top N
    return results.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
  }
}
// Create the similarity checker
const similarityChecker = new Word2VecSimilarity(network, vocabulary);

// Interactive CLI
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log("Word Similarity Tester - Type a word to find its most similar word");
console.log("Type 'exit' to quit");

rl.on('line', (input) => {
  if (input.toLowerCase() === 'exit') {
    console.log("Goodbye!");
    rl.close();
  } else {
    try {
      const results = similarityChecker.findTopSimilar(input.toLowerCase().trim(), 5);
      
      if (results.length > 0) {
        console.log(`\nTop 5 words similar to "${input}":`);
        results.forEach((res, i) => {
          console.log(`  ${i + 1}. ${res.word} (similarity: ${res.similarity.toFixed(4)})`);
        });
        console.log(''); // New line for spacing
      } else {
        console.log(`Could not find similar words for "${input}"`);
      }
    } catch (e) {
      console.log(`Error: ${e.message}`);
    }
  }
});