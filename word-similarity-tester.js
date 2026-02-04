import fs from 'fs';
import readline from 'readline';
import { NeuralNetwork } from './neural-network-2.js';

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

  findMostSimilar(word) {
    const targetEmbedding = this.getEmbedding(word);
    let maxSimilarity = -1;
    let mostSimilarWord = null;

    for (const vocabWord of this.vocabulary.vocab) {
      if (vocabWord === word) continue;

      try {
        const embedding = this.getEmbedding(vocabWord);
        const similarity = this.cosineSimilarity(targetEmbedding, embedding);
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
          mostSimilarWord = vocabWord;
        }
      } catch (e) {
        // Skip words not in vocabulary
      }
    }

    return { word: mostSimilarWord, similarity: maxSimilarity };
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
      const result = similarityChecker.findMostSimilar(input);
      if (result.word) {
        console.log(`Most similar to "${input}": "${result.word}" (similarity: ${result.similarity.toFixed(4)})`);
      } else {
        console.log(`Could not find a similar word for "${input}"`);
      }
    } catch (e) {
      console.log(`Error: ${e.message}`);
    }
  }
});