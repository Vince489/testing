import fs from 'fs';
import { NeuralNetwork } from './neural-network-2.js'; // Using neural-network-2.js as in word2vec-similarity.js
import { Word2VecSimilarity } from './wow1.js'; // Import the Word2VecSimilarity class

// Load the trained network and vocabulary
const networkData = JSON.parse(fs.readFileSync('goals_embedding_network.json', 'utf8'));
const vocabulary = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));

// Create neural network instance and load the trained weights
const network = new NeuralNetwork(networkData.config);
network.weights = networkData.weights;
network.biases = networkData.biases;

// Create the Word2Vec similarity checker
const similarityChecker = new Word2VecSimilarity(network, vocabulary);

// Test words to find similarities for
const testWords = ['success', 'goals', 'achieve', 'money', 'happiness'];

// Find and display similar words for each test word with limited output
testWords.forEach(word => {
  try {
    console.log(`\nTop 5 words similar to "${word}":`);
    const similarWords = similarityChecker.findSimilarWords(word, 5);

    similarWords.forEach((similarWord, i) => {
      console.log(`${i + 1}. ${similarWord.word} (similarity: ${similarWord.similarity.toFixed(6)})`);
    });
  } catch (e) {
    console.log(`\nCould not find similarities for "${word}": ${e.message}`);
  }
});

console.log("\nDemo complete!");
