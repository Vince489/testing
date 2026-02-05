import fs from 'fs';
import TextPreprocessor from './text-preprocessor.js';
import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

function performVectorAnalogy(embeddings, wordA, wordB, wordC) {
    const vecA = embeddings[wordA];
    const vecB = embeddings[wordB];
    const vecC = embeddings[wordC];

    if (!vecA || !vecB || !vecC) return `Error: Words not found.`;

    const resultVec = vecA.map((val, i) => val - vecB[i] + vecC[i]);
    return getClosestWords(resultVec, embeddings);
}

/**
 * Finds words that are mathematically the "opposite" 
 * (Negative Cosine Similarity)
 */
function findOppositeWord(embeddings, targetWord, topN = 5) {
    const targetVec = embeddings[targetWord];
    if (!targetVec) return `Error: ${targetWord} not found.`;

    const similarities = [];
    Object.keys(embeddings).forEach(word => {
        const wordVec = embeddings[word];
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        for (let i = 0; i < wordVec.length; i++) {
            dotProduct += targetVec[i] * wordVec[i];
            norm1 += targetVec[i] * targetVec[i];
            norm2 += wordVec[i] * wordVec[i];
        }
        const denom = (Math.sqrt(norm1) * Math.sqrt(norm2));
        const similarity = denom === 0 ? 0 : dotProduct / denom;
        similarities.push({ word, similarity });
    });

    // Sort ascending (most negative first)
    return similarities.sort((a, b) => a.similarity - b.similarity).slice(0, topN);
}

function getClosestWords(targetVec, embeddings, topN = 5) {
    const similarities = [];
    Object.keys(embeddings).forEach(word => {
        const wordVec = embeddings[word];
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        for (let i = 0; i < wordVec.length; i++) {
            dotProduct += targetVec[i] * wordVec[i];
            norm1 += targetVec[i] * targetVec[i];
            norm2 += wordVec[i] * wordVec[i];
        }
        const denom = (Math.sqrt(norm1) * Math.sqrt(norm2));
        const similarity = denom === 0 ? 0 : dotProduct / denom;
        similarities.push({ word, similarity });
    });
    return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, topN);
}

async function runAnalogyDemo() {
    const processor = new TextPreprocessor();
    processor.cleanText('./Goals-Brian-Tracy.txt');
    processor.buildVocab();
    
    // Load the network using the binary format
    const network = await OptimizedNeuralNetwork.load('./goals_embedding_network.json');
    
    // Extract embeddings from the first layer
    const inputLayer = network.layers[0];
    const vocabSize = processor.vocab.length;
    const embeddingDim = 20;
    
    const embeddings = {};
    processor.vocab.forEach((word, index) => {
        const wordEmbedding = [];
        for (let j = 0; j < embeddingDim; j++) {
            wordEmbedding.push(inputLayer.weights[index * embeddingDim + j]);
        }
        embeddings[word] = wordEmbedding;
    });
runAnalogyDemo().catch(err => console.error(err));