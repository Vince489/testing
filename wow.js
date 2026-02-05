import fs from 'fs';
import TextPreprocessor from './text-preprocessor.js';
import { OptimizedNeuralNetwork } from './optimized-neural-network.js';

/**
 * Helper function to perform vector arithmetic
 * Result = vecA - vecB + vecC
 */
function performVectorAnalogy(embeddings, wordA, wordB, wordC) {
    const vecA = embeddings[wordA];
    const vecB = embeddings[wordB];
    const vecC = embeddings[wordC];

    if (!vecA || !vecB || !vecC) {
        const missing = [];
        if (!vecA) missing.push(wordA);
        if (!vecB) missing.push(wordB);
        if (!vecC) missing.push(wordC);
        return `Error: Words missing from vocab: ${missing.join(', ')}`;
    }

    // Vector math: result = A - B + C
    const resultVec = vecA.map((val, i) => val - vecB[i] + vecC[i]);

    const queryWords = [wordA, wordB, wordC];
    const similarities = [];

    Object.keys(embeddings).forEach(word => {
        // Skip words that were part of the input equation to see NEW relationships
        if (queryWords.includes(word)) return;

        const wordVec = embeddings[word];
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < wordVec.length; i++) {
            dotProduct += resultVec[i] * wordVec[i];
            norm1 += resultVec[i] * resultVec[i];
            norm2 += wordVec[i] * wordVec[i];
        }
        
        const denom = (Math.sqrt(norm1) * Math.sqrt(norm2));
        const similarity = denom === 0 ? 0 : dotProduct / denom;
        
        similarities.push({ word, similarity });
    });

    return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, 5);
}

async function runAnalogyDemo() {
    const processor = new TextPreprocessor();
    const filePath = './Goals-Brian-Tracy.txt';
    const networkPath = './goals_embedding_network.json';

    console.log("--- Initializing Preprocessing ---");
    processor.cleanText(filePath);
    processor.buildVocab();
    
    if (!fs.existsSync(networkPath)) {
        console.error(`Error: ${networkPath} not found!`);
        return;
    }

    console.log("Loading Network Weights...");
    const network = await OptimizedNeuralNetwork.load(networkPath);
    
    // Get the weights from the loaded network
    const weights = network.getAllWeightsAsFloat32();
    const embeddings = {};

    console.log(`Mapping ${processor.vocab.length} words to embedding weights...`);
    
    processor.vocab.forEach((word, index) => {
        // Extract the embedding vector for this word from the weights
        const embeddingSize = 20; // Assuming 20-dimensional embeddings
        const wordEmbedding = [];
        for (let i = 0; i < embeddingSize; i++) {
            wordEmbedding.push(weights[index * embeddingSize + i]);
        }
        embeddings[word] = wordEmbedding;
    });

    console.log(`✓ Mapping complete. Found ${Object.keys(embeddings).length} embeddings.`);

    // Run the actual tests
    console.log("\n=== TEST 1: The 'Action' Analogy ===");
    console.log("Equation: Success - Work + Thinking = ?");
    const test1 = performVectorAnalogy(embeddings, "success", "work", "thinking");
    console.log(test1);

    console.log("\n=== TEST 2: The 'Plurality' Analogy ===");
    console.log("Equation: Goals - Goal + Success = ?");
    const test2 = performVectorAnalogy(embeddings, "goals", "goal", "success");
    console.log(test2);

    console.log("\n=== TEST 3: The 'Life' Analogy ===");
    console.log("Equation: Money - Business + Life = ?");
    const test3 = performVectorAnalogy(embeddings, "money", "business", "life");
    console.log(test3);
}

// Start the demo
runAnalogyDemo().catch(err => {
    console.error("Critical Error during demo:");
    console.error(err);
});