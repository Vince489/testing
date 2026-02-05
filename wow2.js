import fs from 'fs';
import TextPreprocessor from './text-preprocessor.js'; 

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
    
    const networkData = JSON.parse(fs.readFileSync('./goals_embedding_network.json', 'utf8'));
    const embeddings = {};
    const weightMatrix = networkData.weights[0];
    processor.vocab.forEach((word, index) => { if (weightMatrix[index]) embeddings[word] = weightMatrix[index]; });

    console.log("\n=== TEST 1: The 'Action' Analogy ===");
    console.log("Leaders - People + Goal = ?");
    console.log(performVectorAnalogy(embeddings, "leaders", "people", "goal"));

    console.log("\n=== TEST 4: The 'Opposite' Test ===");
    console.log("What is the opposite of 'Success' in this model?");
    console.log(findOppositeWord(embeddings, "success"));

    console.log("\nWhat is the opposite of 'Force'?");
    console.log(findOppositeWord(embeddings, "force"));
}

runAnalogyDemo().catch(err => console.error(err));