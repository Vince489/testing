import fs from 'fs';
import TextPreprocessor from './text-preprocessor.js'; 

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

/**
 * Converts a sentence into a single vector by averaging its word vectors
 */
function interpretSentence(sentence, embeddings) {
    const words = sentence.toLowerCase().replace(/[^a-z\s]/g, '').split(/\s+/);
    const vectors = words.map(w => embeddings[w]).filter(v => v !== undefined);

    if (vectors.length === 0) return null;

    // Average the vectors: Sum / Count
    const vecSize = vectors[0].length;
    const avgVec = new Array(vecSize).fill(0);
    
    for (const vec of vectors) {
        for (let i = 0; i < vecSize; i++) {
            avgVec[i] += vec[i];
        }
    }

    return avgVec.map(val => val / vectors.length);
}

async function runAnalogyDemo() {
    const processor = new TextPreprocessor();
    processor.cleanText('./Goals-Brian-Tracy.txt');
    processor.buildVocab();
    
    const networkData = JSON.parse(fs.readFileSync('./goals_embedding_network.json', 'utf8'));
    const embeddings = {};
    const weightMatrix = networkData.weights[0];
    processor.vocab.forEach((word, index) => { if (weightMatrix[index]) embeddings[word] = weightMatrix[index]; });

    console.log("\n=== TEST 5: Sentence Interpretation ===");
    
    // Try a sentence Brian Tracy might agree with
    const sentence1 = "Work hard to get what you want";
    console.log(`Sentence: "${sentence1}"`);
    const vec1 = interpretSentence(sentence1, embeddings);
    console.log("Model interprets this as:", getClosestWords(vec1, embeddings));

    // Try a sentence that is "lazy" to see if it pivots to negative clusters
    const sentence2 = "Goals are primary";
    console.log(`\nSentence: "${sentence2}"`);
    const vec2 = interpretSentence(sentence2, embeddings);
    if (vec2) {
        console.log("Model interprets this as:", getClosestWords(vec2, embeddings));
    } else {
        console.log("Error: None of these words are in the vocabulary.");
    }
}

runAnalogyDemo().catch(err => console.error(err));