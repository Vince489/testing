import fs from 'fs';
import readline from 'readline';

// 1. Setup & Data Loading
const networkData = JSON.parse(fs.readFileSync('./goals_embedding_network.json', 'utf8'));
const vocabData = JSON.parse(fs.readFileSync('./goals_vocabulary.json', 'utf8'));
const bookText = fs.readFileSync('./Goals-Brian-Tracy.txt', 'utf8'); 

const wordToIndex = vocabData.wordToIndex;
const weights = networkData.weights[0]; 
const dimensions = 20;

// Filter out noise (page numbers/short fragments) during vectorization
const sentences = bookText.split(/[.!?]+/).filter(s => {
    const text = s.trim();
    const digitCount = (text.match(/\d/g) || []).length;
    return text.length > 30 && digitCount < 5; 
});

// Helper functions (same as your successful 'try.js')
function getWordVector(word) {
    const index = wordToIndex[word.toLowerCase()];
    if (index === undefined) return null;
    return weights[index];
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return (normA === 0 || normB === 0) ? 0 : dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 2. Pre-calculate sentence embeddings for speed
console.log(`Analyzing ${sentences.length} lessons from Brian Tracy...`);
const sentenceEmbeddings = sentences.map(sentence => {
    const words = sentence.toLowerCase().match(/\b(\w+)\b/g) || [];
    let sentenceVec = new Array(dimensions).fill(0);
    let count = 0;
    words.forEach(word => {
        const vec = getWordVector(word);
        if (vec) {
            for (let i = 0; i < dimensions; i++) sentenceVec[i] += vec[i];
            count++;
        }
    });
    if (count > 0) sentenceVec = sentenceVec.map(v => v / count);
    return { text: sentence.trim().replace(/\s+/g, ' '), vector: sentenceVec };
});

// 3. The Search Engine Logic
function getAdvice(query, topK = 3) {
    const queryWords = query.toLowerCase().match(/\b(\w+)\b/g) || [];
    let queryVec = new Array(dimensions).fill(0);
    let count = 0;
    queryWords.forEach(word => {
        const vec = getWordVector(word);
        if (vec) {
            for (let i = 0; i < dimensions; i++) queryVec[i] += vec[i];
            count++;
        }
    });
    if (count === 0) return null;
    queryVec = queryVec.map(v => v / count);

    return sentenceEmbeddings
        .map(s => ({ text: s.text, score: cosineSimilarity(queryVec, s.vector) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
}

// 4. Interactive Terminal Interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const askQuestion = () => {
    console.log("\n" + "=".repeat(50));
    rl.question('What is your goal or challenge? (Type "exit" to quit): ', (answer) => {
        if (answer.toLowerCase() === 'exit') {
            console.log("Good luck reaching your goals!");
            rl.close();
            return;
        }

        const results = getAdvice(answer);

        if (!results) {
            console.log("\nI don't recognize those words. Try asking about success, habits, or planning.");
        } else {
            console.log(`\nBrian Tracy's top advice for "${answer}":`);
            results.forEach((res, i) => {
                console.log(`\n[${i + 1}] (Match: ${(res.score * 100).toFixed(1)}%)`);
                console.log(`> ${res.text}`);
            });
        }
        askQuestion(); // Loop back for the next question
    });
};

console.log("\n--- Goal-Seeker AI Consultant Loaded ---");
askQuestion();