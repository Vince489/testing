import fs from 'fs';

// 1. Load your specific project files
const networkData = JSON.parse(fs.readFileSync('./goals_embedding_network.json', 'utf8'));
const vocabData = JSON.parse(fs.readFileSync('./goals_vocabulary.json', 'utf8'));
const bookText = fs.readFileSync('./Goals-Brian-Tracy.txt', 'utf8'); 

// Access the wordToIndex object directly from your JSON
const wordToIndex = vocabData.wordToIndex;
const weights = networkData.weights[0]; 
const dimensions = 20;

console.log(`Loaded vocabulary with ${vocabData.vocabSize} words.`);

// Helper: Get vector for a single word using Object lookup
function getWordVector(word) {
    const index = wordToIndex[word.toLowerCase()];
    // Check if the word exists (index could be 0, so we check for undefined)
    if (index === undefined) return null;
    return weights[index];
}

// Helper: Cosine Similarity
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 2. Vectorize the book
// Split by sentence, cleaning up newlines and extra spaces
const sentences = bookText.split(/[.!?]+/).filter(s => s.trim().length > 20);

console.log(`--- Vectorizing ${sentences.length} sentences ---`);

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

    if (count > 0) {
        sentenceVec = sentenceVec.map(v => v / count); 
    }
    return { text: sentence.trim().replace(/\s+/g, ' '), vector: sentenceVec };
});

// 3. Search Function
function semanticSearch(query, topK = 5) {
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

    if (count === 0) return [{ text: "No matches found for your query terms.", score: 0 }];
    queryVec = queryVec.map(v => v / count);

    return sentenceEmbeddings
        .map(s => ({
            text: s.text,
            score: cosineSimilarity(queryVec, s.vector)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
}

// --- RUN ---
const myQuery = "how do i succeed in life"; 
console.log(`\nSearching Brian Tracy for: "${myQuery}"`);
console.table(semanticSearch(myQuery));