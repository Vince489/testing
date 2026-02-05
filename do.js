import fs from 'fs';

// 1. Load your assets
const modelData = JSON.parse(fs.readFileSync('intent_weights.json', 'utf8'));
const vocabData = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));

const { weights, biases, intents } = modelData;
const { wordToIndex } = vocabData;

// 2. The Prediction Function
function predictIntent(sentence) {
    // a. Tokenize and clean the input
    const tokens = sentence.toLowerCase().split(/\W+/).filter(t => wordToIndex[t] !== undefined);
    
    if (tokens.length === 0) return { intent: "Unknown", confidence: 0 };

    // b. Create a "Score" for each intent
    // We sum the Bias + the Weights of every word found in the sentence
    const scores = intents.map((intent, i) => {
        let score = biases[i]; // Start with the Prior (Bias)
        
        tokens.forEach(token => {
            const wordIdx = wordToIndex[token];
            score += weights[i][wordIdx]; // Add the Log-Likelihood (Weight)
        });
        
        return score;
    });

    // c. Find the winner (the highest score / least negative)
    const maxScore = Math.max(...scores);
    const winnerIdx = scores.indexOf(maxScore);

    return {
        intent: intents[winnerIdx],
        score: maxScore.toFixed(4)
    };
}

// 3. Test it out
const testSentences = [
    "I need help to create a plan",
    "How can I save more money and build a business?",
    "the consequences of doing or not doing something"
];

console.log("=== Brian Tracy Intent Classification ===\n");
testSentences.forEach(s => {
    const result = predictIntent(s);
    console.log(`Input: "${s}"`);
    console.log(`Predicted Intent: ${result.intent} (Score: ${result.score})\n`);
});