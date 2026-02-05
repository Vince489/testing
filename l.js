import fs from 'fs';

// 1. Load your actual vocabulary from your previous project files
const vocabData = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));
const vocabulary = vocabData.vocab; // This is the array of 1360 words

// 2. Define Brian Tracy Intents (Sample Data)
const intentData = {
    "GoalSetting": [
        "write down your goals clearly",
        "make a list of everything you want to achieve",
        "set a deadline for your major goal",
        "plan every day in advance"
    ],
    "TimeManagement": [
        "use your time effectively",
        "focus on your most important tasks",
        "the biggest waste of time is doing things that don't matter",
        "discipline yourself to work on one thing at a time"
    ],
    "FinancialSuccess": [
        "increase your earning ability",
        "money is a result of the value you create",
        "become a financial success in your business",
        "save ten percent of your income"
    ]
};

// --- THE FUNCTION ---
function initializeWeightsFromBayes(intentData, vocabulary) {
    const numIntents = Object.keys(intentData).length;
    const vocabSize = vocabulary.length;
    const alpha = 1; // Laplace Smoothing

    let weights = Array.from({ length: numIntents }, () => new Array(vocabSize).fill(0));
    let biases = new Array(numIntents).fill(0);

    const intents = Object.keys(intentData);
    const totalDocs = Object.values(intentData).flat().length;

    intents.forEach((intent, j) => {
        const sentences = intentData[intent];
        const wordCounts = {};
        let totalWordsInIntent = 0;

        biases[j] = Math.log(sentences.length / totalDocs);

        sentences.forEach(sent => {
            sent.toLowerCase().split(/\W+/).forEach(word => {
                if (vocabulary.includes(word)) {
                    wordCounts[word] = (wordCounts[word] || 0) + 1;
                    totalWordsInIntent++;
                }
            });
        });

        vocabulary.forEach((word, i) => {
            const count = wordCounts[word] || 0;
            weights[j][i] = Math.log((count + alpha) / (totalWordsInIntent + alpha * vocabSize));
        });
    });

    return { weights, biases, intents };
}

// 3. EXECUTION
const result = initializeWeightsFromBayes(intentData, vocabulary);

// 4. LOG THE RESULTS
console.log("=== Multinomial Naive Bayes to NN Weights ===\n");
console.log(`Intents mapped: ${result.intents.join(", ")}`);
console.log(`Vocab Size: ${vocabulary.length}`);
console.log(`Bias for ${result.intents[0]}: ${result.biases[0].toFixed(4)}`);

// Show the top weight for "GoalSetting"
const goalWeights = result.weights[0];
const topWeightIndex = goalWeights.indexOf(Math.max(...goalWeights));
console.log(`Top weight for GoalSetting: "${vocabulary[topWeightIndex]}" (${goalWeights[topWeightIndex].toFixed(4)})`);

// 5. Save for your Neural Network
fs.writeFileSync('intent_weights.json', JSON.stringify({
    weights: result.weights,
    biases: result.biases,
    intents: result.intents
}, null, 2));

console.log("\nSaved intent_weights.json! You can now load this into your Neural Network.");