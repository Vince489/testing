import fs from 'fs';
import { NeuralNetwork } from './neural-network-3.js';

async function testModel() {
    // 1. Load the vocabulary and the network
    // We use the static load method from your neural-network-3.js class
    const vocabData = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));
    const nn = NeuralNetwork.load('goals_embedding_network.json');

    const { wordToIndex, indexToWord, vocabSize } = vocabData;

    /**
     * Helper to predict the most likely middle word
     * @param {string[]} contextWords - The surrounding words
     */
    function predictMissingWord(contextWords) {
        const inputVector = new Array(vocabSize).fill(0);
        
        // Convert words to indices, filtering out ones not in our 831-word vocab
        const validIndices = contextWords
            .map(w => w.toLowerCase())
            .filter(w => w in wordToIndex)
            .map(w => wordToIndex[w]);

        if (validIndices.length === 0) {
            return "Error: None of these words are in the vocabulary.";
        }

        // Apply CBOW averaging (matching your training script logic)
        validIndices.forEach(idx => {
            inputVector[idx] = 1 / validIndices.length;
        });

        // 2. Perform the forward pass
        const output = nn.predict(inputVector);

        // 3. Find the index with the highest probability (The winner)
        let maxProb = -1;
        let predictionIndex = -1;

        for (let i = 0; i < output.length; i++) {
            if (output[i] > maxProb) {
                maxProb = output[i];
                predictionIndex = i;
            }
        }

        return {
            word: indexToWord[predictionIndex],
            confidence: (maxProb * 100).toFixed(2) + '%'
        };
    }

    // 4. Test Cases based on Brian Tracy's "Goals" themes
    const tests = [
        ["how", "to", "your"],           // Likely: "achieve", "set", or "reach"
        ["write", "down", "your"],      // Likely: "goals"
        ["you", "must", "a"],           // Likely: "have" or "become"
        ["the", "most", "is"]           // General context
    ];

    console.log('\n=== Brian Tracy Model Inference Test ===\n');
    tests.forEach(test => {
        const result = predictMissingWord(test);
        console.log(`Context: [${test.join(', ')}]`);
        if (typeof result === 'string') {
            console.log(`  ${result}`);
        } else {
            console.log(`  Predicted: "${result.word}" (${result.confidence} confidence)`);
        }
        console.log('------------------------------------------');
    });
}

testModel().catch(err => {
    console.error("Failed to run test script:");
    console.error(err);
});