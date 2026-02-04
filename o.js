import fs from 'fs';
import { NeuralNetwork } from './neural-network-3.js';

async function trainGoalsModel() {
    console.log('--- Loading Preprocessed Data ---');
    const vocabData = JSON.parse(fs.readFileSync('goals_vocabulary.json', 'utf8'));
    const trainingPairs = JSON.parse(fs.readFileSync('goals_training_pairs.json', 'utf8'));
    
    const vocabSize = vocabData.vocabSize;
    const embeddingDim = 30;

    console.log(`Vocab Size: ${vocabSize}`);
    console.log(`Training Pairs: ${trainingPairs.length}`);

    console.log('--- Formatting Training Data (This may take a minute) ---');
    const totalPairs = trainingPairs.length;
    
    const formattedData = trainingPairs.map((pair, index) => {
        // Log progress every 5000 items so you know formatting isn't stuck
        if (index % 1000 === 0) {
            console.log(`  Formatting: ${((index / totalPairs) * 100).toFixed(1)}%...`);
        }

        const inputVector = new Array(vocabSize).fill(0);
        pair.context.forEach(idx => {
            inputVector[idx] = 1 / pair.context.length; 
        });

        const outputVector = new Array(vocabSize).fill(0);
        outputVector[pair.target] = 1;

        return { input: inputVector, output: outputVector };
    });
    console.log('✓ Formatting Complete.\n');

    const nnConfig = {
        layers: [vocabSize, embeddingDim, vocabSize],
        learningRate: 0.05,
        activation: 'tanh',
        outputActivation: 'sigmoid',
        momentum: 0.9,
        batchSize: 64,
        verbose: true
    };

    const nn = new NeuralNetwork(nnConfig);

    console.log('--- Starting Training ---');
    console.log('Legend: Each "." represents one batch processed.\n');

    nn.train(formattedData, 5, {
        earlyStopping: true,
        patience: 5,
        minDelta: 0.0001
    });

    nn.save('goals_embedding_network.json');
    console.log('\n🎉 Training complete. Model saved!');
}

trainGoalsModel().catch(err => console.error(err));