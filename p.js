import fs from 'fs';
import { NeuralNetwork } from './neural-network-3.js';

async function trainGoalsModel() {
    console.log('--- Loading Preprocessed Data ---');
    // We load the small integer indices, which take very little RAM
    const vocabData = JSON.parse(fs.readFileSync('tracy_merged_vocabulary.json', 'utf8'));
    const trainingPairs = JSON.parse(fs.readFileSync('tracy_merged_training_pairs.json', 'utf8'));
    
    const vocabSize = vocabData.vocabSize;
    const embeddingDim = 100; // Increased for better resolution with merged data
    const batchSize = 128;    // Memory efficient batch size

    const nnConfig = {
        layers: [vocabSize, embeddingDim, vocabSize],
        learningRate: 0.05,
        activation: 'tanh',
        outputActivation: 'sigmoid',
        momentum: 0.9,
        batchSize: batchSize,
        verbose: false // We will handle our own logging
    };

    const nn = new NeuralNetwork(nnConfig);

    console.log(`Vocab Size: ${vocabSize}`);
    console.log(`Training Pairs: ${trainingPairs.length}`);
    console.log('--- Starting On-The-Fly Training ---');

    const epochs = 5;

    for (let epoch = 1; epoch <= epochs; epoch++) {
        let epochLoss = 0;
        
        // Shuffle the small pair indices only (low memory cost)
        const shuffled = trainingPairs.sort(() => Math.random() - 0.5);

        for (let i = 0; i < shuffled.length; i += batchSize) {
            // 1. Get a small slice of indices
            const chunk = shuffled.slice(i, i + batchSize);

            // 2. GENERATE VECTORS ON-THE-FLY (Only 128 exist at a time)
            const miniBatch = chunk.map(pair => {
                const input = new Array(vocabSize).fill(0);
                pair.context.forEach(idx => input[idx] = 1 / pair.context.length);
                
                const output = new Array(vocabSize).fill(0);
                output[pair.target] = 1;
                
                return { input, output };
            });

            // 3. Train on this batch and immediately release memory
            // We use the internal _trainBatch method from your class
            epochLoss += nn._trainBatch(miniBatch, nn.config.learningRate);

            if (i % 5120 === 0) {
                process.stdout.write('.');
            }
        }

        const avgLoss = epochLoss / (shuffled.length / batchSize);
        console.log(`\nEpoch ${epoch} Complete. Average Loss: ${avgLoss.toFixed(6)}`);
    }

    nn.save('goals_embedding_network.json');
    console.log('\n🎉 Training complete. Model saved without RAM overhead!');
}

trainGoalsModel().catch(err => console.error(err));