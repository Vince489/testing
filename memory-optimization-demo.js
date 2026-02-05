import { OptimizedTextPreprocessor } from './optimized-text-preprocessor.js';
import fs from 'fs';

/**
 * Memory Optimization Demo
 * 
 * This script demonstrates the memory savings achieved by using sparse indices
 * instead of dense one-hot vectors for CBOW training.
 */

function demonstrateMemoryOptimization() {
  console.log('=== MEMORY OPTIMIZATION DEMO ===\n');
  
  // Create a sample text with repeated words to simulate real text
  const sampleText = `
    The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree.
    The fox was very quick and brown. The tree was big and green. The dog woke up and barked.
    The fox ran away quickly. The dog chased the fox but could not catch it. The fox was too fast.
    The dog went back to sleep under the tree. The tree provided good shade. The sun was hot.
    The fox found another place to rest. The place was cool and quiet. The fox felt safe there.
    The dog continued sleeping. The fox watched from afar. The fox was clever and smart.
    The dog was lazy but friendly. The fox and dog lived in harmony. The forest was peaceful.
    The birds were singing. The wind was blowing. The leaves were rustling. The day was beautiful.
    The fox enjoyed the weather. The dog enjoyed the nap. The fox was active. The dog was resting.
    The fox found food. The dog found peace. The fox was hunting. The dog was dreaming.
    The fox was wild. The dog was tame. The fox was free. The dog was loved. The fox was independent.
    The dog was dependent. The fox was strong. The dog was weak. The fox was fast. The dog was slow.
    The fox was agile. The dog was clumsy. The fox was smart. The dog was simple. The fox was complex.
    The dog was basic. The fox was advanced. The dog was beginner. The fox was expert. The dog was novice.
    The fox was master. The dog was student. The fox was teacher. The dog was learner. The fox was wise.
    The dog was foolish. The fox was clever. The dog was dumb. The fox was intelligent. The dog was stupid.
    The fox was bright. The dog was dim. The fox was sharp. The dog was dull. The fox was keen.
    The dog was blunt. The fox was acute. The dog was obtuse. The fox was precise. The dog was vague.
    The fox was clear. The dog was cloudy. The fox was bright. The dog was dark. The fox was light.
    The dog was heavy. The fox was fast. The dog was slow. The fox was quick. The dog was sluggish.
    The fox was rapid. The dog was gradual. The fox was sudden. The dog was steady. The fox was swift.
    The dog was sluggish. The fox was speedy. The dog was lethargic. The fox was energetic. The dog was tired.
    The fox was lively. The dog was sleepy. The fox was awake. The dog was drowsy. The fox was alert.
    The dog was lazy. The fox was active. The dog was passive. The fox was dynamic. The dog was static.
    The fox was moving. The dog was still. The fox was running. The dog was walking. The fox was jumping.
    The dog was crawling. The fox was leaping. The dog was creeping. The fox was bounding. The dog was strolling.
    The fox was sprinting. The dog was ambling. The fox was racing. The dog was sauntering. The fox was dashing.
    The dog was meandering. The fox was charging. The dog was wandering. The fox was pursuing. The dog was avoiding.
    The fox was chasing. The dog was fleeing. The fox was hunting. The dog was hiding. The fox was seeking.
    The dog was concealing. The fox was finding. The dog was losing. The fox was winning. The dog was failing.
    The fox was succeeding. The dog was losing. The fox was gaining. The dog was falling. The fox was rising.
    The dog was dropping. The fox was climbing. The dog was descending. The fox was ascending. The dog was sinking.
    The fox was floating. The dog was drowning. The fox was swimming. The dog was splashing. The fox was gliding.
    The dog was floundering. The fox was soaring. The dog was struggling. The fox was thriving. The dog was suffering.
    The fox was flourishing. The dog was withering. The fox was growing. The dog was shrinking. The fox was expanding.
    The dog was contracting. The fox was developing. The dog was deteriorating. The fox was improving. The dog was worsening.
    The fox was progressing. The dog was regressing. The fox was advancing. The dog was retreating. The fox was moving forward.
    The dog was moving backward. The fox was going ahead. The dog was falling behind. The fox was leading. The dog was following.
    The fox was guiding. The dog was trailing. The fox was directing. The dog was wandering. The fox was navigating. The dog was lost.
  `;

  // Write sample text to a temporary file
  const tempFilePath = './sample_text.txt';
  fs.writeFileSync(tempFilePath, sampleText);

  try {
    // Initialize the optimized preprocessor
    const preprocessor = new OptimizedTextPreprocessor();
    
    // Step 1: Clean text
    console.log('1. Cleaning text...');
    preprocessor.cleanText(tempFilePath);
    
    // Step 2: Build vocabulary
    console.log('\n2. Building vocabulary...');
    preprocessor.buildVocab(preprocessor.tokens, 2);
    
    // Step 3: Initialize embedding layer
    console.log('\n3. Initializing embedding layer...');
    preprocessor.initEmbeddingLayer(100); // 100-dimensional embeddings
    
    // Step 4: Generate training pairs
    console.log('\n4. Generating training pairs...');
    const trainingPairs = preprocessor.generateTrainingPairs(preprocessor.tokens, 2);
    
    // Step 5: Prepare sparse training data (MEMORY OPTIMIZED)
    console.log('\n5. Preparing sparse training data...');
    const sparseData = preprocessor.prepareSparseTrainingData(trainingPairs);
    
    // Step 6: Demonstrate memory savings
    console.log('\n6. Memory Usage Analysis:');
    console.log('========================');
    
    // Calculate actual memory usage
    const vocabSize = preprocessor.vocabSize;
    const numPairs = trainingPairs.length;
    const avgContextSize = trainingPairs.reduce((sum, pair) => sum + pair.context.length, 0) / numPairs;
    
    // Dense representation memory (current implementation)
    const denseMemoryBytes = numPairs * vocabSize * 8; // 8 bytes per float64
    const denseMemoryMB = denseMemoryBytes / (1024 * 1024);
    
    // Sparse representation memory (optimized implementation)
    const sparseMemoryBytes = trainingPairs.reduce((total, pair) => {
      return total + pair.context.length + 1; // context indices + target index
    }, 0) * 8; // 8 bytes per float64
    const sparseMemoryMB = sparseMemoryBytes / (1024 * 1024);
    
    // Embedding layer memory
    const embeddingMemoryBytes = vocabSize * 100 * 8; // vocabSize * embeddingDim * 8 bytes
    const embeddingMemoryMB = embeddingMemoryBytes / (1024 * 1024);
    
    console.log(`Vocabulary size: ${vocabSize}`);
    console.log(`Number of training pairs: ${numPairs}`);
    console.log(`Average context size: ${avgContextSize.toFixed(2)}`);
    console.log();
    console.log('Memory Usage Comparison:');
    console.log(`Dense one-hot vectors: ${denseMemoryMB.toFixed(4)} MB`);
    console.log(`Sparse indices: ${sparseMemoryMB.toFixed(4)} MB`);
    console.log(`Embedding layer: ${embeddingMemoryMB.toFixed(4)} MB`);
    console.log(`Total optimized: ${(sparseMemoryMB + embeddingMemoryMB).toFixed(4)} MB`);
    console.log();
    console.log('Memory Savings:');
    console.log(`Raw memory savings: ${((1 - sparseMemoryMB / denseMemoryMB) * 100).toFixed(2)}%`);
    console.log(`Total system memory: ${((1 - (sparseMemoryMB + embeddingMemoryMB) / denseMemoryMB) * 100).toFixed(2)}%`);
    
    // Step 7: Demonstrate on-demand conversion
    console.log('\n7. On-Demand Conversion Demo:');
    console.log('==============================');
    
    // Convert only a small batch for demonstration
    const demoBatch = sparseData.slice(0, 5);
    console.log(`Converting ${demoBatch.length} training examples to dense format...`);
    
    const startTime = Date.now();
    const denseBatch = preprocessor.convertToDenseFormat(demoBatch);
    const conversionTime = Date.now() - startTime;
    
    console.log(`Conversion completed in ${conversionTime}ms`);
    console.log(`Each example: input size ${denseBatch[0].input.length}, output size ${denseBatch[0].output.length}`);
    
    // Step 8: Show scalability
    console.log('\n8. Scalability Analysis:');
    console.log('========================');
    
    // Simulate larger vocabulary and dataset
    const scenarios = [
      { vocabSize: 10000, numPairs: 100000 },
      { vocabSize: 50000, numPairs: 500000 },
      { vocabSize: 100000, numPairs: 1000000 }
    ];
    
    scenarios.forEach((scenario, index) => {
      const denseMem = (scenario.numPairs * scenario.vocabSize * 8) / (1024 * 1024 * 1024); // GB
      const sparseMem = (scenario.numPairs * 5 * 8) / (1024 * 1024 * 1024); // GB (assuming avg context size of 5)
      const embeddingMem = (scenario.vocabSize * 100 * 8) / (1024 * 1024 * 1024); // GB
      
      console.log(`Scenario ${index + 1} (Vocab: ${scenario.vocabSize}, Pairs: ${scenario.numPairs}):`);
      console.log(`  Dense: ${denseMem.toFixed(2)} GB`);
      console.log(`  Sparse + Embeddings: ${(sparseMem + embeddingMem).toFixed(2)} GB`);
      console.log(`  Savings: ${((1 - (sparseMem + embeddingMem) / denseMem) * 100).toFixed(1)}%`);
    });
    
    console.log('\n=== DEMO COMPLETE ===');
    console.log('\nKey Benefits of the Optimized Approach:');
    console.log('• Memory usage scales with actual data, not vocabulary size');
    console.log('• On-demand conversion reduces memory pressure during training');
    console.log('• Proper CBOW averaging instead of multi-hot vectors');
    console.log('• Embedding layer provides efficient lookups');
    console.log('• Can handle much larger vocabularies and datasets');
    
  } finally {
    // Clean up temporary file
    if (fs.existsSync(tempFilePath)) {
      fs.unlinkSync(tempFilePath);
    }
  }
}

// Run the demo
demonstrateMemoryOptimization();