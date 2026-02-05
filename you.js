import fs from 'fs';
import TextPreprocessor from './text-preprocessor.js';

/**
 * Combined Cleaning Script: Goals & Eat That Frog
 * Prepares a merged dataset for a unified Brian Tracy "Success Model"
 */
async function cleanMergedText() {
  console.log('=== Brian Tracy Merged Text Cleaning Script ===\n');

  try {
    // 1. Initialize the preprocessor
    console.log('Step 1: Initializing TextPreprocessor...');
    const preprocessor = new TextPreprocessor();

    // 2. Load and clean both books
    console.log('Step 2: Loading and cleaning text files...');
    
    // Check if files exist before trying to read
    const files = ['Goals-Brian-Tracy.txt', 'Eat-That-Frog.txt'];
    let allTokens = [];

    for (const file of files) {
      if (fs.existsSync(file)) {
        console.log(`- Processing ${file}...`);
        const fileTokens = preprocessor.cleanText(file);
        allTokens = [...allTokens, ...fileTokens];
      } else {
        console.warn(`! Warning: ${file} not found. Skipping.`);
      }
    }

    if (allTokens.length === 0) {
      throw new Error("No tokens were extracted. Check your file paths.");
    }
    
    // Set the internal tokens to the merged list
    preprocessor.tokens = allTokens;
    console.log(`✓ Successfully merged ${allTokens.length} total tokens`);

    // 3. Build vocabulary
    console.log('\nStep 3: Building vocabulary...');
    const minCount = 5; 
    const vocabInfo = preprocessor.buildVocab(allTokens, minCount);
    console.log(`✓ Vocabulary built with ${vocabInfo.vocabSize} words (min count: ${minCount})`);

    // 4. Generate training pairs
    // We use the corrected logic: Center target with context around it
    console.log('\nStep 4: Generating training pairs...');
    const windowSize = 3; 
    const trainingPairs = preprocessor.generateTrainingPairs(allTokens, windowSize);
    console.log(`✓ Generated ${trainingPairs.length} training pairs`);

    // 5. Calculate statistics
    const stats = preprocessor.getVocabStats();

    // 6. Save all outputs
    console.log('\nStep 6: Saving outputs to files...');

    // 6a. Merged Tokens
    fs.writeFileSync('tracy_merged_tokens.json', JSON.stringify(allTokens, null, 2));

    // 6b. Merged Vocabulary
    const vocabData = {
      vocabSize: vocabInfo.vocabSize,
      wordToIndex: vocabInfo.wordToIndex,
      indexToWord: vocabInfo.indexToWord,
      vocab: vocabInfo.vocab,
      minCount: minCount
    };
    fs.writeFileSync('tracy_merged_vocabulary.json', JSON.stringify(vocabData, null, 2));

    // 6c. Merged Training Pairs
    fs.writeFileSync('tracy_merged_training_pairs.json', JSON.stringify(trainingPairs, null, 2));

    // 6d. Stats File
    const statsText = `
=== MERGED PREPROCESSING RESULTS ===
Files: Goals-Brian-Tracy.txt, Eat-That-Frog.txt
Generated: ${new Date().toISOString()}

=== STATISTICS ===
Total tokens: ${stats.totalTokens}
Unique tokens: ${stats.uniqueTokens}
Vocabulary size (minCount >= ${minCount}): ${stats.vocabSize}
Tokens in vocabulary: ${stats.tokensInVocab}
Vocabulary coverage: ${stats.coverage}

=== TOP 30 MOST FREQUENT WORDS (MERGED) ===
${vocabInfo.vocab.slice(0, 30).map((word, index) => `  ${index + 1}. "${word}"`).join('\n')}
`;
    fs.writeFileSync('tracy_merged_stats.txt', statsText);

    console.log('\n=== MERGED CLEANING COMPLETED SUCCESSFULLY ===');
    console.log(`- Output: tracy_merged_training_pairs.json`);
    console.log(`- Use these pairs to train your Neural Network for the combined model.`);

  } catch (error) {
    console.error('❌ Error during preprocessing:', error.message);
  }
}

cleanMergedText();