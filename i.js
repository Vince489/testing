import fs from 'fs';

/**
 * Goals-Brian-Tracy.txt Cleaning Script
 * 
 * This script uses the TextPreprocessor class to clean and preprocess the Goals-Brian-Tracy.txt file,
 * then saves the results to multiple output files for further analysis and training.
 */

import TextPreprocessor from './text-preprocessor.js';

async function cleanGoalsText() {
  console.log('=== Goals-Brian-Tracy.txt Cleaning Script ===\n');

  try {
    // 1. Initialize the preprocessor
    console.log('Step 1: Initializing TextPreprocessor...');
    const preprocessor = new TextPreprocessor();

    // 2. Load and clean the text
    console.log('Step 2: Loading and cleaning Goals-Brian-Tracy.txt...');
    const tokens = preprocessor.cleanText('Goals-Brian-Tracy.txt');
    
    console.log(`✓ Successfully loaded ${tokens.length} tokens`);

    // 3. Build vocabulary with configurable frequency threshold
    console.log('\nStep 3: Building vocabulary...');
    const minCount = 10; // Keep words that appear at least 10 times
    const vocabInfo = preprocessor.buildVocab(tokens, minCount);
    
    console.log(`✓ Vocabulary built with ${vocabInfo.vocabSize} words (min count: ${minCount})`);

    // 4. Generate sequence-based training pairs (Language Modeling approach)
    console.log('\nStep 4: Generating sequence-based training pairs (Language Modeling)...');
    const windowSize = 3; // Use window size of 3
    const trainingPairs = preprocessor.generateTrainingPairs(tokens, windowSize);

    console.log(`✓ Generated ${trainingPairs.length} sequence-based training pairs (window size: ${windowSize})`);

    // 5. REMOVED: Prepare training data for neural networks
    // Reason: Memory efficiency. One-hot vectors generated on-demand in NeuralNetwork.

    // 6. Extract vocabulary statistics
    console.log('\nStep 6: Extracting vocabulary statistics...');
    const stats = preprocessor.getVocabStats();
    
    console.log('✓ Vocabulary statistics calculated');

    // 7. Save all outputs to files
    console.log('\nStep 7: Saving outputs to files...');

    // Save cleaned tokens
    
    // 7a. Save cleaned tokens
    fs.writeFileSync('goals_cleaned_tokens.json', JSON.stringify(tokens, null, 2));
    console.log('✓ Saved cleaned tokens to goals_cleaned_tokens.json');

    // 7b. Save vocabulary information
    const vocabData = {
      vocabSize: vocabInfo.vocabSize,
      wordToIndex: vocabInfo.wordToIndex,
      indexToWord: vocabInfo.indexToWord,
      vocab: vocabInfo.vocab,
      minCount: minCount,
      totalUniqueWords: Object.keys(stats).includes('uniqueTokens') ? stats.uniqueTokens : 'N/A'
    };
    
    fs.writeFileSync('goals_vocabulary.json', JSON.stringify(vocabData, null, 2));
    console.log('✓ Saved vocabulary to goals_vocabulary.json');

    // 7c. Save training pairs
    fs.writeFileSync('goals_training_pairs.json', JSON.stringify(trainingPairs, null, 2));
    console.log('✓ Saved training pairs to goals_training_pairs.json');

    // 7d. REMOVED: Saving goals_training_data.json
    // Reason: Memory efficiency. One-hot vectors generated on-demand in NeuralNetwork.

    // 7e. Save vocabulary statistics as text
    const statsText = `
=== GOALS-BRIAN-TRACY.TXT PREPROCESSING RESULTS ===

File: Goals-Brian-Tracy.txt
Generated: ${new Date().toISOString()}

=== STATISTICS ===
Total tokens: ${stats.totalTokens}
Unique tokens: ${stats.uniqueTokens}
Vocabulary size (min count >= ${minCount}): ${stats.vocabSize}
Tokens in vocabulary: ${stats.tokensInVocab}
Vocabulary coverage: ${stats.coverage}
Minimum count threshold: ${minCount}
Window size: ${windowSize}

=== VOCABULARY INFORMATION ===
Vocabulary contains the ${vocabInfo.vocab.length} most frequent words that appear at least ${minCount} times.

Top 30 most frequent words:
${vocabInfo.vocab.slice(0, 30).map((word, index) => `  ${index + 1}. "${word}"`).join('\n')}

=== FILE OUTPUTS ===
- goals_cleaned_tokens.json: Array of all cleaned tokens
- goals_vocabulary.json: Vocabulary mappings and word frequencies
- goals_training_pairs.json: Sequence-based training pairs [context, target]
- goals_vocabulary_stats.txt: This statistics file
- goals_preprocessing_log.txt: Detailed processing log
`;

    fs.writeFileSync('goals_vocabulary_stats.txt', statsText);
    console.log('✓ Saved vocabulary statistics to goals_vocabulary_stats.txt');

    // 7f. Save detailed processing log
    const logText = `
=== GOALS-BRIAN-TRACY.TXT PREPROCESSING LOG ===

Timestamp: ${new Date().toISOString()}

=== PROCESSING STEPS ===

1. TEXT LOADING AND CLEANING:
   - Loaded Goals-Brian-Tracy.txt
   - Converted to lowercase
   - Removed punctuation, numbers, and special characters
   - Split by whitespace and filtered empty strings
   - Total tokens extracted: ${tokens.length}

2. VOCABULARY BUILDING:
   - Counted word frequencies across all tokens
   - Filtered words appearing at least ${minCount} times
   - Sorted vocabulary by frequency (descending)
   - Vocabulary size: ${vocabInfo.vocabSize}
   - Coverage: ${stats.coverage} of total tokens

3. TRAINING PAIR GENERATION:
   - Used Language Modeling (Forward-Only) approach
   - Window size: ${windowSize} (previous words only)
   - Generated ${trainingPairs.length} sequence-based training pairs
   - Each pair contains: { context: [indices], target: index }

=== VOCABULARY DETAILS ===

Word-to-Index Mappings:
${Object.entries(vocabInfo.wordToIndex).slice(0, 20).map(([word, index]) => `  "${word}" -> ${index}`).join('\n')}

Index-to-Word Mappings:
${Object.entries(vocabInfo.indexToWord).slice(0, 20).map(([index, word]) => `  ${index} -> "${word}"`).join('\n')}

=== EXAMPLE TRAINING PAIRS ===

${trainingPairs.slice(0, 5).map((pair, index) => `
Pair ${index + 1}:
  Context indices: [${pair.context.join(', ')}]
  Context words: [${pair.context.map(idx => vocabInfo.indexToWord[idx]).join(', ')}]
  Target index: ${pair.target}
  Target word: "${vocabInfo.indexToWord[pair.target]}"
`).join('')}

=== USAGE NOTES ===

The generated files can be used for:

1. Word2Vec training with the NeuralNetwork class
2. Vocabulary analysis and word frequency studies
3. Natural language processing tasks
4. Text analysis and pattern recognition

For neural network training, use:
- Input layer size: ${vocabInfo.vocabSize}
- Hidden layer size: 50-300 (embedding dimension)
- Output layer size: ${vocabInfo.vocabSize}
`;

    fs.writeFileSync('goals_preprocessing_log.txt', logText);
    console.log('✓ Saved detailed processing log to goals_preprocessing_log.txt');

    // 8. Display summary
    console.log('\n=== CLEANING COMPLETED SUCCESSFULLY ===');
    console.log('\nSUMMARY:');
    console.log(`- Original file: Goals-Brian-Tracy.txt`);
    console.log(`- Total tokens: ${tokens.length}`);
    console.log(`- Vocabulary size: ${vocabInfo.vocabSize} words`);
    console.log(`- Training pairs: ${trainingPairs.length}`);
    console.log(`- Vocabulary coverage: ${stats.coverage}`);
    
    console.log('\nOUTPUT FILES CREATED:');
    console.log('- goals_cleaned_tokens.json');
    console.log('- goals_vocabulary.json');
    console.log('- goals_training_pairs.json');
    console.log('- goals_vocabulary_stats.txt');
    console.log('- goals_preprocessing_log.txt');
    
    console.log('\n🎉 Goals-Brian-Tracy.txt has been successfully cleaned and preprocessed!');
    console.log('\nNext steps:');
    console.log('1. Review the vocabulary statistics in goals_vocabulary_stats.txt');
    console.log('2. Use the training pairs for Word2Vec model training');
    console.log('3. Analyze word frequencies and patterns');
    console.log('4. Train neural networks for word embeddings');

  } catch (error) {
    console.error('❌ Error during preprocessing:', error.message);
    console.error('Stack trace:', error.stack);
  }
}

// Run the cleaning script
if (process.argv[1] && process.argv[1].includes('i.js')) {
  cleanGoalsText();
}

export { cleanGoalsText };