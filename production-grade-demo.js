import { OptimizedTextPreprocessor } from './optimized-text-preprocessor.js';

/**
 * Production-Grade Word2Vec Demo
 * 
 * This script demonstrates the complete production-grade Word2Vec implementation
 * with all advanced optimizations: subsampling, negative sampling, and memory efficiency.
 */

function demonstrateProductionGradeOptimizations() {
  console.log('=== PRODUCTION-GRADE WORD2VEC DEMO ===\n');
  console.log('Demonstrating enterprise-level Word2Vec optimizations:\n');
  console.log('✓ Subsampling of Frequent Words (Mikolov Formula)');
  console.log('✓ Negative Sampling with Unigram Table (0.75 Power)');
  console.log('✓ Sparse Training Data (Memory Optimization)');
  console.log('✓ Efficient Embedding Lookups');
  console.log('✓ O(1) Negative Sample Selection\n');

  // Create a larger sample text to demonstrate the optimizations
  const sampleText = `
    The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree.
    The fox was very quick and brown. The tree was big and green. The dog woke up and barked.
    The fox ran away quickly. The dog chased the fox but could not catch it. The fox was too fast.
    The dog went back to sleep under the tree. The tree provided good shade. The sun was hot.
    The fox found another place to rest. The place was cool and quiet. The fox felt safe there.
