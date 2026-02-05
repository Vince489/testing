import { OptimizedTextPreprocessor } from './optimized-text-preprocessor.js';
import TextPreprocessor from './text-preprocessor.js';
import EnhancedTextPreprocessor from './enhanced-text-preprocessor.js';

/**
 * Performance Validation Script
 * 
 * This script validates the performance improvements made to the text preprocessor
 * by comparing the original implementation with the optimized versions.
 */

class PerformanceValidator {
  constructor() {
    this.results = [];
  }

  /**
   * Measure execution time of a function
   * @param {Function} fn - Function to measure
   * @param {string} name - Name of the test
   * @returns {Object} Execution result with timing
   */
  async measurePerformance(fn, name) {
    const startTime = process.hrtime.bigint();
    const startMemory = process.memoryUsage();
    
    try {
      const result = await fn();
      const endTime = process.hrtime.bigint();
      const endMemory = process.memoryUsage();
      
      const executionTime = Number(endTime - startTime) / 1000000; // Convert to milliseconds
      const memoryDelta = endMemory.heapUsed - startMemory.heapUsed;
      
      return {
        name,
        success: true,
        executionTime,
        memoryDelta,
        result
      };
    } catch (error) {
      const endTime = process.hrtime.bigint();
      const executionTime = Number(endTime - startTime) / 1000000;
      
      return {
        name,
        success: false,
        executionTime,
        error: error.message
      };
    }
  }

  /**
   * Create test data of varying sizes
   * @param {number} size - Size multiplier for test data
   * @returns {string} Test text
   */
  createTestData(size = 1) {
    const baseText = `
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

    return baseText.repeat(size);
  }

  /**
   * Test original implementation
   * @param {string} testData - Test data
   * @returns {Object} Test results
   */
  async testOriginalImplementation(testData) {
    return this.measurePerformance(async () => {
      const preprocessor = new TextPreprocessor();
      
      // Clean text
      preprocessor.cleanText('./test_data.txt');
      
      // Build vocabulary
      preprocessor.buildVocab(preprocessor.tokens, 2);
      
      // Generate training pairs
      const trainingPairs = preprocessor.generateTrainingPairs(preprocessor.tokens, 2);
      
      // Prepare dense training data (MEMORY INTENSIVE)
      const trainingData = preprocessor.prepareTrainingData(trainingPairs);
      
      return {
        vocabSize: preprocessor.vocabSize,
        numPairs: trainingPairs.length,
        dataSize: trainingData.length,
        avgInputSize: trainingData[0]?.input?.length || 0,
        avgOutputSize: trainingData[0]?.output?.length || 0
      };
    }, 'Original Implementation (Dense)');
  }

  /**
   * Test optimized implementation
   * @param {string} testData - Test data
   * @returns {Object} Test results
   */
  async testOptimizedImplementation(testData) {
    return this.measurePerformance(async () => {
      const preprocessor = new OptimizedTextPreprocessor();
      
      // Clean text
      preprocessor.cleanText('./test_data.txt');
      
      // Build vocabulary
      preprocessor.buildVocab(preprocessor.tokens, 2);
      
      // Initialize embedding layer
      preprocessor.initEmbeddingLayer(100);
      
      // Generate training pairs
      const trainingPairs = preprocessor.generateTrainingPairs(preprocessor.tokens, 2);
      
      // Prepare sparse training data (MEMORY OPTIMIZED)
      const sparseData = preprocessor.prepareSparseTrainingData(trainingPairs);
      
      return {
        vocabSize: preprocessor.vocabSize,
        numPairs: trainingPairs.length,
        dataSize: sparseData.length,
        embeddingDim: 100,
        avgContextSize: trainingPairs.reduce((sum, pair) => sum + pair.context.length, 0) / trainingPairs.length
      };
    }, 'Optimized Implementation (Sparse)');
  }

  /**
   * Test enhanced implementation
   * @param {string} testData - Test data
   * @returns {Object} Test results
   */
  async testEnhancedImplementation(testData) {
    return this.measurePerformance(async () => {
      const preprocessor = new EnhancedTextPreprocessor({
        progressCallback: (progress) => {
          // Silent progress for testing
        }
      });
      
      // Clean text
      preprocessor.cleanText('./test_data.txt');
      
      // Build vocabulary
      preprocessor.buildVocab(preprocessor.tokens, 2);
      
      // Generate training pairs
      const trainingPairs = preprocessor.generateTrainingPairs(preprocessor.tokens, 2);
      
      // Prepare sparse training data
      const sparseData = preprocessor.prepareSparseTrainingData(trainingPairs);
      
      return {
        vocabSize: preprocessor.vocabSize,
        numPairs: trainingPairs.length,
        dataSize: sparseData.length,
        memoryStats: preprocessor.getMemoryStats(),
        cleaningOptions: preprocessor.cleaningOptions
      };
    }, 'Enhanced Implementation (Sparse + Monitoring)');
  }

  /**
   * Run comprehensive performance tests
   */
  async runPerformanceTests() {
    console.log('=== PERFORMANCE VALIDATION TESTS ===\n');
    
    const testSizes = [1, 5, 10]; // Small, Medium, Large
    
    for (const size of testSizes) {
      console.log(`\n--- Testing with data size: ${size}x ---`);
      
      const testData = this.createTestData(size);
      
      // Write test data to file
      const fs = require('fs');
      fs.writeFileSync('./test_data.txt', testData);
      
      try {
        // Test original implementation
        console.log('\n1. Testing Original Implementation...');
        const originalResult = await this.testOriginalImplementation(testData);
        this.results.push(originalResult);
        
        if (originalResult.success) {
          console.log(`   ✓ Completed in ${originalResult.executionTime.toFixed(2)}ms`);
          console.log(`   ✓ Memory delta: ${(originalResult.memoryDelta / (1024 * 1024)).toFixed(2)} MB`);
          console.log(`   ✓ Vocab size: ${originalResult.result.vocabSize}`);
          console.log(`   ✓ Training pairs: ${originalResult.result.numPairs}`);
        } else {
          console.log(`   ✗ Failed: ${originalResult.error}`);
        }
        
        // Test optimized implementation
        console.log('\n2. Testing Optimized Implementation...');
        const optimizedResult = await this.testOptimizedImplementation(testData);
        this.results.push(optimizedResult);
        
        if (optimizedResult.success) {
          console.log(`   ✓ Completed in ${optimizedResult.executionTime.toFixed(2)}ms`);
          console.log(`   ✓ Memory delta: ${(optimizedResult.memoryDelta / (1024 * 1024)).toFixed(2)} MB`);
          console.log(`   ✓ Vocab size: ${optimizedResult.result.vocabSize}`);
          console.log(`   ✓ Training pairs: ${optimizedResult.result.numPairs}`);
          console.log(`   ✓ Embedding dimension: ${optimizedResult.result.embeddingDim}`);
        } else {
          console.log(`   ✗ Failed: ${optimizedResult.error}`);
        }
        
        // Test enhanced implementation
        console.log('\n3. Testing Enhanced Implementation...');
        const enhancedResult = await this.testEnhancedImplementation(testData);
        this.results.push(enhancedResult);
        
        if (enhancedResult.success) {
          console.log(`   ✓ Completed in ${enhancedResult.executionTime.toFixed(2)}ms`);
          console.log(`   ✓ Memory delta: ${(enhancedResult.memoryDelta / (1024 * 1024)).toFixed(2)} MB`);
          console.log(`   ✓ Vocab size: ${enhancedResult.result.vocabSize}`);
          console.log(`   ✓ Training pairs: ${enhancedResult.result.numPairs}`);
          console.log(`   ✓ Peak memory: ${enhancedResult.result.memoryStats.peakMemoryMB} MB`);
        } else {
          console.log(`   ✗ Failed: ${enhancedResult.error}`);
        }
        
      } finally {
        // Clean up test file
        if (fs.existsSync('./test_data.txt')) {
          fs.unlinkSync('./test_data.txt');
        }
      }
    }
    
    this.displayResults();
  }

  /**
   * Display performance test results
   */
  displayResults() {
    console.log('\n\n=== PERFORMANCE TEST RESULTS ===');
    console.log('==================================\n');
    
    // Group results by test size and implementation
    const groupedResults = {};
    
    this.results.forEach(result => {
      const match = result.name.match(/(\d+)x/);
      const size = match ? match[1] : 'unknown';
      const impl = result.name.replace(/ \(\d+x\)/, '');
      
      if (!groupedResults[size]) {
        groupedResults[size] = {};
      }
      groupedResults[size][impl] = result;
    });
    
    // Display comparison for each test size
    Object.keys(groupedResults).forEach(size => {
      console.log(`\n--- Data Size: ${size}x ---`);
      const results = groupedResults[size];
      
      console.log('\nExecution Time Comparison:');
      Object.keys(results).forEach(impl => {
        const result = results[impl];
        if (result.success) {
          console.log(`  ${impl}: ${result.executionTime.toFixed(2)}ms`);
        } else {
          console.log(`  ${impl}: FAILED - ${result.error}`);
        }
      });
      
      console.log('\nMemory Usage Comparison:');
      Object.keys(results).forEach(impl => {
        const result = results[impl];
        if (result.success) {
          console.log(`  ${impl}: ${(result.memoryDelta / (1024 * 1024)).toFixed(2)} MB`);
        }
      });
      
      // Calculate improvements
      if (results['Optimized Implementation (Sparse)'] && results['Original Implementation (Dense)']) {
        const original = results['Original Implementation (Dense)'];
        const optimized = results['Optimized Implementation (Sparse)'];
        
        if (original.success && optimized.success) {
          const timeImprovement = ((original.executionTime - optimized.executionTime) / original.executionTime) * 100;
          const memoryImprovement = ((original.memoryDelta - optimized.memoryDelta) / original.memoryDelta) * 100;
          
          console.log('\nImprovements:');
          console.log(`  Time: ${timeImprovement.toFixed(1)}% faster`);
          console.log(`  Memory: ${memoryImprovement.toFixed(1)}% reduction`);
        }
      }
    });
    
    // Summary
    console.log('\n\n=== SUMMARY ===');
    console.log('===============');
    
    const successfulTests = this.results.filter(r => r.success);
    const totalTests = this.results.length;
    const successRate = (successfulTests.length / totalTests) * 100;
    
    console.log(`Overall Success Rate: ${successRate.toFixed(1)}% (${successfulTests.length}/${totalTests})`);
    
    if (successfulTests.length > 0) {
      const avgTime = successfulTests.reduce((sum, r) => sum + r.executionTime, 0) / successfulTests.length;
      const avgMemory = successfulTests.reduce((sum, r) => sum + r.memoryDelta, 0) / successfulTests.length;
      
      console.log(`Average Execution Time: ${avgTime.toFixed(2)}ms`);
      console.log(`Average Memory Usage: ${(avgMemory / (1024 * 1024)).toFixed(2)} MB`);
    }
    
    console.log('\n=== VALIDATION COMPLETE ===');
  }
}

// Run the performance validation
const validator = new PerformanceValidator();
validator.runPerformanceTests().catch(console.error);