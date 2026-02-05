import { CBOWModel } from './cbow-model.js';
import fs from 'fs';

/**
 * CBOW Coverage Validator
 *
 * Validates vocabulary coverage and training quality for the CBOW model
 */
class CBOWCoverageValidator {
  /**
   * Initialize the validator
   * @param {Array} vocabulary - Array of vocabulary terms
   * @param {Object} chapterMap - Mapping of terms to chapters
   */
  constructor(vocabulary, chapterMap) {
    this.vocabulary = vocabulary;
    this.chapterMap = chapterMap;
    this.termIndexCache = {};
    vocabulary.forEach((term, index) => {
      this.termIndexCache[term] = index;
    });
  }

  /**
   * Comprehensive CBOW coverage analysis
   * @param {Array} trainingPairs - Array of training pairs
   * @returns {Object} Analysis results
   */
  analyzeCoverage(trainingPairs) {
    const results = {
      totalTerms: this.vocabulary.length,
      coveredTerms: new Set(),
      termStats: {},
      chapterCoverage: {},
      contextStats: {
        totalContexts: 0,
        avgContextSize: 0,
        contextSizeDistribution: {}
      },
      missingTerms: []
    };

    // First pass: collect basic coverage stats
    trainingPairs.forEach(pair => {
      results.coveredTerms.add(pair.target);
      pair.context.forEach(idx => results.coveredTerms.add(idx));
      results.contextStats.totalContexts += pair.context.length;
    });

    // Calculate average context size
    results.contextStats.avgContextSize =
      results.contextStats.totalContexts / trainingPairs.length;

    // Second pass: gather detailed statistics
    this.vocabulary.forEach((term, index) => {
      const isCovered = results.coveredTerms.has(index);
      const chapter = this.chapterMap[term] || 'unknown';

      results.termStats[term] = {
        index,
        covered: isCovered,
        chapter,
        asTarget: 0,
        asContext: 0,
        contextPositions: []
      };

      if (!results.chapterCoverage[chapter]) {
        results.chapterCoverage[chapter] = { total: 0, covered: 0 };
      }
      results.chapterCoverage[chapter].total++;

      if (isCovered) {
        results.chapterCoverage[chapter].covered++;
      } else {
        results.missingTerms.push({ term, index, chapter });
      }
    });

    // Third pass: count term appearances and context positions
    trainingPairs.forEach((pair, pairIndex) => {
      // Count target appearances
      const targetTerm = this.vocabulary[pair.target];
      if (results.termStats[targetTerm]) {
        results.termStats[targetTerm].asTarget++;
      }

      // Count context appearances and positions
      pair.context.forEach((contextIndex, position) => {
        const contextTerm = this.vocabulary[contextIndex];
        if (results.termStats[contextTerm]) {
          results.termStats[contextTerm].asContext++;
          results.termStats[contextTerm].contextPositions.push(position);

          // Track context size distribution
          const contextSize = pair.context.length;
          if (!results.contextStats.contextSizeDistribution[contextSize]) {
            results.contextStats.contextSizeDistribution[contextSize] = 0;
          }
          results.contextStats.contextSizeDistribution[contextSize]++;
        }
      });
    });

    // Calculate coverage percentage
    results.coveragePercentage =
      (results.coveredTerms.size / this.vocabulary.length) * 100;

    // Generate comprehensive report
    this.generateCoverageReport(results);

    return results;
  }

  /**
   * Generate detailed coverage report
   * @param {Object} results - Analysis results
   */
  generateCoverageReport(results) {
    console.log(`
CBOW COVERAGE REPORT
====================

1. OVERALL COVERAGE
-------------------
  Total terms: ${results.totalTerms}
  Covered terms: ${results.coveredTerms.size} (${results.coveragePercentage.toFixed(1)}%)
  Missing terms: ${results.missingTerms.length}

2. CONTEXT STATISTICS
---------------------
  Total Context Windows: ${results.contextStats.totalContexts}
  Average Context Size: ${results.contextStats.avgContextSize.toFixed(2)}
  Context Size Distribution: ${Object.entries(results.contextStats.contextSizeDistribution)
    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
    .map(([size, count]) => `${size}:${count}`)
    .join(', ')}

3. CHAPTER COVERAGE
-------------------
`);

    for (const [chapter, stats] of Object.entries(results.chapterCoverage).sort()) {
      console.log(`  ${chapter}:
    Terms: ${stats.total}
    Coverage: ${(stats.covered / stats.total * 100).toFixed(1)}%
    Missing: ${stats.total - stats.covered}`);
    }

    if (results.missingTerms.length > 0) {
      console.log(`
4. SAMPLE MISSING TERMS
-----------------------
`);
      console.table(results.missingTerms.slice(0, 10));
    }

    console.log(`
5. TERM APPEARANCE ANALYSIS
---------------------------
  Sample term appearances (target/context):
`);

    // Show sample term appearances
    const sampleTerms = ['goal', 'persistence', 'visualization', 'success', 'discipline'];
    sampleTerms.forEach(term => {
      const stats = results.termStats[term];
      if (stats) {
        console.log(`  ${term} (${stats.chapter}):
    Target appearances: ${stats.asTarget}
    Context appearances: ${stats.asContext}
    Total appearances: ${stats.asTarget + stats.asContext}
    Coverage: ${stats.covered ? 'YES' : 'NO'}`);
      }
    });
  }
}

/**
 * Performance Tester
 *
 * Tests memory usage and performance of the CBOW model
 */
class PerformanceTester {
  constructor() {
    this.history = [];
  }

  /**
   * Run memory and performance test
   * @param {CBOWModel} network - CBOW model instance
   * @param {Array} trainingPairs - Array of training pairs
   * @param {Object} testOptions - Test options
   * @returns {Object} Test results
   */
  runMemoryTest(network, trainingPairs, testOptions = {}) {
    const {
      epochs = 1,
      initialBatchSize = 1000,
      testName = 'Default Test'
    } = testOptions;

    const testResult = {
      testName,
      startTime: Date.now(),
      startMemory: process.memoryUsage(),
      peakMemory: { ...process.memoryUsage() },
      endMemory: null,
      trainingMetrics: [],
      gcStats: []
    };

    console.log(`\nStarting ${testName}...
      Initial memory: ${Math.round(testResult.startMemory.heapUsed/1024/1024)}MB
      Training pairs: ${trainingPairs.length}
      Vocabulary size: ${network.vocabularySize}
      Epochs: ${epochs}
      Initial batch size: ${initialBatchSize}`);

    // Run training with memory monitoring
    const trainingResult = network.trainBatch(trainingPairs, {
      epochs,
      batchSize: initialBatchSize
    });

    // Capture final state
    testResult.endMemory = process.memoryUsage();
    testResult.durationMs = Date.now() - testResult.startTime;
    testResult.trainingResult = trainingResult;

    // Calculate metrics
    testResult.memoryIncreaseMB = (testResult.peakMemory.heapUsed - testResult.startMemory.heapUsed)/1024/1024;
    testResult.averageMemoryMB = testResult.trainingMetrics.reduce(
      (sum, metric) => sum + metric.memory.heapUsed, 0
    ) / testResult.trainingMetrics.length / 1024 / 1024;

    // Generate report
    this.generateTestReport(testResult);
    this.history.push(testResult);

    return testResult;
  }

  /**
   * Generate test report
   * @param {Object} testResult - Test results
   */
  generateTestReport(testResult) {
    console.log(`
${testResult.testName} - Performance Report
--------------------------------------------
Training Duration: ${(testResult.durationMs/1000).toFixed(1)} seconds
Total Pairs Processed: ${testResult.trainingResult ? testResult.trainingResult.processedPairs || 'N/A' : 'N/A'}

Memory Usage:
  Start: ${Math.round(testResult.startMemory.heapUsed/1024/1024)}MB
  Peak: ${Math.round(testResult.peakMemory.heapUsed/1024/1024)}MB
  End: ${Math.round(testResult.endMemory.heapUsed/1024/1024)}MB
  Increase: ${testResult.memoryIncreaseMB.toFixed(1)}MB
  Average: ${testResult.averageMemoryMB ? testResult.averageMemoryMB.toFixed(1) : 'N/A'}MB

Performance:
  Pairs per second: ${testResult.trainingResult ? (testResult.trainingResult.processedPairs*1000/testResult.durationMs).toFixed(1) : 'N/A'}
  GC Events: ${testResult.gcStats.length}
`);
  }

  /**
   * Compare two tests
   * @param {Object} test1 - First test
   * @param {Object} test2 - Second test
   */
  compareTests(test1, test2) {
    console.log(`
Comparison: ${test1.testName} vs ${test2.testName}
--------------------------------------------
Memory Efficiency:
  ${test1.testName}: ${test1.trainingResult ? (test1.trainingResult.processedPairs/test1.memoryIncreaseMB).toFixed(1) : 'N/A'} pairs/MB
  ${test2.testName}: ${test2.trainingResult ? (test2.trainingResult.processedPairs/test2.memoryIncreaseMB).toFixed(1) : 'N/A'} pairs/MB
  Improvement: ${test1.trainingResult && test2.trainingResult ?
    ((test2.trainingResult.processedPairs/test2.memoryIncreaseMB -
      test1.trainingResult.processedPairs/test1.memoryIncreaseMB) /
     (test1.trainingResult.processedPairs/test1.memoryIncreaseMB)*100).toFixed(1) : 'N/A'}%

Speed:
  ${test1.testName}: ${test1.trainingResult ? (test1.trainingResult.processedPairs*1000/test1.durationMs).toFixed(1) : 'N/A'} pairs/sec
  ${test2.testName}: ${test2.trainingResult ? (test2.trainingResult.processedPairs*1000/test2.durationMs).toFixed(1) : 'N/A'} pairs/sec
  Improvement: ${test1.trainingResult && test2.trainingResult ?
    ((test2.trainingResult.processedPairs/test2.durationMs -
      test1.trainingResult.processedPairs/test1.durationMs) /
     (test1.trainingResult.processedPairs/test1.durationMs)*100).toFixed(1) : 'N/A'}%
`);
  }
}

/**
 * Training Validator
 *
 * Validates the quality of training results
 */
class TrainingValidator {
  /**
   * Initialize the validator
   * @param {CBOWModel} network - CBOW model instance
   * @param {Array} vocabulary - Array of vocabulary terms
   * @param {Object} chapterMap - Mapping of terms to chapters
   */
  constructor(network, vocabulary, chapterMap) {
    this.network = network;
    this.vocabulary = vocabulary;
    this.chapterMap = chapterMap;
    this.termIndexCache = {};
    this.vocabulary.forEach((term, index) => {
      this.termIndexCache[term] = index;
    });
  }

  /**
   * Validate training results
   * @returns {Object} Validation results
   */
  validateTraining() {
    const results = {
      termTests: this.testCriticalTerms(),
      semanticTests: this.testSemanticRelationships(),
      embeddingStats: this.analyzeEmbeddingProperties(),
      chapterCoverage: this.testChapterCoverage()
    };

    this.generateValidationReport(results);
    return results;
  }

  /**
   * Test embeddings for critical terms
   * @returns {Array} Test results
   */
  testCriticalTerms() {
    const testPairs = [
      { early: "introduction", late: "persistence", relationship: "foundation vs execution" },
      { early: "goal", late: "visualization", relationship: "planning vs technique" },
      { early: "plan", late: "discipline", relationship: "preparation vs execution" },
      { early: "success", late: "obstacle", relationship: "outcome vs challenge" },
      { early: "dream", late: "action", relationship: "aspiration vs implementation" }
    ];

    return testPairs.map(pair => {
      const earlyIdx = this.termIndexCache[pair.early];
      const lateIdx = this.termIndexCache[pair.late];

      if (earlyIdx === undefined || lateIdx === undefined) {
        return { ...pair, status: "missing", error: "Term not found in vocabulary" };
      }

      const earlyEmbed = this.network.weights[0][earlyIdx];
      const lateEmbed = this.network.weights[0][lateIdx];

      // Calculate embedding properties
      const earlyMag = this.calculateMagnitude(earlyEmbed);
      const lateMag = this.calculateMagnitude(lateEmbed);
      const similarity = this.cosineSimilarity(earlyEmbed, lateEmbed);

      return {
        ...pair,
        status: "complete",
        earlyMagnitude: earlyMag,
        lateMagnitude: lateMag,
        similarity,
        magnitudeRatio: lateMag / earlyMag,
        analysis: this.analyzeTermPair(earlyMag, lateMag, similarity, pair.relationship)
      };
    });
  }

  /**
   * Test semantic relationships
   * @returns {Array} Test results
   */
  testSemanticRelationships() {
    const relationshipTests = [
      ["goal", "objective", [0.7, 1.0], "synonyms"],
      ["plan", "strategy", [0.6, 0.9], "related concepts"],
      ["success", "failure", [0.3, 0.6], "opposites"],
      ["action", "implementation", [0.7, 1.0], "synonyms"],
      ["visualization", "imagination", [0.6, 0.9], "related techniques"],
      ["persistence", "discipline", [0.7, 1.0], "complementary virtues"],
      ["obstacle", "challenge", [0.8, 1.0], "synonyms"],
      ["dream", "reality", [0.4, 0.7], "contrasting concepts"]
    ];

    return relationshipTests.map(([term1, term2, [minSim, maxSim], desc]) => {
      const idx1 = this.termIndexCache[term1];
      const idx2 = this.termIndexCache[term2];

      if (idx1 === undefined || idx2 === undefined) {
        return {
          terms: [term1, term2],
          relationship: desc,
          status: "missing",
          error: "One or both terms not found"
        };
      }

      const embed1 = this.network.weights[0][idx1];
      const embed2 = this.network.weights[0][idx2];
      const similarity = this.cosineSimilarity(embed1, embed2);

      const inRange = similarity >= minSim && similarity <= maxSim;
      const mag1 = this.calculateMagnitude(embed1);
      const mag2 = this.calculateMagnitude(embed2);

      return {
        terms: [term1, term2],
        relationship: desc,
        similarity,
        expectedRange: [minSim, maxSim],
        inRange,
        magnitude1: mag1,
        magnitude2: mag2,
        status: inRange ? "pass" : "fail",
        analysis: inRange ?
          "Relationship correctly learned" :
          `Similarity ${similarity.toFixed(3)} outside expected range [${minSim}, ${maxSim}]`
      };
    });
  }

  /**
   * Calculate vector magnitude
   * @param {Float64Array} vector - Vector to calculate magnitude for
   * @returns {number} Magnitude
   */
  calculateMagnitude(vector) {
    return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  }

  /**
   * Calculate cosine similarity between two vectors
   * @param {Float64Array} a - First vector
   * @param {Float64Array} b - Second vector
   * @returns {number} Cosine similarity
   */
  cosineSimilarity(a, b) {
    let dotProduct = 0;
    let aMagnitude = 0;
    let bMagnitude = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      aMagnitude += a[i] * a[i];
      bMagnitude += b[i] * b[i];
    }

    aMagnitude = Math.sqrt(aMagnitude);
    bMagnitude = Math.sqrt(bMagnitude);

    return aMagnitude * bMagnitude === 0 ? 0 : dotProduct / (aMagnitude * bMagnitude);
  }

  /**
   * Generate validation report
   * @param {Object} results - Validation results
   */
  generateValidationReport(results) {
    console.log(`
TRAINING VALIDATION REPORT
==========================

1. CRITICAL TERM ANALYSIS
-------------------------
`);
    results.termTests.forEach(test => {
      console.log(`  ${test.early} vs ${test.late} (${test.relationship}):
    Early magnitude: ${test.earlyMagnitude ? test.earlyMagnitude.toFixed(4) : 'N/A'}
    Late magnitude: ${test.lateMagnitude ? test.lateMagnitude.toFixed(4) : 'N/A'}
    Similarity: ${test.similarity ? test.similarity.toFixed(4) : 'N/A'}
    Analysis: ${test.analysis || 'N/A'}`);
    });

    console.log(`
2. SEMANTIC RELATIONSHIP TESTS
-----------------------------
`);
    results.semanticTests.forEach(test => {
      console.log(`  ${test.terms[0]} vs ${test.terms[1]} (${test.relationship}):
    Similarity: ${test.similarity ? test.similarity.toFixed(4) : 'N/A'}
    Expected: [${test.expectedRange ? test.expectedRange.join(', ') : 'N/A'}]
    Status: ${test.status || 'N/A'}
    Analysis: ${test.analysis || 'N/A'}`);
    });

    console.log(`
3. EMBEDDING STATISTICS
-----------------------
    Average magnitude: ${results.embeddingStats.averageMagnitude ? results.embeddingStats.averageMagnitude.toFixed(4) : 'N/A'}
    Magnitude range: [${results.embeddingStats.minMagnitude ? results.embeddingStats.minMagnitude.toFixed(4) : 'N/A'},
                     ${results.embeddingStats.maxMagnitude ? results.embeddingStats.maxMagnitude.toFixed(4) : 'N/A'}]`);
  }

  /**
   * Analyze embedding properties
   * @returns {Object} Embedding statistics
   */
  analyzeEmbeddingProperties() {
    let totalMagnitude = 0;
    let minMagnitude = Infinity;
    let maxMagnitude = -Infinity;
    let count = 0;

    for (let i = 0; i < this.network.weights[0].length; i++) {
      const embedding = this.network.weights[0][i];
      const magnitude = this.calculateMagnitude(embedding);

      totalMagnitude += magnitude;
      minMagnitude = Math.min(minMagnitude, magnitude);
      maxMagnitude = Math.max(maxMagnitude, magnitude);
      count++;
    }

    return {
      averageMagnitude: totalMagnitude / count,
      minMagnitude,
      maxMagnitude
    };
  }
}

// Export the validation classes
export { CBOWCoverageValidator, PerformanceTester, TrainingValidator };