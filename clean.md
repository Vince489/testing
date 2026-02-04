

Here are the steps to clean and prepare data for a **CBOW** model.

---

## 1. The Cleaning Pipeline

You want to remove everything that doesn't carry "semantic meaning." For basic embeddings, "Apple," "apple," and "apple!" should all be treated as the same coordinate.

1. **Lowercasing:** Convert all text to lowercase so the model doesn't think "The" and "the" are different concepts.
2. **Noise Removal:** Use a **Regular Expression** (Regex) to strip out punctuation, special characters, and numbers.
3. **Tokenization:** Split the long string into an array of individual words.

```javascript
const fs = require('fs');

function cleanText(filePath) {
    const rawText = fs.readFileSync(filePath, 'utf8');
    
    return rawText
        .toLowerCase()
        // Replace everything that isn't a letter or space with nothing
        .replace(/[^a-z\s]/g, '') 
        // Split by whitespace and filter out empty strings
        .split(/\s+/)
        .filter(word => word.length > 0);
}

const tokens = cleanText('mybook.txt');
console.log(`Total tokens: ${tokens.length}`);

```

---

## 2. Building the Vocabulary (The Dictionary)

Now you need to count how often words appear. As mentioned, rare words (appearing only 1 or 2 times) are "noise" for a small CPU-based model.

* **Count Frequencies:** Use a Map to store word counts.
* **Set a Threshold:** Discard words that appear fewer than, say, 5 times.
* **Assign IDs:** Map each unique word to a number (0, 1, 2...).

---

## 3. Creating the Lookup Tables

Your `NeuralNetwork` class only speaks "Number." You need two "translation" objects:

1. **`wordToIndex`**: `{"coffee": 0, "tea": 1, ...}`
2. **`indexToWord`**: `{0: "coffee", 1: "tea", ...}`

```javascript
function buildVocab(tokens, minCount = 5) {
    const counts = {};
    tokens.forEach(w => counts[w] = (counts[w] || 0) + 1);

    const vocab = Object.keys(counts).filter(w => counts[w] >= minCount);
    const wordToIndex = {};
    const indexToWord = {};

    vocab.forEach((word, i) => {
        wordToIndex[word] = i;
        indexToWord[i] = word;
    });

    return { wordToIndex, indexToWord, vocabSize: vocab.length };
}

```

---

## 4. Generating CBOW Training Pairs

In **CBOW**, your input is the **context** and your output is the **target word**.
If your window size is 2, for every word in the book, you grab 2 words before and 2 words after.

**Example Sentence:** "The quick brown fox jumps"

* **Target:** "brown" (index 2)
* **Context:** ["the", "quick", "fox", "jumps"] (indices [0, 1, 3, 4])

---

## 5. Why this helps your "No-GPU" situation

By doing this pre-processing:

* You reduce your "Input Size" from potentially 15,000 unique words down to ~3,000 useful words.
* You convert the entire 70k word book into a single **Int32Array**. This is much faster for the CPU to iterate over than a massive array of strings.