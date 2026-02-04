import fs from 'fs';
const infile = process.argv[2] || 'goals_embedding_network.json';
const outCsv = 'embeddings.csv';
const outJson = 'embeddings.json';

function loadJSON(p) {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch (e) {
    console.error('Failed to read/parse', p, e.message);
    process.exit(2);
  }
}

const json = loadJSON(infile);
if (!json.weights || !Array.isArray(json.weights) || json.weights.length === 0) {
  console.error('Invalid network file: missing weights');
  process.exit(3);
}

const w0 = json.weights[0];
if (!Array.isArray(w0)) {
  console.error('Unexpected weights[0] structure — expected array.');
  process.exit(4);
}

const vocab = w0.length;
const first = w0[0];
if (!first) {
  console.error('weights[0] appears empty');
  process.exit(5);
}

// Determine embedding dimension: object keys or array length
let dim;
if (Array.isArray(first)) dim = first.length;
else dim = Object.keys(first).length;

console.log('vocab size:', vocab);
console.log('embedding dim:', dim);

// Write JSON array of embeddings
const embeddings = new Array(vocab);
for (let i = 0; i < vocab; i++) {
  const rowObj = w0[i];
  const row = new Array(dim);
  for (let j = 0; j < dim; j++) {
    if (Array.isArray(rowObj)) row[j] = Number(rowObj[j] || 0);
    else row[j] = Number(rowObj[j] !== undefined ? rowObj[j] : rowObj[String(j)] || 0);
  }
  embeddings[i] = row;
}
fs.writeFileSync(outJson, JSON.stringify({vocab, dim, embeddings}, null, 2));

// Write CSV
const header = ['index', ...Array.from({length: dim}, (_,i) => `d${i}`)].join(',');
const lines = [header];
for (let i = 0; i < embeddings.length; i++) {
  lines.push([i, ...embeddings[i]].join(','));
}
fs.writeFileSync(outCsv, lines.join('\n'));

console.log('Wrote', outJson, 'and', outCsv);

// Print a small sample
console.log('Sample row 0:', embeddings[0].slice(0, Math.min(10, dim)));

process.exit(0);
