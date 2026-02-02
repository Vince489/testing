const express = require('express');
const app = express();
const PORT = 3000;

// The "Hello World" route
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});