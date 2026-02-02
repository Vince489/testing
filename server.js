const express = require('express');
const app = express();
const PORT = 3000;

// Serve static files from the current directory
app.use(express.static('.'));

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
