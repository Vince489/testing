import express from 'express';
const app = express();
const PORT = 3000;
import fs from 'fs';
import path from 'path';

// Serve static files from the current directory
app.use(express.static('.'));

// Middleware to parse JSON request bodies
app.use(express.json());

// Endpoint to register account numbers
app.post('/register-account', (req, res) => {
  console.log('Received request:', req.method, req.url);
  console.log('Request body:', req.body);
  console.log('Request headers:', req.headers);

  const accountNumber = req.body.accountNumber;

  if (!accountNumber) {
    console.log('Account number is missing or empty');
    return res.status(400).json({ error: 'Account number is required' });
  }

const accountsFilePath = path.join(process.cwd(), 'accounts.json');

  try {
    // Read existing accounts
    let accounts = [];
    if (fs.existsSync(accountsFilePath)) {
      const data = fs.readFileSync(accountsFilePath, 'utf8');
      accounts = JSON.parse(data);
    }

    // Add new account as object
    accounts.push({ accountNumber: accountNumber });

    // Write updated accounts back to file
    fs.writeFileSync(accountsFilePath, JSON.stringify(accounts, null, 2));

    res.send('account registered successfully');
  } catch (error) {
    console.error('Error registering account:', error);
    res.status(500).json({ error: 'Failed to register account' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});