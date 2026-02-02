import { KeyGen } from './keygen.js';

async function main() {
  // Generate a new keypair
  const mnemonic = KeyGen.generateMnemonic();
  const keypair = await KeyGen.deriveAccount(mnemonic);

  // Extract public key as account number
  const accountNumber = keypair.publicKey;

  // Prepare the account data
  const accountData = JSON.stringify({ accountNumber });

  // Use fetch API to send request
  async function createAccount() {
    try {
      const response = await fetch('http://localhost:3000/register-account', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: accountData
      });

      const responseData = await response.text();
      console.log('Response received:', response.status);
      console.log('Response data:', responseData);

      if (response.status === 200) {
        console.log('Account created successfully!');
        console.log('Public Key (Account Number):', keypair.publicKey);
        console.log('Secret Key:', keypair.secretKey);
        console.log('Path:', keypair.path);
        console.log('Index:', keypair.index);
      } else {
        throw new Error(`Server responded with status ${response.status}: ${responseData}`);
      }
    } catch (error) {
      throw new Error(`Error creating account: ${error.message}`);
    }
  }

  await createAccount();
}

main();
