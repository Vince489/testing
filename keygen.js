import * as bip39 from 'bip39';
import { HDKey } from '@scure/bip32';
import nacl from 'tweetnacl';
import bs58 from 'bs58';

class KeyGen {
    /**
     * Creates a new 12-word mnemonic phrase.
     */
    static generateMnemonic() {
        return bip39.generateMnemonic();
    }

    /**
     * Derives a full Ed25519 keypair from a mnemonic and a specific user index.
     * Use this for your centralized identity management.
     */
static async deriveAccount(mnemonic, index = 0) {
        console.log('Starting keypair derivation...');
        console.log('Input mnemonic:', mnemonic);

        try {
            const seed = await bip39.mnemonicToSeed(mnemonic);
            console.log('Seed generated successfully');

            const root = HDKey.fromMasterSeed(seed);
            console.log('HDKey root created successfully');

            // Path: m/0'/index' (Hardened for Ed25519 security)
            const path = `m/0'/${index}'`;
            console.log('Derivation path:', path);

            const child = root.derive(path);
            console.log('Child key derived successfully');

            // Derive the TweetNaCl keypair using the 32-byte private key as the seed
            const keyPair = nacl.sign.keyPair.fromSeed(child.privateKey);
            console.log('Keypair generated successfully');

            const result = {
                path,
                index,
                publicKey: bs58.encode(keyPair.publicKey),
                secretKey: bs58.encode(keyPair.secretKey),
                raw: keyPair // Useful if you need the Uint8Arrays immediately
            };

            return result;
        } catch (error) {
            console.error('Error in deriveAccount:', error);
            throw error;
        }
    }

    /**
     * Signs a string message using a Base58 encoded secret key.
     */
    static sign(message, secretKeyBase58) {
        const messageBytes = new TextEncoder().encode(message);
        const secretKeyBytes = bs58.decode(secretKeyBase58);
        const signatureBytes = nacl.sign.detached(messageBytes, secretKeyBytes);
        return bs58.encode(signatureBytes);
    }

    /**
     * Verifies a signature against a message and a Base58 encoded public key.
     */
    static verify(message, signatureBase58, publicKeyBase58) {
        try {
            const messageBytes = new TextEncoder().encode(message);
            const signatureBytes = bs58.decode(signatureBase58);
            const publicKeyBytes = bs58.decode(publicKeyBase58);
            return nacl.sign.detached.verify(messageBytes, signatureBytes, publicKeyBytes);
        } catch (e) {
            return false;
        }
    }
}

export { KeyGen };