import { KeyGen } from "./keygen.js";

async function runSystemTest() {
    // 1. System Setup (Do this once)
const mnemonic = KeyGen.generateMnemonic();
const MY_MASTER_SECRET = mnemonic;

    // 2. User Signup: Create Account for User #4
    const user4 = await KeyGen.deriveAccount(MY_MASTER_SECRET, 4);
    console.log(`Public Key (Store in DB): ${user4.publicKey}`);

    // 3. User Action: User #4 signs a request
    const action = "Delete-Object-99";
    const userSignature = KeyGen.sign(action, user4.secretKey);
    console.log(`User Signature: ${userSignature}`);

    // 4. Server Validation: Verify the signature using the DB record
    // In a real app, you'd fetch user4.publicKey from MongoDB
    const isLegit = KeyGen.verify(action, userSignature, user4.publicKey);

    console.log("\n--- Server Security Check ---");
    console.log(`Is request valid? ${isLegit ? "✅ AUTHORIZED" : "❌ REJECTED"}`);
}

runSystemTest();