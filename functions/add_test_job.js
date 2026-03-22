// Run from functions/ directory: node add_test_job.js
// Uses Firebase CLI's application default credentials (must be logged in via firebase login)

const admin = require("firebase-admin");
const { v4: uuidv4 } = require("uuid");

admin.initializeApp({
  credential: admin.credential.applicationDefault(),
  storageBucket: "spawnlab-53283.firebasestorage.app",
});

const db = admin.firestore();
const bucket = admin.storage().bucket();

// Minimal 64x64 solid orange PNG (generated offline, no canvas needed)
function makeTestPng() {
  const { createCanvas } = (() => {
    try { return require("canvas"); } catch { return null; }
  })() || {};

  if (createCanvas) {
    const c = createCanvas(64, 64);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ff6633";
    ctx.fillRect(0, 0, 64, 64);
    return c.toBuffer("image/png");
  }

  // Fallback: minimal 1x1 red PNG (raw bytes)
  return Buffer.from(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009001" +
    "2e0000000c4944415408d76360f8cfc00000000200016e21bc330000000049454e44ae426082",
    "hex"
  );
}

async function main() {
  const jobId = uuidv4();
  console.log(`Creating test job: ${jobId}`);

  const imgBuffer = makeTestPng();
  await bucket.file(`inputs/${jobId}.png`).save(imgBuffer, { contentType: "image/png" });
  console.log("Image uploaded to Storage.");

  await db.collection("jobs").doc(jobId).set({
    uid: "test-user",
    status: "pending",
    params: {
      resolution: 512,
      seed: 42,
      decimation_target: 100000,
      texture_size: 1024,
      remove_bg: false,
    },
    created_at: admin.firestore.FieldValue.serverTimestamp(),
    updated_at: admin.firestore.FieldValue.serverTimestamp(),
    heartbeat_at: null,
    worker_id: null,
    error: null,
  });

  console.log(`Done. Job is now pending.`);
  console.log(`Firestore: https://console.firebase.google.com/project/spawnlab-53283/firestore/data/~2Fjobs~2F${jobId}`);

  // Verify claimJob picks it up
  const axios = require("axios").default;
  const resp = await axios.post(
    "https://us-central1-spawnlab-53283.cloudfunctions.net/claimJob",
    { worker_id: "test-local" },
    { headers: { Authorization: "Bearer xFdOkPL_AH_RmblUE--ZUfT0m5Rz51aZw6OJvD9uHkw" } }
  );
  console.log(`claimJob response: ${resp.status}`, resp.data);
  process.exit(0);
}

main().catch(e => { console.error(e.message); process.exit(1); });
