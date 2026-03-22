const { onRequest } = require("firebase-functions/v2/https");
const { onSchedule } = require("firebase-functions/v2/scheduler");
const admin = require("firebase-admin");

admin.initializeApp();
const db = admin.firestore();

// ── claimJob ──────────────────────────────────────────────────────────────────
// Workers call this to atomically claim the next pending job.
// Auth: Authorization: Bearer <WORKER_SECRET>
//
// Response 200: { job_id, params }
// Response 204: no pending jobs

exports.claimJob = onRequest({ secrets: ["WORKER_SECRET"], invoker: "public" }, async (req, res) => {
  if (req.method !== "POST") return res.status(405).send("Method Not Allowed");

  const auth = req.headers.authorization || "";
  if (auth !== `Bearer ${process.env.WORKER_SECRET}`) {
    return res.status(401).send("Unauthorized");
  }

  const { worker_id } = req.body || {};
  if (!worker_id) return res.status(400).json({ error: "Missing worker_id" });

  const now = admin.firestore.FieldValue.serverTimestamp();

  try {
    const result = await db.runTransaction(async (tx) => {
      const snapshot = await tx.get(
        db.collection("jobs")
          .where("status", "==", "pending")
          .orderBy("created_at", "asc")
          .limit(1)
      );

      if (snapshot.empty) return null;

      const doc = snapshot.docs[0];
      tx.update(doc.ref, {
        status: "processing",
        worker_id,
        updated_at: now,
        heartbeat_at: now,
      });

      return { job_id: doc.id, params: doc.data().params };
    });

    if (!result) return res.status(204).send();
    return res.status(200).json(result);

  } catch (e) {
    console.error("claimJob error:", e);
    return res.status(500).json({ error: e.message });
  }
});
