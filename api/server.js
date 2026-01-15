const express = require("express");
const multer = require("multer");
const { execFile } = require("child_process");
const path = require("path");

const app = express();
const upload = multer({ dest: "uploads/" });

const sqlite3 = require("sqlite3").verbose();
const db = new sqlite3.Database("predictions.db");


app.post("/predict", upload.single("audio"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No audio file uploaded" });
    }

    const audioPath = path.resolve(req.file.path);

    execFile(
        "python3",
        ["../predict.py", audioPath],
        (error, stdout, stderr) => {
            if (error) {
                console.error("STDERR:", stderr);
                console.error("ERROR:", error);
                return res.status(500).json({
                    error: "Prediction failed",
                    details: stderr || error.message
                });
            }


            const output = stdout.trim();

            // Parse label and confidence
            const match = output.match(/Predicted: (\w+) \(confidence: ([0-9.]+)\)/);

            if (!match) {
                return res.status(500).json({ error: "Invalid model output" });
            }

            const label = match[1];
            const confidence = parseFloat(match[2]);

            db.run(
                `INSERT INTO predictions (filename, label, confidence)
                        VALUES (?, ?, ?)`,
                [req.file.originalname, label, confidence]
            );

            res.json({
                label,
                confidence
            });

        }
    );
});

app.get("/predictions", (req, res) => {
    db.all(
        `SELECT id, filename, label, confidence, created_at
         FROM predictions
         ORDER BY created_at DESC
         LIMIT 20`,
        [],
        (err, rows) => {
            if (err) {
                console.error(err);
                return res.status(500).json({ error: "Database query failed" });
            }

            res.json(rows);
        }
    );
});


app.listen(3000, () => {
    console.log("API running at http://localhost:3000");
});
