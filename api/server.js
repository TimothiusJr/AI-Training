const express = require("express");
const multer = require("multer");
const { execFile } = require("child_process");
const path = require("path");

const app = express();
const upload = multer({ dest: "uploads/" });

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


            res.json({ result: stdout.trim() });
        }
    );
});

app.listen(3000, () => {
    console.log("API running at http://localhost:3000");
});
