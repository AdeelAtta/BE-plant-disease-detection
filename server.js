const express = require('express');
const multer = require('multer');
const onnx = require('onnxruntime-node');
const sharp = require('sharp');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Load class labels
const classLabels = JSON.parse(fs.readFileSync(path.join(__dirname, './classes.json')));

// Global variable to hold our ONNX session
let session;

// Function to load the ONNX model
async function loadModel() {
    try {
        session = await onnx.InferenceSession.create(path.join(__dirname, './model.onnx'));
        console.log('Model loaded successfully');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
    } catch (error) {
        console.error('Error loading model:', error);
        throw error;
    }
}

// Preprocess image function
async function preprocessImage(imageBuffer) {
    try {
        // Resize image to 256x256 to match model's expected input size
        const processedImage = await sharp(imageBuffer)
            .resize(256, 256)  // Updated from 224 to 256
            .removeAlpha()    
            .raw()            
            .toBuffer();

        // Convert to float32 array and normalize
        const float32Data = new Float32Array(processedImage.length);
        for (let i = 0; i < processedImage.length; i++) {
            float32Data[i] = processedImage[i] / 255.0;  // Normalize to [0,1]
        }

        // Reshape to match model's input shape [1, 3, 256, 256]
        const tensorData = new Float32Array(1 * 3 * 256 * 256);
        for (let c = 0; c < 3; c++) {
            for (let h = 0; h < 256; h++) {
                for (let w = 0; w < 256; w++) {
                    tensorData[c * 256 * 256 + h * 256 + w] = 
                        float32Data[(h * 256 + w) * 3 + c];
                }
            }
        }

        return tensorData;
    } catch (error) {
        throw new Error('Error preprocessing image: ' + error.message);
    }
}

// Update the prediction endpoint
app.post('/api/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        if (!session) {
            return res.status(500).json({ error: 'Model not loaded' });
        }

        const preprocessedData = await preprocessImage(req.file.buffer);
        const inputTensor = new onnx.Tensor(
            'float32',
            preprocessedData,
            [1, 3, 256, 256]
        );

        const feeds = { [session.inputNames[0]]: inputTensor };
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]];

        // Apply softmax to get proper probabilities
        const probabilities = softmax(Array.from(output.data));
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const predictedClass = classLabels[maxIndex];
        const confidenceScore = (probabilities[maxIndex] * 100).toFixed(2)
        // Return only the top prediction
        res.json({
            predicted_class: predictedClass ,
            confidence: confidenceScore 
        });

    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Error processing image: ' + error.message });
    }
});

// Add softmax function to normalize probabilities
function softmax(arr) {
    const maxVal = Math.max(...arr);
    const expArr = arr.map(val => Math.exp(val - maxVal));
    const sumExp = expArr.reduce((acc, val) => acc + val, 0);
    return expArr.map(val => val / sumExp);
}

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok',
        modelLoaded: !!session,
        numClasses: classLabels.length
    });
});

// Start the server
const PORT = process.env.PORT || 3500;
app.listen(PORT, async () => {
    try {
        await loadModel();
        console.log(`Server running on http://localhost:${PORT}`);
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
});