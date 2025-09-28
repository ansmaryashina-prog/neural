// Titanic Binary Classifier using TensorFlow.js
// Schema: Target: Survived (0/1), Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// To reuse with other datasets: Update the schema variables below and preprocessing logic

// Global variables to store data and model
let trainData = null;
let testData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let testPredictions = null;
let rocData = null;
let aucScore = 0;

// Schema configuration - UPDATE THESE FOR OTHER DATASETS
const TARGET_COLUMN = 'Survived';
const FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
const ID_COLUMN = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// DOM elements
const elements = {
    loadDataBtn: document.getElementById('loadData'),
    preprocessDataBtn: document.getElementById('preprocessData'),
    createModelBtn: document.getElementById('createModel'),
    trainModelBtn: document.getElementById('trainModel'),
    evaluateModelBtn: document.getElementById('evaluateModel'),
    predictTestBtn: document.getElementById('predictTest'),
    exportSubmissionBtn: document.getElementById('exportSubmission'),
    exportProbabilitiesBtn: document.getElementById('exportProbabilities'),
    saveModelBtn: document.getElementById('saveModel'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    dataStatus: document.getElementById('dataStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    exportStatus: document.getElementById('exportStatus'),
    accuracy: document.getElementById('accuracy'),
    precision: document.getElementById('precision'),
    recall: document.getElementById('recall'),
    f1: document.getElementById('f1'),
    auc: document.getElementById('auc')
};

// Event listeners
elements.loadDataBtn.addEventListener('click', loadData);
elements.preprocessDataBtn.addEventListener('click', preprocessData);
elements.createModelBtn.addEventListener('click', createModel);
elements.trainModelBtn.addEventListener('click', trainModel);
elements.evaluateModelBtn.addEventListener('click', evaluateModel);
elements.predictTestBtn.addEventListener('click', predictTestData);
elements.exportSubmissionBtn.addEventListener('click', exportSubmission);
elements.exportProbabilitiesBtn.addEventListener('click', exportProbabilities);
elements.saveModelBtn.addEventListener('click', saveModel);
elements.thresholdSlider.addEventListener('input', updateThreshold);

// Update threshold display
function updateThreshold() {
    const threshold = parseFloat(elements.thresholdSlider.value);
    elements.thresholdValue.textContent = threshold.toFixed(2);
    if (rocData) {
        updateMetrics(threshold);
    }
}

// Set status message
function setStatus(element, message, type = 'info') {
    element.textContent = message;
    element.className = `status ${type}`;
}

// Load CSV data from file inputs
async function loadData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile) {
        alert('Please select a training CSV file');
        return;
    }
    
    try {
        setStatus(elements.dataStatus, 'Loading data...', 'loading');
        
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data if provided
        if (testFile) {
            const testText = await readFile(testFile);
            testData = parseCSV(testText);
        }
        
        // Data inspection
        await inspectData();
        
        setStatus(elements.dataStatus, `Data loaded! Train: ${trainData.length} samples${testData ? `, Test: ${testData.length} samples` : ''}`, 'success');
        
    } catch (error) {
        setStatus(elements.dataStatus, `Error loading data: ${error.message}`, 'error');
        console.error('Data loading error:', error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    return lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index];
            // Handle numeric values
            if (!isNaN(value) && value !== '') {
                value = parseFloat(value);
            }
            // Handle empty values
            if (value === '') {
                value = null;
            }
            row[header] = value;
        });
        return row;
    });
}

// Inspect and visualize data
async function inspectData() {
    if (!trainData || trainData.length === 0) return;
    
    // Calculate basic statistics
    const shape = { rows: trainData.length, cols: Object.keys(trainData[0]).length };
    const missingPercentages = calculateMissingValues(trainData);
    
    console.log('Data Shape:', shape);
    console.log('Missing Values:', missingPercentages);
    
    // Show data preview
    console.table(trainData.slice(0, 5));
    
    // Create visualizations with tfjs-vis
    await createDataVisualizations();
}

// Calculate missing values percentage
function calculateMissingValues(data) {
    const missing = {};
    const total = data.length;
    
    Object.keys(data[0]).forEach(column => {
        const missingCount = data.filter(row => row[column] === null || row[column] === undefined).length;
        missing[column] = (missingCount / total * 100).toFixed(2) + '%';
    });
    
    return missing;
}

// Create data visualizations using tfjs-vis
async function createDataVisualizations() {
    if (!trainData || !tfvis) return;
    
    // Survival by Sex
    const sexSurvival = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== null) {
            const key = `${row.Sex}-${row.Survived}`;
            sexSurvival[key] = (sexSurvival[key] || 0) + 1;
        }
    });
    
    const sexData = Object.entries(sexSurvival).map(([key, count]) => ({
        index: key,
        value: count
    }));
    
    // Survival by Pclass
    const classSurvival = {};
    trainData.forEach(row => {
        if (row.Pclass && row.Survived !== null) {
            const key = `Class ${row.Pclass}-${row.Survived}`;
            classSurvival[key] = (classSurvival[key] || 0) + 1;
        }
    });
    
    const classData = Object.entries(classSurvival).map(([key, count]) => ({
        index: key,
        value: count
    }));
    
    // Create surface for visualizations
    const surface = { name: 'Data Distribution', tab: 'Data Inspection' };
    
    await tfvis.render.barchart(surface, sexData, {
        xLabel: 'Sex & Survival',
        yLabel: 'Count'
    });
    
    const surface2 = { name: 'Survival by Class', tab: 'Data Inspection' };
    await tfvis.render.barchart(surface2, classData, {
        xLabel: 'Class & Survival',
        yLabel: 'Count'
    });
}

// Preprocess the data
function preprocessData() {
    if (!trainData) {
        alert('Please load data first');
        return;
    }
    
    try {
        setStatus(elements.preprocessStatus, 'Preprocessing data...', 'loading');
        
        // Process training data
        const processedTrain = processDataset(trainData, true);
        
        // Process test data if available
        let processedTest = null;
        if (testData) {
            processedTest = processDataset(testData, false);
        }
        
        // Update global variables
        trainData = processedTrain.processedData;
        testData = processedTest ? processedTest.processedData : null;
        
        console.log('Processed features:', processedTrain.featureNames);
        console.log('Training data shape:', processedTrain.features.shape);
        console.log('Training labels shape:', processedTrain.labels.shape);
        
        setStatus(elements.preprocessStatus, 
                 `Preprocessing complete! Features: ${processedTrain.featureNames.length}, Shape: ${processedTrain.features.shape}`, 
                 'success');
                 
    } catch (error) {
        setStatus(elements.preprocessStatus, `Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

// Process a dataset (train or test)
function processDataset(data, isTraining) {
    const features = [];
    const labels = [];
    const featureNames = new Set();
    
    // Calculate imputation values from training data
    let ageMedian = 28;
    let embarkedMode = 'S';
    let fareMedian = 14.45;
    
    if (isTraining) {
        const ages = data.map(row => row.Age).filter(age => age !== null);
        const embarked = data.map(row => row.Embarked).filter(e => e !== null);
        const fares = data.map(row => row.Fare).filter(fare => fare !== null);
        
        ageMedian = median(ages);
        embarkedMode = mode(embarked);
        fareMedian = median(fares);
    }
    
    // Process each row
    data.forEach(row => {
        const featureVector = [];
        
        // Handle numerical features
        NUMERICAL_FEATURES.forEach(feature => {
            let value = row[feature];
            
            // Impute missing values
            if (value === null || value === undefined) {
                if (feature === 'Age') value = ageMedian;
                else if (feature === 'Fare') value = fareMedian;
                else value = 0;
            }
            
            // Standardize Age and Fare
            if (feature === 'Age') {
                value = (value - ageMedian) / (data.reduce((max, r) => Math.max(max, r.Age || 0), 0) - ageMedian);
            } else if (feature === 'Fare') {
                value = (value - fareMedian) / (data.reduce((max, r) => Math.max(max, r.Fare || 0), 0) - fareMedian);
            }
            
            featureVector.push(value);
            featureNames.add(feature);
        });
        
        // Handle categorical features (one-hot encoding)
        CATEGORICAL_FEATURES.forEach(feature => {
            let value = row[feature];
            
            // Impute missing values
            if (value === null || value === undefined) {
                if (feature === 'Embarked') value = embarkedMode;
                else value = CATEGORICAL_FEATURES[feature] || 0;
            }
            
            // Create one-hot encoding
            const categories = getCategories(feature);
            const oneHot = new Array(categories.length).fill(0);
            const index = categories.indexOf(value.toString());
            if (index !== -1) {
                oneHot[index] = 1;
            }
            
            oneHot.forEach((val, idx) => {
                featureVector.push(val);
                featureNames.add(`${feature}_${categories[idx]}`);
            });
        });
        
        // Add engineered features if enabled
        const addFamilySize = document.getElementById('addFamilySize').checked;
        const addIsAlone = document.getElementById('addIsAlone').checked;
        
        if (addFamilySize) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            featureVector.push(familySize);
            featureNames.add('FamilySize');
        }
        
        if (addIsAlone) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            featureVector.push(isAlone);
            featureNames.add('IsAlone');
        }
        
        features.push(featureVector);
        
        // Add label for training data
        if (isTraining && row[TARGET_COLUMN] !== null && row[TARGET_COLUMN] !== undefined) {
            labels.push([row[TARGET_COLUMN]]);
        }
    });
    
    return {
        processedData: data, // Keep original data with processed features
        features: tf.tensor2d(features),
        labels: isTraining ? tf.tensor2d(labels) : null,
        featureNames: Array.from(featureNames)
    };
}

// Utility functions
function median(arr) {
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mode(arr) {
    const frequency = {};
    let max = 0;
    let result = null;
    
    arr.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > max) {
            max = frequency[value];
            result = value;
        }
    });
    
    return result;
}

function getCategories(feature) {
    const categories = {
        'Pclass': ['1', '2', '3'],
        'Sex': ['male', 'female'],
        'Embarked': ['C', 'Q', 'S']
    };
    return categories[feature] || [];
}

// Create the neural network model
function createModel() {
    try {
        setStatus(elements.modelStatus, 'Creating model...', 'loading');
        
        // Simple sequential model with one hidden layer
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    // For other datasets, adjust units based on feature complexity
                    units: 16,
                    activation: 'relu',
                    // Input shape will be inferred during first training
                    inputDim: undefined // Will be set during training
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid'
                })
            ]
        });
        
        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Print model summary
        model.summary();
        
        setStatus(elements.modelStatus, 'Model created successfully!', 'success');
        
    } catch (error) {
        setStatus(elements.modelStatus, `Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

// Train the model
async function trainModel() {
    if (!model) {
        alert('Please create the model first');
        return;
    }
    
    if (!trainData || !trainData.features) {
        alert('Please preprocess the data first');
        return;
    }
    
    try {
        setStatus(elements.trainingStatus, 'Training model...', 'loading');
        
        const features = trainData.features;
        const labels = trainData.labels;
        
        // Set input shape if not already set
        if (!model.layers[0].input.shape) {
            model.layers[0] = tf.layers.dense({
                units: 16,
                activation: 'relu',
                inputShape: [features.shape[1]]
            });
            model.compile({
                optimizer: 'adam',
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
        }
        
        // Create validation split (80/20)
        const splitIndex = Math.floor(features.shape[0] * 0.8);
        
        const trainFeatures = features.slice(0, splitIndex);
        const trainLabels = labels.slice(0, splitIndex);
        const valFeatures = features.slice(splitIndex);
        const valLabels = labels.slice(splitIndex);
        
        validationData = { features: valFeatures, labels: valLabels };
        
        // Training configuration
        const trainingConfig = {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                { callbacks: ['onEpochEnd'] }
            )
        };
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, trainingConfig);
        
        setStatus(elements.trainingStatus, 'Training completed!', 'success');
        
    } catch (error) {
        setStatus(elements.trainingStatus, `Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

// Evaluate the model and compute metrics
async function evaluateModel() {
    if (!model || !validationData) {
        alert('Please train the model first');
        return;
    }
    
    try {
        // Get predictions
        const predictions = model.predict(validationData.features);
        const probs = await predictions.data();
        const trueLabels = await validationData.labels.data();
        
        // Compute ROC curve and AUC
        rocData = computeROC(trueLabels, probs);
        aucScore = computeAUC(rocData);
        
        // Plot ROC curve
        plotROC(rocData, aucScore);
        
        // Update metrics with default threshold
        updateMetrics(0.5);
        
        elements.auc.textContent = aucScore.toFixed(4);
        
    } catch (error) {
        console.error('Evaluation error:', error);
        alert('Error during model evaluation');
    }
}

// Compute ROC curve data
function computeROC(trueLabels, probabilities) {
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (let i = 0; i < trueLabels.length; i++) {
            const prediction = probabilities[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 1 && actual === 0) fp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 0 && actual === 1) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocPoints.push({ threshold, fpr, tpr, tp, fp, tn, fn });
    });
    
    return rocPoints;
}

// Compute AUC using trapezoidal rule
function computeAUC(rocData) {
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        const prev = rocData[i - 1];
        const curr = rocData[i];
        auc += (curr.fpr - prev.fpr) * (curr.tpr + prev.tpr) / 2;
    }
    return Math.abs(auc);
}

// Plot ROC curve using tfjs-vis
function plotROC(rocData, auc) {
    const rocSeries = rocData.map(point => ({
        x: point.fpr,
        y: point.tpr
    }));
    
    const surface = { name: 'ROC Curve', tab: 'Model Evaluation' };
    
    tfvis.render.linechart(surface, {
        values: [rocSeries],
        series: ['ROC Curve']
    }, {
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        width: 400,
        height: 400
    });
    
    // Add AUC to the plot
    const aucSurface = { name: 'AUC Score', tab: 'Model Evaluation' };
    tfvis.render.table(aucSurface, {
        headers: ['Metric', 'Value'],
        values: [['AUC-ROC', auc.toFixed(4)]]
    });
}

// Update metrics based on threshold
function updateMetrics(threshold) {
    if (!rocData) return;
    
    const point = rocData.find(p => Math.abs(p.threshold - threshold) < 0.01) || rocData[Math.round(threshold * 100)];
    
    if (point) {
        const accuracy = (point.tp + point.tn) / (point.tp + point.tn + point.fp + point.fn);
        const precision = point.tp / (point.tp + point.fp) || 0;
        const recall = point.tp / (point.tp + point.fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        elements.accuracy.textContent = accuracy.toFixed(4);
        elements.precision.textContent = precision.toFixed(4);
        elements.recall.textContent = recall.toFixed(4);
        elements.f1.textContent = f1.toFixed(4);
        
        // Plot confusion matrix
        plotConfusionMatrix(point);
    }
}

// Plot confusion matrix
function plotConfusionMatrix(point) {
    const surface = { name: 'Confusion Matrix', tab: 'Model Evaluation' };
    
    tfvis.render.confusionMatrix(surface, {
        values: [
            [point.tn, point.fp],
            [point.fn, point.tp]
        ]
    }, {
        labels: ['Not Survived (0)', 'Survived (1)']
    });
}

// Predict on test data
async function predictTestData() {
    if (!model || !testData || !testData.features) {
        alert('Please load test data and train the model first');
        return;
    }
    
    try {
        setStatus(elements.exportStatus, 'Making predictions...', 'loading');
        
        const predictions = model.predict(testData.features);
        testPredictions = await predictions.data();
        
        setStatus(elements.exportStatus, `Predictions complete! ${testPredictions.length} samples processed`, 'success');
        
    } catch (error) {
        setStatus(elements.exportStatus, `Prediction error: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

// Export submission CSV
function exportSubmission() {
    if (!testPredictions || !testData) {
        alert('Please make predictions first');
        return;
    }
    
    try {
        const threshold = parseFloat(elements.thresholdSlider.value);
        let csvContent = 'PassengerId,Survived\n';
        
        testData.originalData.forEach((row, index) => {
            const passengerId = row[ID_COLUMN];
            const survival = testPredictions[index] >= threshold ? 1 : 0;
            csvContent += `${passengerId},${survival}\n`;
        });
        
        downloadCSV(csvContent, 'titanic_submission.csv');
        setStatus(elements.exportStatus, 'Submission CSV exported!', 'success');
        
    } catch (error) {
        setStatus(elements.exportStatus, `Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

// Export probabilities CSV
function exportProbabilities() {
    if (!testPredictions || !testData) {
        alert('Please make predictions first');
        return;
    }
    
    try {
        let csvContent = 'PassengerId,Survived_Probability\n';
        
        testData.originalData.forEach((row, index) => {
            const passengerId = row[ID_COLUMN];
            const probability = testPredictions[index];
            csvContent += `${passengerId},${probability.toFixed(6)}\n`;
        });
        
        downloadCSV(csvContent, 'titanic_probabilities.csv');
        setStatus(elements.exportStatus, 'Probabilities CSV exported!', 'success');
        
    } catch (error) {
        setStatus(elements.exportStatus, `Export error: ${error.message}`, 'error');
        console.error('Export error:', error);
    }
}

// Save the model
async function saveModel() {
    if (!model) {
        alert('Please create and train the model first');
        return;
    }
    
    try {
        setStatus(elements.exportStatus, 'Saving model...', 'loading');
        
        await model.save('downloads://titanic-tfjs-model');
        setStatus(elements.exportStatus, 'Model saved successfully!', 'success');
        
    } catch (error) {
        setStatus(elements.exportStatus, `Model save error: ${error.message}`, 'error');
        console.error('Model save error:', error);
    }
}

// Utility function to download CSV
function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Initialize the app
console.log('Titanic Binary Classifier initialized');
console.log('To reuse with other datasets, update the schema variables in app.js');