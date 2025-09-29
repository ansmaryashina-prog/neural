// Titanic Binary Classifier using TensorFlow.js
// Improved version with better architecture and feature engineering

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
    auc: document.getElementById('auc'),
    metricsInterpretation: document.getElementById('metricsInterpretation')
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
        
        // Store original data for reference
        if (trainData) {
            trainData.originalData = [...trainData];
        }
        if (testData) {
            testData.originalData = [...testData];
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
    
    // Calculate class distribution
    const classDistribution = calculateClassDistribution(trainData);
    console.log('Class Distribution:', classDistribution);
    
    // Show data preview
    console.table(trainData.slice(0, 5));
    
    // Create visualizations with tfjs-vis
    await createDataVisualizations();
}

// Calculate class distribution
function calculateClassDistribution(data) {
    const survived = data.filter(row => row[TARGET_COLUMN] === 1).length;
    const died = data.filter(row => row[TARGET_COLUMN] === 0).length;
    const total = data.length;
    
    return {
        survived: survived,
        died: died,
        survivalRate: (survived / total * 100).toFixed(1) + '%',
        deathRate: (died / total * 100).toFixed(1) + '%'
    };
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

// Extract title from name
function extractTitle(name) {
    if (!name) return 'Unknown';
    
    const titleMatch = name.match(/\b([A-Za-z]+)\./);
    if (titleMatch) {
        const title = titleMatch[1];
        // Group rare titles
        const rareTitles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'];
        if (rareTitles.includes(title)) return 'Rare';
        if (title === 'Mlle' || title === 'Ms') return 'Miss';
        if (title === 'Mme') return 'Mrs';
        return title;
    }
    return 'Unknown';
}

// Preprocess the data
function preprocessData() {
    if (!trainData) {
        alert('Please load data first');
        return;
    }
    
    try {
        setStatus(elements.preprocessStatus, 'Preprocessing data with feature engineering...', 'loading');
        
        // Process training data
        const processedTrain = processDataset(trainData, true);
        
        // Process test data if available
        let processedTest = null;
        if (testData) {
            processedTest = processDataset(testData, false);
        }
        
        // Update global variables
        trainData = processedTrain;
        testData = processedTest;
        
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
        
        // Handle numerical features with better normalization
        NUMERICAL_FEATURES.forEach(feature => {
            let value = row[feature];
            
            // Impute missing values
            if (value === null || value === undefined) {
                if (feature === 'Age') value = ageMedian;
                else if (feature === 'Fare') value = fareMedian;
                else value = 0;
            }
            
            // Robust standardization
            if (feature === 'Age') {
                value = (value - ageMedian) / 20; // Use fixed scale
            } else if (feature === 'Fare') {
                // Log transform for Fare to handle skewness
                value = Math.log(value + 1);
                value = (value - Math.log(fareMedian + 1)) / 2;
            } else if (feature === 'SibSp' || feature === 'Parch') {
                // Keep small integers as is
                value = value / 5; // Normalize to 0-1 range
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
        
        // Add engineered features
        const addFamilySize = document.getElementById('addFamilySize').checked;
        const addIsAlone = document.getElementById('addIsAlone').checked;
        const addTitle = document.getElementById('addTitle').checked;
        const addFarePerPerson = document.getElementById('addFarePerPerson').checked;
        
        if (addFamilySize) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            featureVector.push(familySize / 8); // Normalize
            featureNames.add('FamilySize');
        }
        
        if (addIsAlone) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            featureVector.push(isAlone);
            featureNames.add('IsAlone');
        }
        
        if (addTitle && row.Name) {
            const title = extractTitle(row.Name);
            const titleCategories = ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'];
            const titleOneHot = new Array(titleCategories.length).fill(0);
            const titleIndex = titleCategories.indexOf(title);
            if (titleIndex !== -1) {
                titleOneHot[titleIndex] = 1;
            } else {
                titleOneHot[titleCategories.length - 1] = 1; // Put in Rare
            }
            
            titleOneHot.forEach((val, idx) => {
                featureVector.push(val);
                featureNames.add(`Title_${titleCategories[idx]}`);
            });
        }
        
        if (addFarePerPerson && row.Fare) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const farePerPerson = row.Fare / familySize;
            const normalizedFarePP = Math.log(farePerPerson + 1) / 4; // Normalized log fare
            featureVector.push(normalizedFarePP);
            featureNames.add('FarePerPerson');
        }
        
        // Add age groups
        if (row.Age !== null && row.Age !== undefined) {
            const age = row.Age;
            const ageGroup = age < 12 ? 0 : age < 18 ? 1 : age < 60 ? 2 : 3; // Child, Teen, Adult, Senior
            const ageGroupOneHot = [0, 0, 0, 0];
            ageGroupOneHot[ageGroup] = 1;
            
            ageGroupOneHot.forEach((val, idx) => {
                featureVector.push(val);
                featureNames.add(`AgeGroup_${idx}`);
            });
        } else {
            // If age is missing, add zeros for age groups
            [0, 0, 0, 0].forEach((val, idx) => {
                featureVector.push(val);
                featureNames.add(`AgeGroup_${idx}`);
            });
        }
        
        features.push(featureVector);
        
        // Add label for training data
        if (isTraining && row[TARGET_COLUMN] !== null && row[TARGET_COLUMN] !== undefined) {
            labels.push([row[TARGET_COLUMN]]);
        }
    });
    
    return {
        originalData: data,
        processedData: data,
        features: tf.tensor2d(features),
        labels: isTraining ? tf.tensor2d(labels) : null,
        featureNames: Array.from(featureNames)
    };
}

// Utility functions
function median(arr) {
    if (arr.length === 0) return 0;
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mode(arr) {
    if (arr.length === 0) return 'S';
    const frequency = {};
    let max = 0;
    let result = arr[0];
    
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

// Create improved neural network model
function createModel() {
    try {
        setStatus(elements.modelStatus, 'Creating improved model...', 'loading');
        
        if (!trainData || !trainData.features) {
            throw new Error('Please preprocess data first to determine input shape');
        }
        
        const inputDim = trainData.features.shape[1];
        const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
        const units1 = parseInt(document.getElementById('units1').value);
        const units2 = parseInt(document.getElementById('units2').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const dropoutRate = parseFloat(document.getElementById('dropoutRate').value);
        
        console.log(`Creating model with input dimension: ${inputDim}`);
        console.log(`Architecture: ${hiddenLayers} hidden layers, units: [${units1}, ${units2}], LR: ${learningRate}, Dropout: ${dropoutRate}`);
        
        // Create improved sequential model
        model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.dense({
            units: units1,
            activation: 'relu',
            inputShape: [inputDim],
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dropout({ rate: dropoutRate }));
        
        // Additional hidden layers based on configuration
        if (hiddenLayers >= 2) {
            model.add(tf.layers.dense({
                units: units2,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
            }));
            
            model.add(tf.layers.dropout({ rate: dropoutRate / 2 }));
        }
        
        if (hiddenLayers >= 3) {
            model.add(tf.layers.dense({
                units: Math.floor(units2 / 2),
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }));
        }
        
        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile the model with customized optimizer
        model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Print model summary
        model.summary();
        
        setStatus(elements.modelStatus, 
                 `Improved model created! Input: [${inputDim}], Layers: ${hiddenLayers}, Units: [${units1}, ${units2}]`, 
                 'success');
        
    } catch (error) {
        setStatus(elements.modelStatus, `Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

// Train the model with improved configuration
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
        setStatus(elements.trainingStatus, 'Training improved model...', 'loading');
        
        const features = trainData.features;
        const labels = trainData.labels;
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);
        const validationSplit = parseFloat(document.getElementById('validationSplit').value);
        
        // Create validation split
        const splitIndex = Math.floor(features.shape[0] * (1 - validationSplit));
        
        const trainFeatures = features.slice(0, splitIndex);
        const trainLabels = labels.slice(0, splitIndex);
        const valFeatures = features.slice(splitIndex);
        const valLabels = labels.slice(splitIndex);
        
        validationData = { features: valFeatures, labels: valLabels };
        
        // Enhanced training configuration
        const trainingConfig = {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                { 
                    callbacks: ['onEpochEnd'],
                    height: 300,
                    width: 400
                }
            ),
            shuffle: true,
            validationSplit: 0 // We manually split above
        };
        
        console.log(`Training with ${epochs} epochs, batch size ${batchSize}, validation split ${validationSplit}`);
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, trainingConfig);
        
        setStatus(elements.trainingStatus, 
                 `Training completed! Final val_accuracy: ${trainingHistory.history.val_accuracy[trainingHistory.history.val_accuracy.length - 1].toFixed(4)}`, 
                 'success');
        
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
        setStatus(elements.metricsInterpretation, 'Evaluating model...', 'loading');
        
        // Get predictions
        const predictions = model.predict(validationData.features);
        const probs = await predictions.data();
        const trueLabels = await validationData.labels.data();
        
        // Debug information
        console.log('True labels distribution:', {
            'Survived (1)': trueLabels.filter(v => v === 1).length,
            'Died (0)': trueLabels.filter(v => v === 0).length
        });
        
        console.log('Predictions statistics:', {
            min: Math.min(...probs),
            max: Math.max(...probs),
            mean: probs.reduce((a, b) => a + b) / probs.length,
            above_0_5: probs.filter(p => p >= 0.5).length
        });
        
        // Compute ROC curve and AUC
        rocData = computeROC(trueLabels, probs);
        aucScore = computeAUC(rocData);
        
        // Plot ROC curve
        plotROC(rocData, aucScore);
        
        // Update metrics with default threshold
        updateMetrics(0.5);
        
        elements.auc.textContent = aucScore.toFixed(4);
        
        // Provide interpretation
        provideMetricsInterpretation();
        
    } catch (error) {
        console.error('Evaluation error:', error);
        setStatus(elements.metricsInterpretation, `Evaluation error: ${error.message}`, 'error');
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
        
        // Add color coding based on performance
        updateMetricColors(accuracy, precision, recall, f1, aucScore);
        
        // Plot confusion matrix
        plotConfusionMatrix(point);
    }
}

// Update metric colors based on performance
function updateMetricColors(accuracy, precision, recall, f1, auc) {
    const metrics = [
        { element: elements.accuracy, value: accuracy, good: 0.75, excellent: 0.85 },
        { element: elements.precision, value: precision, good: 0.70, excellent: 0.80 },
        { element: elements.recall, value: recall, good: 0.65, excellent: 0.75 },
        { element: elements.f1, value: f1, good: 0.70, excellent: 0.80 },
        { element: elements.auc, value: auc, good: 0.75, excellent: 0.85 }
    ];
    
    metrics.forEach(metric => {
        if (metric.value >= metric.excellent) {
            metric.element.className = 'metric-good';
        } else if (metric.value >= metric.good) {
            metric.element.className = '';
        } else {
            metric.element.className = 'metric-poor';
        }
    });
}

// Provide interpretation of metrics
function provideMetricsInterpretation() {
    const accuracy = parseFloat(elements.accuracy.textContent);
    const precision = parseFloat(elements.precision.textContent);
    const recall = parseFloat(elements.recall.textContent);
    const f1 = parseFloat(elements.f1.textContent);
    const auc = parseFloat(elements.auc.textContent);
    
    let interpretation = '';
    
    if (accuracy > 0.85 && auc > 0.85) {
        interpretation = '🎉 Excellent model! Performance is very strong.';
    } else if (accuracy > 0.80 && auc > 0.80) {
        interpretation = '✅ Good model! Solid performance for Titanic dataset.';
    } else if (accuracy > 0.75 && auc > 0.75) {
        interpretation = '⚠️ Acceptable model. Consider tuning hyperparameters.';
    } else {
        interpretation = '❌ Model needs improvement. Check data preprocessing and model architecture.';
    }
    
    setStatus(elements.metricsInterpretation, interpretation, 'success');
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
console.log('Improved Titanic Binary Classifier initialized');
console.log('Features: Enhanced architecture, feature engineering, and model configuration');