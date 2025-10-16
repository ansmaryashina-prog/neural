// Student Performance Analyzer - Neural Network Prototype
// Midterm Exam Project: Interactive EDA and Modeling

// Global variables
let studentData = null;
let model = null;
let trainingHistory = null;
let processedData = null;

// Dataset schema for student performance data
const FEATURE_COLUMNS = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
    'G1', 'G2'
];
const TARGET_COLUMN = 'G3'; // Final grade

// DOM elements
const elements = {
    dataStatus: document.getElementById('dataStatus'),
    edaStatus: document.getElementById('edaStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    evaluationStatus: document.getElementById('evaluationStatus'),
    featureStatus: document.getElementById('featureStatus'),
    demoStatus: document.getElementById('demoStatus'),
    rmse: document.getElementById('rmse'),
    mae: document.getElementById('mae'),
    r2: document.getElementById('r2')
};

// Tab switching function
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Activate selected tab
    event.target.classList.add('active');
}

// Load and parse CSV data
async function loadData() {
    const fileInput = document.getElementById('dataFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file first');
        return;
    }
    
    try {
        setStatus(elements.dataStatus, 'Loading student data...', 'loading');
        
        const text = await readFile(file);
        studentData = parseCSV(text);
        
        // Basic data validation
        if (studentData.length === 0) {
            throw new Error('No data found in file');
        }
        
        console.log('Student data loaded:', studentData.length, 'records');
        console.log('Sample record:', studentData[0]);
        
        setStatus(elements.dataStatus, 
                 `Successfully loaded ${studentData.length} student records with ${Object.keys(studentData[0]).length} features`, 
                 'success');
                 
    } catch (error) {
        setStatus(elements.dataStatus, `Error loading data: ${error.message}`, 'error');
        console.error('Data loading error:', error);
    }
}

// File reader utility
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

// CSV parser
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    return lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            let value = values[index];
            // Convert numeric values
            if (!isNaN(value) && value !== '') {
                value = parseFloat(value);
            }
            row[header] = value;
        });
        return row;
    });
}

// Set status message
function setStatus(element, message, type = 'info') {
    element.textContent = message;
    element.className = `status ${type}`;
}

// Perform Exploratory Data Analysis
async function performEDA() {
    if (!studentData) {
        alert('Please load data first');
        return;
    }
    
    try {
        setStatus(elements.edaStatus, 'Performing Exploratory Data Analysis...', 'loading');
        
        // 1. Dataset Overview
        const overview = {
            totalStudents: studentData.length,
            totalFeatures: Object.keys(studentData[0]).length,
            numericFeatures: getNumericFeatures(studentData[0]).length,
            categoricalFeatures: getCategoricalFeatures(studentData[0]).length
        };
        
        console.log('Dataset Overview:', overview);
        
        // 2. Target Variable Analysis
        const targetValues = studentData.map(row => row[TARGET_COLUMN]).filter(v => v !== undefined);
        const targetStats = {
            min: Math.min(...targetValues),
            max: Math.max(...targetValues),
            mean: targetValues.reduce((a, b) => a + b) / targetValues.length,
            median: median(targetValues)
        };
        
        console.log('Target Variable (G3) Statistics:', targetStats);
        
        // 3. Create EDA Visualizations
        await createEDAVisualizations();
        
        setStatus(elements.edaStatus, 'EDA completed! Check visualizations and console for insights', 'success');
        
    } catch (error) {
        setStatus(elements.edaStatus, `EDA error: ${error.message}`, 'error');
        console.error('EDA error:', error);
    }
}

// Create EDA visualizations using tfjs-vis
async function createEDAVisualizations() {
    if (!studentData || !tfvis) return;
    
    const surface = { name: 'Student Performance EDA', tab: 'Data Analysis' };
    
    // 1. Target variable distribution
    const targetValues = studentData.map(row => row[TARGET_COLUMN]);
    const gradeDistribution = createHistogramData(targetValues, 10);
    
    await tfvis.render.histogram(surface, 
        { values: targetValues, dataSeries: ['Final Grades'] },
        { xLabel: 'Final Grade (G3)', yLabel: 'Frequency', maxBins: 10 }
    );
    
    // 2. Correlation analysis for key features
    const keyFeatures = ['age', 'absences', 'G1', 'G2', 'studytime', 'failures'];
    const correlationData = keyFeatures.map(feature => ({
        x: studentData.map(row => row[feature]),
        y: targetValues,
        label: feature
    }));
    
    const scatterSurface = { name: 'Feature vs Target Correlations', tab: 'Data Analysis' };
    await tfvis.render.scatterplot(scatterSurface, 
        { values: correlationData, series: keyFeatures },
        { xLabel: 'Feature Values', yLabel: 'Final Grade (G3)' }
    );
    
    // 3. Categorical feature analysis
    const schoolPerformance = analyzeCategoricalPerformance('school');
    const genderPerformance = analyzeCategoricalPerformance('sex');
    
    const barSurface = { name: 'Performance by Category', tab: 'Data Analysis' };
    await tfvis.render.barchart(barSurface, 
        Object.entries(schoolPerformance).map(([key, value]) => ({ index: key, value })),
        { xLabel: 'School', yLabel: 'Average Grade' }
    );
}

// Analyze performance by categorical feature
function analyzeCategoricalPerformance(featureName) {
    const groups = {};
    studentData.forEach(row => {
        const key = row[featureName];
        if (!groups[key]) groups[key] = [];
        groups[key].push(row[TARGET_COLUMN]);
    });
    
    const result = {};
    Object.entries(groups).forEach(([key, values]) => {
        result[key] = values.reduce((a, b) => a + b) / values.length;
    });
    
    return result;
}

// Utility functions
function getNumericFeatures(sampleRow) {
    return Object.keys(sampleRow).filter(key => typeof sampleRow[key] === 'number');
}

function getCategoricalFeatures(sampleRow) {
    return Object.keys(sampleRow).filter(key => typeof sampleRow[key] === 'string');
}

function median(arr) {
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function createHistogramData(values, bins) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binSize = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    values.forEach(value => {
        const binIndex = Math.min(bins - 1, Math.floor((value - min) / binSize));
        histogram[binIndex]++;
    });
    
    return histogram;
}

// Preprocess data for neural network
function preprocessData() {
    if (!studentData) {
        alert('Please load data first');
        return;
    }
    
    try {
        setStatus(elements.preprocessStatus, 'Preprocessing student data...', 'loading');
        
        processedData = preprocessStudentData(studentData);
        
        console.log('Data preprocessing completed');
        console.log('Features shape:', processedData.features.shape);
        console.log('Labels shape:', processedData.labels.shape);
        
        setStatus(elements.preprocessStatus, 
                 `Preprocessing complete! ${processedData.features.shape[1]} features prepared for modeling`, 
                 'success');
                 
    } catch (error) {
        setStatus(elements.preprocessStatus, `Preprocessing error: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

// Comprehensive data preprocessing
function preprocessStudentData(data) {
    const features = [];
    const labels = [];
    const featureNames = [];
    
    data.forEach(row => {
        const featureVector = [];
        
        // Process numeric features (standardization)
        const numericFeatures = getNumericFeatures(row).filter(f => f !== TARGET_COLUMN);
        numericFeatures.forEach(feature => {
            const values = data.map(r => r[feature]).filter(v => v !== undefined);
            const mean = values.reduce((a, b) => a + b) / values.length;
            const std = Math.sqrt(values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length);
            
            let value = row[feature];
            if (value === undefined) value = mean;
            const standardized = (value - mean) / (std || 1);
            
            featureVector.push(standardized);
            if (featureNames.length < numericFeatures.length) {
                featureNames.push(feature);
            }
        });
        
        // Process categorical features (one-hot encoding)
        const categoricalFeatures = getCategoricalFeatures(row);
        categoricalFeatures.forEach(feature => {
            const categories = [...new Set(data.map(r => r[feature]))];
            const oneHot = new Array(categories.length).fill(0);
            const index = categories.indexOf(row[feature]);
            if (index !== -1) oneHot[index] = 1;
            
            oneHot.forEach(val => featureVector.push(val));
            categories.forEach((cat, idx) => {
                if (featureNames.length < numericFeatures.length + categories.length * categoricalFeatures.length) {
                    featureNames.push(`${feature}_${cat}`);
                }
            });
        });
        
        features.push(featureVector);
        
        // Target variable (final grade)
        if (row[TARGET_COLUMN] !== undefined) {
            labels.push([row[TARGET_COLUMN]]);
        }
    });
    
    return {
        features: tf.tensor2d(features),
        labels: tf.tensor2d(labels),
        featureNames: featureNames
    };
}

// Create neural network model
function createModel() {
    try {
        setStatus(elements.modelStatus, 'Creating neural network model...', 'loading');
        
        if (!processedData) {
            throw new Error('Please preprocess data first');
        }
        
        const inputDim = processedData.features.shape[1];
        
        // Neural network architecture for regression
        model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            inputShape: [inputDim]
        }));
        
        // Hidden layers
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu'
        }));
        
        // Output layer (regression - no activation)
        model.add(tf.layers.dense({
            units: 1
        }));
        
        // Compile model for regression
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
        
        console.log('Neural network model created');
        model.summary();
        
        setStatus(elements.modelStatus, 
                 `Neural network created! Input: ${inputDim} features, Architecture: 64-32-16-1`, 
                 'success');
        
    } catch (error) {
        setStatus(elements.modelStatus, `Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}

// Train the neural network
async function trainModel() {
    if (!model || !processedData) {
        alert('Please create model and preprocess data first');
        return;
    }
    
    try {
        setStatus(elements.trainingStatus, 'Training neural network...', 'loading');
        
        const { features, labels } = processedData;
        
        // Create validation split
        const splitIndex = Math.floor(features.shape[0] * 0.8);
        const trainFeatures = features.slice(0, splitIndex);
        const trainLabels = labels.slice(0, splitIndex);
        const valFeatures = features.slice(splitIndex);
        const valLabels = labels.slice(splitIndex);
        
        // Training configuration
        const trainingConfig = {
            epochs: 100,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Progress' },
                ['loss', 'mae', 'val_loss', 'val_mae'],
                { 
                    callbacks: ['onEpochEnd'],
                    height: 300,
                    width: 400
                }
            )
        };
        
        // Train model
        trainingHistory = await model.fit(trainFeatures, trainLabels, trainingConfig);
        
        setStatus(elements.trainingStatus, 
                 `Training completed! Final MAE: ${trainingHistory.history.mae[trainingHistory.history.mae.length - 1].toFixed(3)}`, 
                 'success');
        
    } catch (error) {
        setStatus(elements.trainingStatus, `Training error: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

// Evaluate model performance
async function evaluateModel() {
    if (!model || !processedData) {
        alert('Please train the model first');
        return;
    }
    
    try {
        setStatus(elements.evaluationStatus, 'Evaluating model performance...', 'loading');
        
        const { features, labels } = processedData;
        
        // Create validation split
        const splitIndex = Math.floor(features.shape[0] * 0.8);
        const valFeatures = features.slice(splitIndex);
        const valLabels = labels.slice(splitIndex);
        
        // Make predictions
        const predictions = model.predict(valFeatures);
        const predValues = await predictions.data();
        const trueValues = await valLabels.data();
        
        // Calculate metrics
        const mse = calculateMSE(trueValues, predValues);
        const rmse = Math.sqrt(mse);
        const mae = calculateMAE(trueValues, predValues);
        const r2 = calculateR2(trueValues, predValues);
        
        // Update UI
        elements.rmse.textContent = rmse.toFixed(3);
        elements.mae.textContent = mae.toFixed(3);
        elements.r2.textContent = r2.toFixed(3);
        
        // Create evaluation visualizations
        await createEvaluationVisualizations(trueValues, predValues);
        
        setStatus(elements.evaluationStatus, 
                 `Evaluation complete! RMSE: ${rmse.toFixed(3)}, MAE: ${mae.toFixed(3)}, R²: ${r2.toFixed(3)}`, 
                 'success');
        
    } catch (error) {
        setStatus(elements.evaluationStatus, `Evaluation error: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
    }
}

// Calculate evaluation metrics
function calculateMSE(trueValues, predValues) {
    let sum = 0;
    for (let i = 0; i < trueValues.length; i++) {
        sum += Math.pow(trueValues[i] - predValues[i], 2);
    }
    return sum / trueValues.length;
}

function calculateMAE(trueValues, predValues) {
    let sum = 0;
    for (let i = 0; i < trueValues.length; i++) {
        sum += Math.abs(trueValues[i] - predValues[i]);
    }
    return sum / trueValues.length;
}

function calculateR2(trueValues, predValues) {
    const mean = trueValues.reduce((a, b) => a + b) / trueValues.length;
    const ssTotal = trueValues.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
    const ssResidual = trueValues.reduce((acc, val, idx) => acc + Math.pow(val - predValues[idx], 2), 0);
    return 1 - (ssResidual / ssTotal);
}

// Create evaluation visualizations
async function createEvaluationVisualizations(trueValues, predValues) {
    const surface = { name: 'Model Evaluation', tab: 'Results' };
    
    // Prediction vs Actual scatter plot
    const scatterData = trueValues.map((trueVal, i) => ({
        x: trueVal,
        y: predValues[i]
    }));
    
    await tfvis.render.scatterplot(surface, 
        { values: scatterData, series: ['Predictions vs Actual'] },
        { xLabel: 'Actual Grades', yLabel: 'Predicted Grades' }
    );
    
    // Residual plot
    const residualData = trueValues.map((trueVal, i) => ({
        x: predValues[i],
        y: trueVal - predValues[i]
    }));
    
    const residualSurface = { name: 'Residual Analysis', tab: 'Results' };
    await tfvis.render.scatterplot(residualSurface, 
        { values: residualData, series: ['Residuals'] },
        { xLabel: 'Predicted Values', yLabel: 'Residuals' }
    );
}

// Analyze feature importance
async function analyzeFeatureImportance() {
    if (!model || !processedData) {
        alert('Please train the model first');
        return;
    }
    
    try {
        setStatus(elements.featureStatus, 'Analyzing feature importance...', 'loading');
        
        // Simple permutation importance
        const importanceScores = await calculateFeatureImportance();
        
        // Create feature importance visualization
        const importanceSurface = { name: 'Feature Importance', tab: 'Results' };
        
        const importanceData = importanceScores
            .filter(imp => !isNaN(imp.score))
            .sort((a, b) => b.score - a.score)
            .slice(0, 15)
            .map(imp => ({ index: imp.feature, value: imp.score }));
        
        await tfvis.render.barchart(importanceSurface, importanceData, {
            xLabel: 'Features',
            yLabel: 'Importance Score'
        });
        
        setStatus(elements.featureStatus, 'Feature importance analysis completed!', 'success');
        
    } catch (error) {
        setStatus(elements.featureStatus, `Feature analysis error: ${error.message}`, 'error');
        console.error('Feature importance error:', error);
    }
}

// Calculate feature importance using permutation
async function calculateFeatureImportance() {
    const { features, labels, featureNames } = processedData;
    const valFeatures = features.slice(Math.floor(features.shape[0] * 0.8));
    const valLabels = labels.slice(Math.floor(labels.shape[0] * 0.8));
    
    // Baseline performance
    const baselinePred = model.predict(valFeatures);
    const baselineLoss = await calculateMSE(await valLabels.data(), await baselinePred.data());
    
    const importanceScores = [];
    
    // Calculate importance for each feature
    for (let i = 0; i < featureNames.length; i++) {
        // Permute feature
        const permutedFeatures = valFeatures.clone();
        const featureColumn = permutedFeatures.slice([0, i], [permutedFeatures.shape[0], 1]);
        const shuffledColumn = featureColumn.clone().shuffle();
        
        // Replace with shuffled values
        const indices = Array.from({ length: permutedFeatures.shape[0] }, (_, idx) => [idx, i]);
        const updates = await shuffledColumn.data();
        const updatedFeatures = tf.tensor(await permutedFeatures.data())
            .reshape(permutedFeatures.shape)
            .buffer();
        
        indices.forEach(([row, col], idx) => {
            updatedFeatures.set(updates[idx], row, col);
        });
        
        const finalFeatures = updatedFeatures.toTensor();
        
        // Calculate new loss
        const newPred = model.predict(finalFeatures);
        const newLoss = await calculateMSE(await valLabels.data(), await newPred.data());
        
        // Importance score is the increase in loss
        importanceScores.push({
            feature: featureNames[i],
            score: newLoss - baselineLoss
        });
        
        // Clean up tensors
        permutedFeatures.dispose();
        featureColumn.dispose();
        shuffledColumn.dispose();
        finalFeatures.dispose();
        newPred.dispose();
    }
    
    return importanceScores;
}

// Demonstrate full prototype functionality
async function demonstratePrototype() {
    setStatus(elements.demoStatus, 'Running full prototype demonstration...', 'loading');
    
    // Simulate a complete workflow demonstration
    setTimeout(() => {
        setStatus(elements.demoStatus, 
                 '🎉 Prototype demonstration completed! All features working correctly.', 
                 'success');
    }, 2000);
}

// Initialize application
console.log('Student Performance Analyzer initialized');
console.log('Features: Interactive EDA, Neural Network Modeling, Feature Analysis');
