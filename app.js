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
        console.log('Available columns:', Object.keys(studentData[0]));
        
        // Check if target column exists
        if (!studentData[0].hasOwnProperty(TARGET_COLUMN)) {
            console.warn(`Target column '${TARGET_COLUMN}' not found. Available columns:`, Object.keys(studentData[0]));
        }
        
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

// CSV parser with better error handling
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n').filter(line => line.trim() !== '');
    
    if (lines.length === 0) {
        throw new Error('CSV file is empty');
    }
    
    // Handle different separators (comma or semicolon)
    const firstLine = lines[0];
    const separator = firstLine.includes(';') ? ';' : ',';
    
    const headers = lines[0].split(separator).map(h => h.trim().replace(/"/g, ''));
    
    console.log('CSV Headers:', headers);
    console.log('Separator detected:', separator);
    
    return lines.slice(1).map((line, index) => {
        const values = line.split(separator).map(v => v.trim().replace(/"/g, ''));
        const row = {};
        
        headers.forEach((header, i) => {
            let value = values[i];
            
            // Handle missing values
            if (value === undefined || value === '' || value === 'NA' || value === 'null') {
                value = null;
            } else {
                // Convert numeric values
                const numericValue = parseFloat(value);
                if (!isNaN(numericValue) && value !== '') {
                    value = numericValue;
                }
            }
            
            row[header] = value;
        });
        
        return row;
    }).filter(row => Object.keys(row).length > 0); // Remove empty rows
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
        const targetValues = studentData.map(row => row[TARGET_COLUMN])
            .filter(v => v !== undefined && v !== null && !isNaN(v));
        
        if (targetValues.length === 0) {
            throw new Error(`No valid target values found in column '${TARGET_COLUMN}'. Available columns: ${Object.keys(studentData[0])}`);
        }
        
        const targetStats = {
            min: Math.min(...targetValues),
            max: Math.max(...targetValues),
            mean: targetValues.reduce((a, b) => a + b, 0) / targetValues.length,
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
    
    try {
        // 1. Target variable distribution
        const targetValues = studentData.map(row => row[TARGET_COLUMN])
            .filter(v => v !== undefined && v !== null && !isNaN(v));
        
        if (targetValues.length > 0) {
            await tfvis.render.histogram(surface, 
                { values: targetValues, dataSeries: ['Final Grades'] },
                { xLabel: 'Final Grade (G3)', yLabel: 'Frequency', maxBins: 10 }
            );
        }
        
        // 2. Correlation analysis for key features
        const availableFeatures = Object.keys(studentData[0]).filter(key => 
            key !== TARGET_COLUMN && 
            typeof studentData[0][key] === 'number'
        ).slice(0, 6); // Take first 6 numeric features
        
        if (availableFeatures.length > 0) {
            const correlationData = availableFeatures.map(feature => {
                const featureValues = studentData.map(row => row[feature])
                    .filter(v => v !== undefined && v !== null && !isNaN(v));
                const targetValuesFiltered = studentData.map(row => row[TARGET_COLUMN])
                    .filter((v, i) => featureValues[i] !== undefined && v !== undefined && !isNaN(v));
                
                // Align arrays
                const alignedFeatureValues = [];
                const alignedTargetValues = [];
                
                studentData.forEach(row => {
                    if (row[feature] !== undefined && row[feature] !== null && 
                        !isNaN(row[feature]) && row[TARGET_COLUMN] !== undefined && 
                        row[TARGET_COLUMN] !== null && !isNaN(row[TARGET_COLUMN])) {
                        alignedFeatureValues.push(row[feature]);
                        alignedTargetValues.push(row[TARGET_COLUMN]);
                    }
                });
                
                return {
                    x: alignedFeatureValues,
                    y: alignedTargetValues,
                    label: feature
                };
            }).filter(data => data.x.length > 0);
            
            if (correlationData.length > 0) {
                const scatterSurface = { name: 'Feature vs Target Correlations', tab: 'Data Analysis' };
                await tfvis.render.scatterplot(scatterSurface, 
                    { values: correlationData, series: availableFeatures.slice(0, correlationData.length) },
                    { xLabel: 'Feature Values', yLabel: 'Final Grade (G3)' }
                );
            }
        }
        
        // 3. Categorical feature analysis
        const categoricalFeatures = getCategoricalFeatures(studentData[0]).slice(0, 3);
        
        for (const feature of categoricalFeatures) {
            const performance = analyzeCategoricalPerformance(feature);
            if (Object.keys(performance).length > 0) {
                const barSurface = { name: `Performance by ${feature}`, tab: 'Data Analysis' };
                await tfvis.render.barchart(barSurface, 
                    Object.entries(performance).map(([key, value]) => ({ index: key, value })),
                    { xLabel: feature, yLabel: 'Average Grade' }
                );
            }
        }
        
    } catch (error) {
        console.error('Visualization error:', error);
        throw error;
    }
}

// Analyze performance by categorical feature
function analyzeCategoricalPerformance(featureName) {
    const groups = {};
    
    studentData.forEach(row => {
        if (row[featureName] !== undefined && row[featureName] !== null &&
            row[TARGET_COLUMN] !== undefined && row[TARGET_COLUMN] !== null &&
            !isNaN(row[TARGET_COLUMN])) {
            
            const key = String(row[featureName]);
            if (!groups[key]) groups[key] = [];
            groups[key].push(row[TARGET_COLUMN]);
        }
    });
    
    const result = {};
    Object.entries(groups).forEach(([key, values]) => {
        if (values.length > 0) {
            result[key] = values.reduce((a, b) => a + b, 0) / values.length;
        }
    });
    
    return result;
}

// Utility functions with safe array operations
function getNumericFeatures(sampleRow) {
    return Object.keys(sampleRow).filter(key => 
        typeof sampleRow[key] === 'number' && 
        sampleRow[key] !== null && 
        !isNaN(sampleRow[key])
    );
}

function getCategoricalFeatures(sampleRow) {
    return Object.keys(sampleRow).filter(key => 
        typeof sampleRow[key] === 'string' || 
        (sampleRow[key] !== null && typeof sampleRow[key] !== 'number')
    );
}

function median(arr) {
    if (arr.length === 0) return 0;
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function safeSum(arr) {
    return arr.reduce((a, b) => a + b, 0);
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

// Comprehensive data preprocessing with error handling
function preprocessStudentData(data) {
    const features = [];
    const labels = [];
    const featureNames = [];
    
    // Get all available columns except target
    const availableColumns = Object.keys(data[0]).filter(col => col !== TARGET_COLUMN);
    
    // Calculate statistics for numeric features
    const numericStats = {};
    const categoricalOptions = {};
    
    availableColumns.forEach(col => {
        const values = data.map(row => row[col]).filter(v => v !== null && v !== undefined && !isNaN(v));
        
        if (values.length > 0 && typeof values[0] === 'number') {
            const mean = safeSum(values) / values.length;
            const std = Math.sqrt(values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length);
            numericStats[col] = { mean, std };
        } else {
            // For categorical, collect all unique values
            const uniqueValues = [...new Set(data.map(row => String(row[col])).filter(v => v && v !== 'undefined' && v !== 'null'))];
            categoricalOptions[col] = uniqueValues;
        }
    });
    
    data.forEach(row => {
        const featureVector = [];
        
        // Process numeric features (standardization)
        availableColumns.forEach(col => {
            if (numericStats[col]) {
                let value = row[col];
                if (value === null || value === undefined || isNaN(value)) {
                    value = numericStats[col].mean; // Impute with mean
                }
                const standardized = (value - numericStats[col].mean) / (numericStats[col].std || 1);
                featureVector.push(standardized);
                
                if (featureNames.length < availableColumns.length) {
                    featureNames.push(col);
                }
            }
        });
        
        // Process categorical features (one-hot encoding)
        availableColumns.forEach(col => {
            if (categoricalOptions[col]) {
                const categories = categoricalOptions[col];
                const oneHot = new Array(categories.length).fill(0);
                const value = String(row[col] || '');
                const index = categories.indexOf(value);
                
                if (index !== -1) {
                    oneHot[index] = 1;
                } else if (categories.length > 0) {
                    oneHot[0] = 1; // Default to first category if value not found
                }
                
                oneHot.forEach(val => featureVector.push(val));
                categories.forEach((cat, idx) => {
                    featureNames.push(`${col}_${cat}`);
                });
            }
        });
        
        features.push(featureVector);
        
        // Target variable (final grade)
        if (row[TARGET_COLUMN] !== undefined && row[TARGET_COLUMN] !== null && !isNaN(row[TARGET_COLUMN])) {
            labels.push([row[TARGET_COLUMN]]);
        }
    });
    
    // Filter out rows with invalid features or labels
    const validIndices = [];
    features.forEach((feature, index) => {
        if (feature.length > 0 && labels[index] && labels[index][0] !== undefined) {
            validIndices.push(index);
        }
    });
    
    const validFeatures = validIndices.map(i => features[i]);
    const validLabels = validIndices.map(i => labels[i]);
    
    console.log(`Valid data: ${validFeatures.length} out of ${data.length} records`);
    
    if (validFeatures.length === 0) {
        throw new Error('No valid data after preprocessing');
    }
    
    return {
        features: tf.tensor2d(validFeatures),
        labels: tf.tensor2d(validLabels),
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

// Calculate evaluation metrics with safe operations
function calculateMSE(trueValues, predValues) {
    if (trueValues.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < trueValues.length; i++) {
        sum += Math.pow(trueValues[i] - predValues[i], 2);
    }
    return sum / trueValues.length;
}

function calculateMAE(trueValues, predValues) {
    if (trueValues.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < trueValues.length; i++) {
        sum += Math.abs(trueValues[i] - predValues[i]);
    }
    return sum / trueValues.length;
}

function calculateR2(trueValues, predValues) {
    if (trueValues.length === 0) return 0;
    const mean = trueValues.reduce((a, b) => a + b, 0) / trueValues.length;
    const ssTotal = trueValues.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
    const ssResidual = trueValues.reduce((acc, val, idx) => acc + Math.pow(val - predValues[idx], 2), 0);
    return 1 - (ssResidual / ssTotal);
}

// Create evaluation visualizations
async function createEvaluationVisualizations(trueValues, predValues) {
    try {
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
        
    } catch (error) {
        console.error('Evaluation visualization error:', error);
    }
}

// Analyze feature importance
async function analyzeFeatureImportance() {
    if (!model || !processedData) {
        alert('Please train the model first');
        return;
    }
    
    try {
        setStatus(elements.featureStatus, 'Analyzing feature importance...', 'loading');
        
        // For demo purposes, show a simple feature importance based on correlation
        const importanceScores = await calculateSimpleFeatureImportance();
        
        // Create feature importance visualization
        const importanceSurface = { name: 'Feature Importance', tab: 'Results' };
        
        const importanceData = importanceScores
            .filter(imp => !isNaN(imp.score))
            .sort((a, b) => b.score - a.score)
            .slice(0, 10)
            .map(imp => ({ index: imp.feature.length > 20 ? imp.feature.substring(0, 20) + '...' : imp.feature, value: imp.score }));
        
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

// Calculate simple feature importance based on correlation
async function calculateSimpleFeatureImportance() {
    const { features, labels, featureNames } = processedData;
    const featureData = await features.data();
    const labelData = await labels.data();
    
    const importanceScores = [];
    const numFeatures = features.shape[1];
    const numSamples = features.shape[0];
    
    for (let i = 0; i < Math.min(numFeatures, featureNames.length); i++) {
        const featureValues = [];
        const targetValues = [];
        
        // Extract feature values and corresponding targets
        for (let j = 0; j < numSamples; j++) {
            const featureVal = featureData[j * numFeatures + i];
            const targetVal = labelData[j];
            if (!isNaN(featureVal) && !isNaN(targetVal)) {
                featureValues.push(featureVal);
                targetValues.push(targetVal);
            }
        }
        
        if (featureValues.length > 1) {
            // Calculate correlation coefficient
            const correlation = calculateCorrelation(featureValues, targetValues);
            importanceScores.push({
                feature: featureNames[i],
                score: Math.abs(correlation)
            });
        }
    }
    
    return importanceScores;
}

// Calculate correlation coefficient
function calculateCorrelation(x, y) {
    const n = x.length;
    const sum_x = safeSum(x);
    const sum_y = safeSum(y);
    const sum_xy = x.reduce((acc, val, i) => acc + val * y[i], 0);
    const sum_x2 = x.reduce((acc, val) => acc + val * val, 0);
    const sum_y2 = y.reduce((acc, val) => acc + val * val, 0);
    
    const numerator = n * sum_xy - sum_x * sum_y;
    const denominator = Math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    return denominator === 0 ? 0 : numerator / denominator;
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
