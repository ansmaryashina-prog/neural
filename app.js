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

// Initialize application when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    console.log('Initializing Student Performance Analyzer...');
    
    // Initialize all event listeners
    initializeEventListeners();
    
    // Set initial status messages
    setStatus('dataStatus', 'Please upload a CSV file to begin', 'info');
    setStatus('edaStatus', 'Load data first to perform EDA', 'info');
    setStatus('preprocessStatus', 'Run EDA first to preprocess data', 'info');
    setStatus('modelStatus', 'Preprocess data first to create model', 'info');
    setStatus('trainingStatus', 'Create model first to begin training', 'info');
    setStatus('evaluationStatus', 'Train model first to evaluate performance', 'info');
    setStatus('featureStatus', 'Evaluate model first to analyze features', 'info');
    setStatus('demoStatus', 'Click to run complete workflow demonstration', 'info');
}

function initializeEventListeners() {
    // Main workflow buttons
    document.getElementById('loadDataBtn').addEventListener('click', loadData);
    document.getElementById('edaBtn').addEventListener('click', performEDA);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    document.getElementById('trainModelBtn').addEventListener('click', trainModel);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('featureImportanceBtn').addEventListener('click', analyzeFeatureImportance);
    document.getElementById('demoBtn').addEventListener('click', demonstratePrototype);
    
    // Practical examples buttons
    document.getElementById('predictRiskBtn').addEventListener('click', () => predictCaseStudy('risk'));
    document.getElementById('predictAverageBtn').addEventListener('click', () => predictCaseStudy('average'));
    document.getElementById('predictHighBtn').addEventListener('click', () => predictCaseStudy('high'));
    document.getElementById('predictCustomBtn').addEventListener('click', predictCustomStudent);
    
    console.log('All event listeners initialized');
}

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

// Set status message
function setStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.className = `status ${type}`;
    }
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
        setStatus('dataStatus', 'Loading student data...', 'loading');
        
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
            const availableColumns = Object.keys(studentData[0]);
            const numericColumns = availableColumns.filter(col => {
                const sample = studentData[0][col];
                return typeof sample === 'number' && !isNaN(sample);
            });
            console.warn(`Target column '${TARGET_COLUMN}' not found. Available numeric columns:`, numericColumns);
        }
        
        setStatus('dataStatus', 
                 `Successfully loaded ${studentData.length} student records with ${Object.keys(studentData[0]).length} features`, 
                 'success');
        
        // Enable EDA button
        setStatus('edaStatus', 'Ready to perform EDA', 'success');
                 
    } catch (error) {
        setStatus('dataStatus', `Error loading data: ${error.message}`, 'error');
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

// Perform Exploratory Data Analysis
async function performEDA() {
    if (!studentData) {
        alert('Please load data first');
        return;
    }
    
    try {
        setStatus('edaStatus', 'Performing Exploratory Data Analysis...', 'loading');
        
        // Clear previous visualizations
        const vizArea = document.getElementById('edaVisualizations');
        if (vizArea) {
            vizArea.innerHTML = '<p>Creating visualizations...</p>';
        }
        
        console.log('=== STARTING EDA ===');
        console.log('Total records:', studentData.length);
        
        // Check target variable
        const targetValues = getValidNumericValues(TARGET_COLUMN);
        console.log('Valid target values:', targetValues.length);
        
        if (targetValues.length === 0) {
            const availableColumns = Object.keys(studentData[0]);
            const numericColumns = availableColumns.filter(col => {
                const values = getValidNumericValues(col);
                return values.length > 0;
            });
            
            throw new Error(
                `No valid numeric values in target column '${TARGET_COLUMN}'. ` +
                `Available numeric columns: ${numericColumns.join(', ')}`
            );
        }
        
        // Create EDA Visualizations
        await createEDAVisualizations();
        
        setStatus('edaStatus', 
                 `EDA completed! Analyzed ${studentData.length} student records`, 
                 'success');
        
        // Enable preprocessing button
        setStatus('preprocessStatus', 'Ready to preprocess data', 'success');
        
    } catch (error) {
        setStatus('edaStatus', `EDA error: ${error.message}`, 'error');
        console.error('EDA error:', error);
    }
}

// SIMPLIFIED EDA Visualizations - FIXED VERSION
async function createEDAVisualizations() {
    if (!studentData || !tfvis) {
        console.error('No data or tfvis available');
        return;
    }
    
    console.log('Creating EDA visualizations...');
    
    try {
        // 1. Target variable distribution - SIMPLIFIED
        const targetValues = getValidNumericValues(TARGET_COLUMN);
        
        if (targetValues.length > 0) {
            console.log('Rendering target distribution with', targetValues.length, 'values');
            
            const surface1 = { name: 'Final Grade Distribution', tab: 'Data Analysis' };
            await tfvis.render.histogram(surface1, targetValues, {
                xLabel: 'Final Grade (G3)',
                yLabel: 'Number of Students',
                maxBins: 10
            });
        }

        // 2. Basic feature distributions
        const numericFeatures = getNumericFeatures(studentData[0])
            .filter(f => f !== TARGET_COLUMN)
            .slice(0, 3);
        
        for (const feature of numericFeatures) {
            const featureValues = getValidNumericValues(feature);
            if (featureValues.length > 0) {
                const surface = { name: `Distribution: ${feature}`, tab: 'Data Analysis' };
                await tfvis.render.histogram(surface, featureValues, {
                    xLabel: feature,
                    yLabel: 'Frequency',
                    maxBins: 8
                });
            }
        }

        console.log('All EDA visualizations completed successfully');

    } catch (error) {
        console.error('Visualization error:', error);
        throw error;
    }
}

// Helper function to get valid numeric values from a column
function getValidNumericValues(columnName) {
    if (!studentData || studentData.length === 0) return [];
    
    return studentData
        .map(row => row[columnName])
        .filter(value => isValidNumber(value));
}

// Check if a value is a valid number
function isValidNumber(value) {
    return value !== null && 
           value !== undefined && 
           typeof value === 'number' && 
           !isNaN(value);
}

// Utility functions
function getNumericFeatures(sampleRow) {
    return Object.keys(sampleRow).filter(key => {
        const value = sampleRow[key];
        return isValidNumber(value);
    });
}

function getCategoricalFeatures(sampleRow) {
    return Object.keys(sampleRow).filter
