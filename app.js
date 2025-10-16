// Practical Examples and Case Studies
function predictCaseStudy(type) {
    if (!model) {
        alert('Please train the model first');
        return;
    }

    // Define case study profiles based on actual data patterns
    const caseStudies = {
        risk: {
            school: 'GP', sex: 'M', age: 16, address: 'U', famsize: 'GT3', Pstatus: 'T',
            Medu: 2, Fedu: 2, Mjob: 'other', Fjob: 'other', reason: 'home', guardian: 'mother',
            traveltime: 2, studytime: 1, failures: 2, schoolsup: 'no', famsup: 'no',
            paid: 'no', activities: 'no', nursery: 'yes', higher: 'yes', internet: 'no',
            romantic: 'no', famrel: 3, freetime: 3, goout: 4, Dalc: 3, Walc: 4,
            health: 3, absences: 15, G1: 8, G2: 9
        },
        average: {
            school: 'GP', sex: 'F', age: 17, address: 'U', famsize: 'GT3', Pstatus: 'T',
            Medu: 3, Fedu: 3, Mjob: 'services', Fjob: 'other', reason: 'course', guardian: 'father',
            traveltime: 1, studytime: 2, failures: 0, schoolsup: 'yes', famsup: 'yes',
            paid: 'no', activities: 'yes', nursery: 'yes', higher: 'yes', internet: 'yes',
            romantic: 'no', famrel: 4, freetime: 3, goout: 3, Dalc: 1, Walc: 2,
            health: 4, absences: 6, G1: 11, G2: 12
        },
        high: {
            school: 'GP', sex: 'F', age: 16, address: 'U', famsize: 'GT3', Pstatus: 'T',
            Medu: 4, Fedu: 4, Mjob: 'teacher', Fjob: 'teacher', reason: 'course', guardian: 'mother',
            traveltime: 1, studytime: 4, failures: 0, schoolsup: 'no', famsup: 'yes',
            paid: 'yes', activities: 'yes', nursery: 'yes', higher: 'yes', internet: 'yes',
            romantic: 'no', famrel: 5, freetime: 2, goout: 2, Dalc: 1, Walc: 1,
            health: 5, absences: 2, G1: 16, G2: 17
        }
    };

    const profile = caseStudies[type];
    const prediction = predictStudentPerformance(profile);
    
    const predictionElement = document.getElementById(`${type}Prediction`);
    const gradeElement = document.getElementById(`${type}Grade`);
    
    if (predictionElement && gradeElement) {
        gradeElement.textContent = prediction.toFixed(1);
        predictionElement.style.display = 'block';
        
        // Add color coding based on prediction
        if (prediction >= 15) {
            gradeElement.style.color = '#27ae60';
        } else if (prediction >= 10) {
            gradeElement.style.color = '#f39c12';
        } else {
            gradeElement.style.color = '#e74c3c';
        }
    }
}

function predictCustomStudent() {
    if (!model) {
        alert('Please train the model first');
        return;
    }

    // Get values from custom inputs
    const customProfile = {
        school: 'GP', sex: 'M', age: 17, address: 'U', famsize: 'GT3', Pstatus: 'T',
        Medu: 3, Fedu: 3, Mjob: 'other', Fjob: 'other', reason: 'course', guardian: 'mother',
        traveltime: 1, studytime: parseInt(document.getElementById('customStudyTime').value),
        failures: 0, schoolsup: 'no', famsup: 'yes', paid: 'no', activities: 'yes',
        nursery: 'yes', higher: 'yes', internet: 'yes', romantic: 'no', famrel: 4,
        freetime: 3, goout: 3, Dalc: 1, Walc: 2, health: 4,
        absences: parseInt(document.getElementById('customAbsences').value),
        G1: parseInt(document.getElementById('customG1').value),
        G2: parseInt(document.getElementById('customG2').value)
    };

    const prediction = predictStudentPerformance(customProfile);
    
    const predictionElement = document.getElementById('customPrediction');
    const gradeElement = document.getElementById('customGrade');
    
    if (predictionElement && gradeElement) {
        gradeElement.textContent = prediction.toFixed(1);
        predictionElement.style.display = 'block';
        
        // Add color coding based on prediction
        if (prediction >= 15) {
            gradeElement.style.color = '#27ae60';
        } else if (prediction >= 10) {
            gradeElement.style.color = '#f39c12';
        } else {
            gradeElement.style.color = '#e74c3c';
        }
    }
}

function predictStudentPerformance(studentProfile) {
    if (!processedData || !model) {
        console.error('Model or processed data not available');
        return 0;
    }

    try {
        // Convert student profile to feature vector using the same preprocessing
        const featureVector = preprocessSingleStudent(studentProfile);
        const featuresTensor = tf.tensor2d([featureVector]);
        
        // Make prediction
        const prediction = model.predict(featuresTensor);
        const predictionValue = prediction.dataSync()[0];
        
        // Clean up tensors
        featuresTensor.dispose();
        prediction.dispose();
        
        return Math.max(0, Math.min(20, predictionValue)); // Clamp between 0-20
        
    } catch (error) {
        console.error('Prediction error:', error);
        return 0;
    }
}

function preprocessSingleStudent(student) {
    // This should match your existing preprocessing logic
    const featureVector = [];
    
    // Add numeric features (standardized)
    const numericFeatures = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                           'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'];
    
    numericFeatures.forEach(feature => {
        let value = student[feature] || 0;
        // Simple standardization (you should use the same as in your main preprocessing)
        if (feature === 'absences') value = value / 10;
        if (feature === 'G1' || feature === 'G2') value = value / 20;
        featureVector.push(value);
    });
    
    // Add categorical features (one-hot encoding simplified)
    const categoricalFeatures = {
        school: ['GP', 'MS'],
        sex: ['F', 'M'],
        address: ['U', 'R'],
        famsize: ['GT3', 'LE3'],
        Pstatus: ['T', 'A'],
        Mjob: ['teacher', 'health', 'services', 'at_home', 'other'],
        Fjob: ['teacher', 'health', 'services', 'at_home', 'other'],
        reason: ['home', 'reputation', 'course', 'other'],
        guardian: ['mother', 'father', 'other'],
        schoolsup: ['yes', 'no'],
        famsup: ['yes', 'no'],
        paid: ['yes', 'no'],
        activities: ['yes', 'no'],
        nursery: ['yes', 'no'],
        higher: ['yes', 'no'],
        internet: ['yes', 'no'],
        romantic: ['yes', 'no']
    };
    
    Object.entries(categoricalFeatures).forEach(([feature, categories]) => {
        const value = student[feature];
        const oneHot = new Array(categories.length).fill(0);
        const index = categories.indexOf(value);
        if (index !== -1) oneHot[index] = 1;
        oneHot.forEach(val => featureVector.push(val));
    });
    
    return featureVector;
}

// Initialize practical examples when model is ready
function initializePracticalExamples() {
    console.log('Practical examples module loaded');
}
