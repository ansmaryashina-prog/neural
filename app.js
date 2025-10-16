// Perform Exploratory Data Analysis with better diagnostics
async function performEDA() {
    if (!studentData) {
        alert('Please load data first');
        return;
    }
    
    try {
        setStatus(elements.edaStatus, 'Performing Exploratory Data Analysis...', 'loading');
        
        console.log('=== EDA DIAGNOSTICS ===');
        console.log('Dataset sample:', studentData[0]);
        console.log('Available columns:', Object.keys(studentData[0]));
        console.log('Target column exists:', TARGET_COLUMN in studentData[0]);
        
        // Check target variable
        const targetValues = studentData.map(row => row[TARGET_COLUMN])
            .filter(v => v !== undefined && v !== null && !isNaN(v) && typeof v === 'number');
        
        console.log('Valid target values:', targetValues.length);
        console.log('Target values sample:', targetValues.slice(0, 5));
        
        if (targetValues.length === 0) {
            const availableColumns = Object.keys(studentData[0]);
            const numericColumns = availableColumns.filter(col => {
                const sample = studentData[0][col];
                return typeof sample === 'number' && !isNaN(sample);
            });
            
            throw new Error(
                `No valid target values in '${TARGET_COLUMN}'. ` +
                `Available numeric columns: ${numericColumns.join(', ')}. ` +
                `You might need to change TARGET_COLUMN to one of these.`
            );
        }
        
        // 1. Dataset Overview
        const overview = {
            totalStudents: studentData.length,
            totalFeatures: Object.keys(studentData[0]).length,
            numericFeatures: getNumericFeatures(studentData[0]).length,
            categoricalFeatures: getCategoricalFeatures(studentData[0]).length,
            targetColumn: TARGET_COLUMN,
            targetStats: {
                min: Math.min(...targetValues),
                max: Math.max(...targetValues),
                mean: targetValues.reduce((a, b) => a + b, 0) / targetValues.length,
                median: median(targetValues)
            }
        };
        
        console.log('Dataset Overview:', overview);
        
        // 2. Create EDA Visualizations
        await createEDAVisualizations();
        
        setStatus(elements.edaStatus, 
                 `EDA completed! Analyzed ${targetValues.length} students with ${overview.numericFeatures} numeric features`, 
                 'success');
        
    } catch (error) {
        setStatus(elements.edaStatus, `EDA error: ${error.message}`, 'error');
        console.error('EDA error details:', error);
    }
}
