// Titanic Dataset EDA - Client-side JavaScript
// Dataset schema - To use with other datasets, modify these arrays accordingly
const NUMERIC_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];
const TARGET_VARIABLE = 'Survived'; // Only available in train data
const IDENTIFIER = 'PassengerId'; // Exclude from analysis

let mergedData = []; // Store the merged dataset

// DOM elements
const loadDataBtn = document.getElementById('loadDataBtn');
const showOverviewBtn = document.getElementById('showOverviewBtn');
const showMissingBtn = document.getElementById('showMissingBtn');
const showStatsBtn = document.getElementById('showStatsBtn');
const showVizBtn = document.getElementById('showVizBtn');
const exportCsvBtn = document.getElementById('exportCsvBtn');
const exportJsonBtn = document.getElementById('exportJsonBtn');

// Event listeners
loadDataBtn.addEventListener('click', loadAndMergeData);
showOverviewBtn.addEventListener('click', showDataOverview);
showMissingBtn.addEventListener('click', analyzeMissingValues);
showStatsBtn.addEventListener('click', generateStatisticalSummary);
showVizBtn.addEventListener('click', generateVisualizations);
exportCsvBtn.addEventListener('click', exportMergedCSV);
exportJsonBtn.addEventListener('click', exportJSONSummary);

// Load and merge train and test data
function loadAndMergeData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both train.csv and test.csv files');
        return;
    }
    
    // Reset previous data
    mergedData = [];
    
    // Parse train data
    Papa.parse(trainFile, {
        header: true,
        dynamicTyping: true,
        quotes: true,
        complete: function(trainResults) {
            // Add source column to identify train data
            const trainData = trainResults.data.map(row => ({...row, source: 'train'}));
            
            // Parse test data
            Papa.parse(testFile, {
                header: true,
                dynamicTyping: true,
                quotes: true,
                complete: function(testResults) {
                    // Add source column to identify test data
                    const testData = testResults.data.map(row => ({...row, source: 'test'}));
                    
                    // Merge datasets
                    mergedData = [...trainData, ...testData];
                    
                    alert(`Data loaded successfully! Total records: ${mergedData.length}`);
                },
                error: function(error) {
                    alert('Error parsing test.csv: ' + error);
                }
            });
        },
        error: function(error) {
            alert('Error parsing train.csv: ' + error);
        }
    });
}

// Show data overview including preview and shape
function showDataOverview() {
    if (mergedData.length === 0) {
        alert('Please load data first');
        return;
    }
    
    const overviewContent = document.getElementById('overviewContent');
    overviewContent.innerHTML = '';
    
    // Display dataset shape
    const rowCount = mergedData.length;
    const colCount = Object.keys(mergedData[0]).length;
    
    const shapeInfo = document.createElement('p');
    shapeInfo.textContent = `Dataset shape: ${rowCount} rows × ${colCount} columns`;
    overviewContent.appendChild(shapeInfo);
    
    // Create preview table (first 10 rows)
    const previewTable = document.createElement('table');
    previewTable.innerHTML = '<caption>First 10 Rows of Merged Data</caption>';
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(mergedData[0]).forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    previewTable.appendChild(headerRow);
    
    // Add data rows
    for (let i = 0; i < Math.min(10, mergedData.length); i++) {
        const row = document.createElement('tr');
        Object.values(mergedData[i]).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value === null || value === undefined ? '' : value.toString();
            row.appendChild(td);
        });
        previewTable.appendChild(row);
    }
    
    overviewContent.appendChild(previewTable);
}

// Analyze and display missing values
function analyzeMissingValues() {
    if (mergedData.length === 0) {
        alert('Please load data first');
        return;
    }
    
    const missingContent = document.getElementById('missingContent');
    missingContent.innerHTML = '';
    
    // Calculate missing values for each column
    const missingValues = {};
    const totalRows = mergedData.length;
    
    Object.keys(mergedData[0]).forEach(col => {
        const missingCount = mergedData.filter(row => 
            row[col] === null || row[col] === undefined || row[col] === ''
        ).length;
        
        missingValues[col] = {
            count: missingCount,
            percentage: (missingCount / totalRows * 100).toFixed(2)
        };
    });
    
    // Create table for missing values
    const missingTable = document.createElement('table');
    missingTable.innerHTML = '<caption>Missing Values Analysis</caption>';
    
    const headerRow = document.createElement('tr');
    ['Column', 'Missing Count', 'Missing Percentage'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    missingTable.appendChild(headerRow);
    
    Object.entries(missingValues).forEach(([col, data]) => {
        const row = document.createElement('tr');
        
        const colCell = document.createElement('td');
        colCell.textContent = col;
        row.appendChild(colCell);
        
        const countCell = document.createElement('td');
        countCell.textContent = data.count;
        row.appendChild(countCell);
        
        const percCell = document.createElement('td');
        percCell.textContent = `${data.percentage}%`;
        row.appendChild(percCell);
        
        missingTable.appendChild(row);
    });
    
    missingContent.appendChild(missingTable);
    
    // Create bar chart for missing values
    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    missingContent.appendChild(chartContainer);
    
    const ctx = document.createElement('canvas');
    chartContainer.appendChild(ctx);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(missingValues),
            datasets: [{
                label: 'Missing Values Percentage',
                data: Object.values(missingValues).map(v => v.percentage),
                backgroundColor: 'rgba(255, 99, 132, 0.7)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Percentage Missing'
                    }
                }
            }
        }
    });
}

// Generate statistical summary
function generateStatisticalSummary() {
    if (mergedData.length === 0) {
        alert('Please load data first');
        return;
    }
    
    const statsContent = document.getElementById('statsContent');
    statsContent.innerHTML = '';
    
    // Separate train and test data for analysis
    const trainData = mergedData.filter(row => row.source === 'train');
    const testData = mergedData.filter(row => row.source === 'test');
    
    // Numeric features summary
    const numericStats = {};
    NUMERIC_FEATURES.forEach(feature => {
        const values = trainData.map(row => row[feature]).filter(val => val !== null && val !== undefined);
        
        if (values.length > 0) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const sorted = [...values].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
            
            numericStats[feature] = {
                mean: mean.toFixed(2),
                median: median.toFixed(2),
                std: std.toFixed(2),
                min: Math.min(...values).toFixed(2),
                max: Math.max(...values).toFixed(2)
            };
        }
    });
    
    // Create numeric stats table
    const numericTable = document.createElement('table');
    numericTable.innerHTML = '<caption>Numeric Features Summary (Train Data)</caption>';
    
    const numericHeader = document.createElement('tr');
    ['Feature', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        numericHeader.appendChild(th);
    });
    numericTable.appendChild(numericHeader);
    
    Object.entries(numericStats).forEach(([feature, stats]) => {
        const row = document.createElement('tr');
        
        [feature, stats.mean, stats.median, stats.std, stats.min, stats.max].forEach(value => {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.appendChild(cell);
        });
        
        numericTable.appendChild(row);
    });
    
    statsContent.appendChild(numericTable);
    
    // Categorical features summary
    CATEGORICAL_FEATURES.forEach(feature => {
        const valueCounts = {};
        trainData.forEach(row => {
            if (row[feature] !== null && row[feature] !== undefined) {
                valueCounts[row[feature]] = (valueCounts[row[feature]] || 0) + 1;
            }
        });
        
        const catTable = document.createElement('table');
        catTable.innerHTML = `<caption>${feature} Value Counts (Train Data)</caption>`;
        
        const catHeader = document.createElement('tr');
        [feature, 'Count', 'Percentage'].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            catHeader.appendChild(th);
        });
        catTable.appendChild(catHeader);
        
        Object.entries(valueCounts).forEach(([value, count]) => {
            const row = document.createElement('tr');
            
            const valueCell = document.createElement('td');
            valueCell.textContent = value;
            row.appendChild(valueCell);
            
            const countCell = document.createElement('td');
            countCell.textContent = count;
            row.appendChild(countCell);
            
            const percCell = document.createElement('td');
            percCell.textContent = ((count / trainData.length) * 100).toFixed(2) + '%';
            row.appendChild(percCell);
            
            catTable.appendChild(row);
        });
        
        statsContent.appendChild(catTable);
    });
    
    // Survival analysis (only for train data)
    if (trainData.some(row => row[TARGET_VARIABLE] !== undefined)) {
        const survivalByFeature = {};
        
        // Analyze survival by categorical features
        CATEGORICAL_FEATURES.forEach(feature => {
            survivalByFeature[feature] = {};
            
            trainData.forEach(row => {
                if (row[feature] !== null && row[feature] !== undefined && 
                    row[TARGET_VARIABLE] !== null && row[TARGET_VARIABLE] !== undefined) {
                    
                    if (!survivalByFeature[feature][row[feature]]) {
                        survivalByFeature[feature][row[feature]] = { survived: 0, total: 0 };
                    }
                    
                    survivalByFeature[feature][row[feature]].total++;
                    if (row[TARGET_VARIABLE] === 1) {
                        survivalByFeature[feature][row[feature]].survived++;
                    }
                }
            });
        });
        
        // Create survival rate tables
        Object.entries(survivalByFeature).forEach(([feature, values]) => {
            const survivalTable = document.createElement('table');
            survivalTable.innerHTML = `<caption>Survival Rate by ${feature} (Train Data)</caption>`;
            
            const survivalHeader = document.createElement('tr');
            [feature, 'Total', 'Survived', 'Survival Rate'].forEach(header => {
                const th = document.createElement('th');
                th.textContent = header;
                survivalHeader.appendChild(th);
            });
            survivalTable.appendChild(survivalHeader);
            
            Object.entries(values).forEach(([value, data]) => {
                const row = document.createElement('tr');
                
                const valueCell = document.createElement('td');
                valueCell.textContent = value;
                row.appendChild(valueCell);
                
                const totalCell = document.createElement('td');
                totalCell.textContent = data.total;
                row.appendChild(totalCell);
                
                const survivedCell = document.createElement('td');
                survivedCell.textContent = data.survived;
                row.appendChild(survivedCell);
                
                const rateCell = document.createElement('td');
                rateCell.textContent = ((data.survived / data.total) * 100).toFixed(2) + '%';
                row.appendChild(rateCell);
                
                survivalTable.appendChild(row);
            });
            
            statsContent.appendChild(survivalTable);
        });
    }
}

// Generate visualizations
function generateVisualizations() {
    if (mergedData.length === 0) {
        alert('Please load data first');
        return;
    }
    
    const vizContent = document.getElementById('vizContent');
    vizContent.innerHTML = '';
    
    const trainData = mergedData.filter(row => row.source === 'train');
    
    // Create container for charts
    const chartsContainer = document.createElement('div');
    chartsContainer.className = 'flex-row';
    vizContent.appendChild(chartsContainer);
    
    // Survival count chart (if target variable exists)
    if (trainData.some(row => row[TARGET_VARIABLE] !== undefined)) {
        const survivalCounts = {0: 0, 1: 0};
        trainData.forEach(row => {
            if (row[TARGET_VARIABLE] !== null && row[TARGET_VARIABLE] !== undefined) {
                survivalCounts[row[TARGET_VARIABLE]]++;
            }
        });
        
        const survivalChartContainer = document.createElement('div');
        survivalChartContainer.className = 'flex-chart';
        chartsContainer.appendChild(survivalChartContainer);
        
        const survivalCanvas = document.createElement('canvas');
        survivalChartContainer.appendChild(survivalCanvas);
        
        new Chart(survivalCanvas, {
            type: 'bar',
            data: {
                labels: ['Did Not Survive', 'Survived'],
                datasets: [{
                    label: 'Passenger Count',
                    data: [survivalCounts[0], survivalCounts[1]],
                    backgroundColor: ['rgba(255, 99, 132, 0.7)', 'rgba(75, 192, 192, 0.7)'],
                    borderColor: ['rgba(255, 99, 132, 1)', 'rgba(75, 192, 192, 1)'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Survival Count'
                    }
                }
            }
        });
    }
    
    // Sex distribution chart
    const sexCounts = {};
    trainData.forEach(row => {
        if (row.Sex) {
            sexCounts[row.Sex] = (sexCounts[row.Sex] || 0) + 1;
        }
    });
    
    const sexChartContainer = document.createElement('div');
    sexChartContainer.className = 'flex-chart';
    chartsContainer.appendChild(sexChartContainer);
    
    const sexCanvas = document.createElement('canvas');
    sexChartContainer.appendChild(sexCanvas);
    
    new Chart(sexCanvas, {
        type: 'pie',
        data: {
            labels: Object.keys(sexCounts),
            datasets: [{
                data: Object.values(sexCounts),
                backgroundColor: ['rgba(54, 162, 235, 0.7)', 'rgba(255, 99, 132, 0.7)'],
                borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Passenger Gender Distribution'
                }
            }
        }
    });
    
    // Pclass distribution chart
    const pclassCounts = {};
    trainData.forEach(row => {
        if (row.Pclass) {
            pclassCounts[row.Pclass] = (pclassCounts[row.Pclass] || 0) + 1;
        }
    });
    
    const pclassChartContainer = document.createElement('div');
    pclassChartContainer.className = 'flex-chart';
    chartsContainer.appendChild(pclassChartContainer);
    
    const pclassCanvas = document.createElement('canvas');
    pclassChartContainer.appendChild(pclassCanvas);
    
    new Chart(pclassCanvas, {
        type: 'bar',
        data: {
            labels: Object.keys(pclassCounts).map(key => `Class ${key}`),
            datasets: [{
                label: 'Passenger Count',
                data: Object.values(pclassCounts),
                backgroundColor: 'rgba(153, 102, 255, 0.7)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Passenger Class Distribution'
                }
            }
        }
    });
    
    // Age distribution histogram
    const ages = trainData
        .map(row => row.Age)
        .filter(age => age !== null && age !== undefined);
    
    if (ages.length > 0) {
        const ageChartContainer = document.createElement('div');
        ageChartContainer.className = 'chart-container';
        vizContent.appendChild(ageChartContainer);
        
        const ageCanvas = document.createElement('canvas');
        ageChartContainer.appendChild(ageCanvas);
        
        new Chart(ageCanvas, {
            type: 'histogram',
            data: {
                datasets: [{
                    label: 'Age Distribution',
                    data: ages,
                    backgroundColor: 'rgba(255, 159, 64, 0.7)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Age Distribution'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Age'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                }
            }
        });
    }
    
    // Fare distribution histogram
    const fares = trainData
        .map(row => row.Fare)
        .filter(fare => fare !== null && fare !== undefined);
    
    if (fares.length > 0) {
        const fareChartContainer = document.createElement('div');
        fareChartContainer.className = 'chart-container';
        vizContent.appendChild(fareChartContainer);
        
        const fareCanvas = document.createElement('canvas');
        fareChartContainer.appendChild(fareCanvas);
        
        new Chart(fareCanvas, {
            type: 'histogram',
            data: {
                datasets: [{
                    label: 'Fare Distribution',
                    data: fares,
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Fare Distribution'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Fare'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                }
            }
        });
    }
}

// Export merged data as CSV
function exportMergedCSV() {
    if (mergedData.length === 0) {
        alert('Please load data first');
        return;
    }
    
    try {
        const csv = Papa.unparse(mergedData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'titanic_merged_data.csv');
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        document.getElementById('exportStatus').textContent = 'CSV exported successfully!';
    } catch (error) {
        alert('Error exporting CSV: ' + error);
    }
}

// Export JSON summary
function exportJSONSummary() {
    if (mergedData.length === 0) {
        alert('Please load data first');
        return;
    }
    
    try {
        const trainData = mergedData.filter(row => row.source === 'train');
        const testData = mergedData.filter(row => row.source === 'test');
        
        // Create summary object
        const summary = {
            dataset: 'Titanic',
            recordCount: {
                total: mergedData.length,
                train: trainData.length,
                test: testData.length
            },
            columns: Object.keys(mergedData[0]),
            numericFeatures: NUMERIC_FEATURES,
            categoricalFeatures: CATEGORICAL_FEATURES,
            generated: new Date().toISOString()
        };
        
        const json = JSON.stringify(summary, null, 2);
        const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'titanic_data_summary.json');
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        document.getElementById('exportStatus').textContent = 'JSON summary exported successfully!';
    } catch (error) {
        alert('Error exporting JSON: ' + error);
    }
}
