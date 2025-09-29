// Create the neural network model (alternative version)
function createModel() {
    try {
        setStatus(elements.modelStatus, 'Creating model...', 'loading');
        
        if (!trainData || !trainData.features) {
            throw new Error('Please preprocess data first to determine input shape');
        }
        
        const inputDim = trainData.features.shape[1];
        console.log(`Creating model with input dimension: ${inputDim}`);
        
        // Simple sequential model with one hidden layer
        model = tf.sequential();
        
        // First layer with explicit input shape
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputDim]
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Print model summary
        model.summary();
        
        setStatus(elements.modelStatus, `Model created! Input shape: [${inputDim}]`, 'success');
        
    } catch (error) {
        setStatus(elements.modelStatus, `Model creation error: ${error.message}`, 'error');
        console.error('Model creation error:', error);
    }
}