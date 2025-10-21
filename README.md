# neural
Prompt

Build an interactive, client-only web app that performs end-to-end EDA and Neural Network churn prediction on the Telco Customer Churn dataset, including a business value simulator. Output exactly two code files:

index.html (HTML structure and UI)
app.js (JavaScript logic)
No other files. The app must run entirely in the browser with no server, suitable for GitHub Pages.

Tech stack and CDNs

Use TensorFlow.js CDN: https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest
Use tfjs-vis CDN: https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest
Use PapaParse for CSV parsing: https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js
No backend. All computation is in-browser.
Link app.js from index.html.
Dataset

Load CSV from: https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
Functional scope

EDA: dataset summary, churn rate, missingness, distributions, categorical breakdowns.
Preprocessing: clean and impute, one-hot encode categoricals from train only, standardize numeric features from train only.
Split: stratified 60/20/20 train/val/test on Churn.
Models: baseline logistic regression (1 dense sigmoid layer) and MLP (deep feed-forward) in TensorFlow.js.
Training: sample weights for class imbalance; early stopping on val_loss; training curves via tfjs-vis.
Evaluation: ROC AUC, PR AUC, accuracy, precision, recall, F1 at chosen thresholds; ROC/PR curves; confusion matrix; calibration plot with Brier score; decile lift chart.
Threshold selection: choose threshold that maximizes F1 and Youden’s J on validation; also allow manual threshold.
Business value simulator: profit-based threshold search and budgeted top-K targeting, with adjustable parameters and plots.
Explainability: permutation importance on a validation subsample for original features; simple partial dependence-like plots for key features.
Robustness: slice metrics by Contract, InternetService, SeniorCitizen.
Persistence: save/load model and preprocessing config in browser (localStorage) and enable downloads via downloads://.
UX flow and UI

Create a responsive UI with sections:
Header and dataset info (with link to CSV)
Controls
Buttons: Load Data, Run EDA, Preprocess & Split, Train Baseline, Train Neural Net, Evaluate, Business Simulation
Hyperparameters:
Hidden layers (comma-separated, e.g., 256,128,64)
Dropout (e.g., 0.3)
L2 regularization (e.g., 1e-4)
Learning rate (e.g., 0.001)
Batch size (e.g., 512)
Epochs (e.g., 100)
Random seed (e.g., 42)
Threshold controls:
Manual threshold input (0 to 1)
Buttons to set to F1-optimal and Youden-optimal from validation
Business parameters with defaults:
Offer cost = 20.0
Save rate = 0.30
Retention months = 6
Margin = 0.4
Budget K% range (min 5%, max 40%)
Status/log panel (append step-by-step messages)
EDA visualizations (tfjs-vis tabs or cards): churn distribution, numeric histograms, categorical stacked bars vs churn, missingness.
Training visuals (tfjs-vis history plots for loss; add a lightweight accuracy metric if desired)
Evaluation visuals:
ROC and PR curves for both models
Confusion matrix at chosen threshold
Calibration/reliability diagram and Brier score
Decile lift chart
Segment metrics table/plot
Business value results:
Profit vs threshold plot
Profit vs top-K% plot
Table of top 5 thresholds (and K%) by profit
Selected profit-optimal operating point and estimated annualized savings (clearly explain scaling assumption)
Save/load:
Buttons: Save Model (localStorage), Load Model (localStorage), Download Model (downloads://), Download Preprocessing JSON, Upload Config (file input)
Ensure all plots are labeled and readable. Use tfjs-vis surfaces for charts.
Data cleaning and preprocessing details

Coerce TotalCharges to numeric; count NaNs introduced; impute with median of TotalCharges.
Drop customerID (not a feature).
Target: Churn -> 1 for "Yes", 0 for "No".
Feature lists:
Numeric: tenure, MonthlyCharges, TotalCharges
Categorical: gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
Build preprocessing maps from the training split only:
One-hot categories discovered from train; unknown categories at inference -> all zeros for that feature group.
Standardization params (mean, std) computed on train numerics. Guard against std=0 (set to 1).
Output transformed dense Float32Array feature matrices for train/val/test, and original rows preserved for downstream business calcs (e.g., MonthlyCharges).
Print/log final feature dimension.
Split logic

Stratified by Churn to achieve approximate 60/20/20.
Implement reproducible RNG (seeded) for shuffling.
Keep indices for train/val/test; store split stats (churn rates per split).
Modeling

Baseline logistic regression:
tf.sequential with Dense(1, activation='sigmoid', kernelRegularizer=l2(...)) and inputShape=[nFeatures].
Loss: binaryCrossentropy; Optimizer: Adam(learningRate); Metrics: binaryAccuracy.
Early stopping on val_loss with patience (e.g., 8).
Use sampleWeight: set positive examples weight = n_neg / n_pos, negative weight = 1. Pass sampleWeight tensor to model.fit.
Neural Net (MLP):
Input layer -> Dense(h1, relu, l2) -> Dropout -> Dense(h2, relu) -> Dropout -> ... -> Dense(1, sigmoid).
Same optimizer/loss, early stopping on val_loss.
Use sampleWeight as above.
Plot training curves via tfvis.show.history (loss; optionally binaryAccuracy).
Make predictions (probabilities) on test for both models.
Evaluation and metrics

Implement in JS:
ROC curve, ROC AUC (trapezoidal rule)
Precision-Recall curve, PR AUC (trapezoidal rule)
Accuracy, Precision, Recall, F1 at:
Threshold = 0.5
Threshold maximizing F1 on validation
Threshold maximizing Youden’s J (TPR − FPR) on validation
Brier score (mean squared error of probabilities vs labels)
Visualize:
ROC/PR curves for both models
Confusion matrix at current threshold (counts and normalized)
Calibration plot: bin predictions into 10 bins; plot predicted vs empirical churn rate with error bars; report Brier score.
Decile lift chart: bin by predicted risk into 10 equal-sized bins; show churn rate per decile and cumulative capture of churners.
Report all metrics in a table-like div.
Threshold selection

Compute F1-optimal and Youden-optimal thresholds from validation predictions for the NN.
Provide buttons to apply these thresholds to test evaluation and refresh metrics/plots.
Allow manual threshold input.
Business value simulator

Use function: expected_value_per_customer(p, monthlyCharge, offerCost, saveRate, retentionMonths, margin) = p * saveRate * margin * monthlyCharge * retentionMonths − offerCost
Two policies, evaluated on test set using NN predictions:
Threshold-based: sweep t in [0.1, 0.9] step 0.01:
For customers with p >= t, compute total profit, profit per targeted, precision/recall.
Identify threshold that maximizes total profit; display KPIs.
Budgeted top-K: sweep K from 5% to 40% step 1%:
Target top K% highest-risk customers; compute KPIs and identify best K.
Use each customer’s own MonthlyCharges for revenue; handle missing/zero safely.
Visualize profit vs threshold and profit vs K%.
Estimate annualized savings by scaling test-set profit:
Annualized profit ≈ (profit_on_test / test_size) * total_dataset_size * 12 / retentionMonths
Show assumption text in the UI.
Explainability and robustness

Permutation importance:
On a validation subsample (e.g., max 2000 rows), for each original feature:
Permute raw column values; re-transform; recompute PR AUC; importance = baseline PR AUC − permuted PR AUC.
Show top 15 features.
Simple dependence plots:
For tenure and MonthlyCharges: bin values and plot average predicted churn per bin.
For Contract and InternetService: bar charts of average predicted churn per category.
Segment metrics:
Compute ROC AUC and calibration error by Contract type, InternetService, SeniorCitizen.
Display in a small table/chart.
Persistence and reproducibility

Set a global seed for Math.random and a seeded shuffle function.
Save:
tfjs model to localStorage: model.save('localstorage://telco-nn') and to downloads: model.save('downloads://telco-nn')
Preprocessing config (feature lists, category maps, numeric means/stds, selected threshold, business parameters) as a downloadable JSON file and to localStorage.
Load:
Reload model from localStorage if available.
Upload a JSON preprocessing config to score new rows.
Provide buttons for save/load and demonstrate scoring of a single manual input row in the UI.
Performance and memory

Use tf.tidy and dispose() to free tensors after operations.
Limit heavy computations (e.g., permutation importance) to a subsample with a user-visible note.
Keep batch operations efficient; prefer Float32Array and tensor2d creation in a single call.
Acceptance criteria

The app runs fully in the browser, no server required.
Clear, labeled UI with the sections described.
End-to-end flow: load data -> EDA -> preprocess/split -> train baseline -> train NN -> evaluate -> threshold selection -> business simulation -> save artifacts.
NN should typically outperform the baseline in PR AUC; if not, display a short note in the UI.
All required metrics and plots are shown: ROC/PR, confusion matrix, calibration and Brier, decile lift, profit curves, feature importance, segment metrics.
Save/load works (localStorage and downloads).
Code is split into exactly two files: index.html and app.js.
Implementation hints (required)

Implement helper functions (suggested names):
parseTelcoCsv(csvText)
imputeAndClean(rows)
buildPreprocessingMaps(trainRows)
transformRows(rows, preprocessingMaps) -> {X, y, meta}
stratifiedSplit(rows, y, seed, fractions={train:0.6,val:0.2,test:0.2})
buildLogisticModel(inputDim, l2)
buildMLPModel(inputDim, layerSizesArray, dropout, l2, learningRate)
computeSampleWeights(yTrain)
fitModel(model, XTrain, yTrain, XVal, yVal, sampleWeights, epochs, batchSize)
predictProba(model, X)
metrics: rocCurve, prCurve, aucFromCurve, confusionMatrixAtThreshold, brierScore
thresholdSearchF1, thresholdSearchYouden
decileLift(probs, labels)
businessProfitThresholdSweep(...)
businessProfitTopK(...)
permutationImportance(...)
segmentMetrics(...)
Use tfjs-vis surfaces (tfvis.visor()) with named tabs/containers, e.g., {name: 'EDA: Churn Distribution', tab: 'EDA'}.
Ensure index.html links tfjs, tfjs-vis, papaparse, and app.js in that order.
Now output the two files.

File 1: index.html

Full HTML document with a basic style.
Include the three script tags for TF.js, tfjs-vis, and PapaParse.
Include UI elements described in UX flow and link app.js at the end.
File 2: app.js

All logic as described, with modular functions and event listeners wiring up the UI.
Plenty of console.log and UI status logs as steps progress.
Use tf.tidy and dispose() where appropriate.
