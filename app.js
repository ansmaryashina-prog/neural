/* Telco Churn EDA + NN + Business Simulator - Client-only (Upload CSV)
Uses: TensorFlow.js, tfjs-vis, PapaParse
*/

const NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges'];
const CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
];
const TARGET_COL = 'Churn';
const ID_COL = 'customerID';

const STATE = {
    seed: 42,
    rawRows: [],
    cleanedRows: [],
    split: null,
    maps: null,
    transformed: null,
    models: { baseline: null, mlp: null },
    predictions: { baseline: { val: null, test: null }, mlp: { val: null, test: null } },
    thresholds: { manual: 0.5, f1: null, youden: null, selected: 'manual' },
    business: { offerCost: 20.0, saveRate: 0.30, retentionMonths: 6, margin: 0.4, kMin: 5, kMax: 40 },
    sampleWeights: null,
    trainingHistory: {},
};

function appendLog(msg) {
    const el = document.getElementById('log');
    const now = new Date().toLocaleTimeString();
    el.textContent += `[${now}] ${msg}\n`;
    el.scrollTop = el.scrollHeight;
    console.log(msg);
}

// Seeded RNG
function mulberry32(a) {
    return function() {
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

function seededShuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}

// CSV parsing
function parseTelcoCsv(csvText) {
    const parsed = Papa.parse(csvText, { header: true, dynamicTyping: false, skipEmptyLines: true });
    let rows = parsed.data;
    rows = rows.map(r => {
        const o = {};
        for (const k in r) o[k] = r[k] === '' ? null : r[k];
        return o;
    });
    return rows;
}

// Load data from user file
async function loadDataFromFile() {
    const input = document.getElementById('datasetFile');
    const file = input.files && input.files[0];
    if (!file) {
        appendLog('Please choose a CSV file in the "Upload dataset CSV" field.');
        return [];
    }
    
    appendLog(`Reading file: ${file.name} (${Math.round(file.size/1024)} KB) ...`);
    
    try {
        const text = await readFileAsText(file);
        const rows = parseTelcoCsv(text);
        
        if (rows.length === 0) {
            appendLog('Error: No data rows found in CSV file.');
            return [];
        }
        
        STATE.rawRows = rows;
        appendLog(`Loaded ${rows.length} rows, ${Object.keys(rows[0]).length} columns from file.`);
        return rows;
    } catch (error) {
        appendLog(`Error loading file: ${error.message}`);
        return [];
    }
}

// Helper function to read file as text
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

// Cleaning and imputation
function imputeAndClean(rows) {
    appendLog('Cleaning data: coercing numerics, imputing TotalCharges median, mapping target...');
    const cleaned = rows.map(r => ({ ...r }));
    let totalChargesVals = [];
    let totalNaNsIntroduced = 0;
    
    for (const r of cleaned) {
        r.tenure = r.tenure != null ? Number(r.tenure) : null;
        r.MonthlyCharges = r.MonthlyCharges != null ? Number(r.MonthlyCharges) : null;
        const tc = r.TotalCharges != null ? Number(r.TotalCharges) : NaN;
        if (isNaN(tc)) totalNaNsIntroduced++;
        else totalChargesVals.push(tc);
    }
    
    const median = (() => {
        const a = totalChargesVals.slice().sort((a, b) => a - b);
        const mid = Math.floor(a.length / 2);
        return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
    })();
    
    for (const r of cleaned) {
        let tc = Number(r.TotalCharges);
        if (isNaN(tc)) tc = median;
        r.TotalCharges = tc;
        if (r.MonthlyCharges == null || isNaN(r.MonthlyCharges)) r.MonthlyCharges = 0;
        if (r.tenure == null || isNaN(r.tenure)) r.tenure = 0;
        r._y = r[TARGET_COL] === 'Yes' ? 1 : 0;
    }
    
    appendLog(`TotalCharges NaNs introduced: ${totalNaNsIntroduced}. Imputed with median=${median.toFixed(2)}.`);
    STATE.cleanedRows = cleaned;
    return cleaned;
}

// Stratified split
function stratifiedSplit(rows, seed, fractions = { train: 0.6, val: 0.2, test: 0.2 }) {
    const idxPos = [], idxNeg = [];
    rows.forEach((r, i) => (r._y === 1 ? idxPos : idxNeg).push(i));
    const rng = mulberry32(seed);
    seededShuffle(idxPos, rng);
    seededShuffle(idxNeg, rng);
    
    const take = (arr, fracs) => {
        const n = arr.length;
        const nTrain = Math.round(fracs.train * n);
        const nVal = Math.round(fracs.val * n);
        const train = arr.slice(0, nTrain);
        const val = arr.slice(nTrain, nTrain + nVal);
        const test = arr.slice(nTrain + nVal);
        return { train, val, test };
    };
    
    const pos = take(idxPos, fractions);
    const neg = take(idxNeg, fractions);
    const trainIdxs = pos.train.concat(neg.train);
    const valIdxs = pos.val.concat(neg.val);
    const testIdxs = pos.test.concat(neg.test);
    
    seededShuffle(trainIdxs, rng);
    seededShuffle(valIdxs, rng);
    seededShuffle(testIdxs, rng);
    
    const stats = s => {
        const ys = s.map(i => rows[i]._y);
        const rate = ys.reduce((a, b) => a + b, 0) / ys.length;
        return { size: s.length, churnRate: rate };
    };
    
    const split = {
        trainIdxs, valIdxs, testIdxs,
        stats: {
            train: stats(trainIdxs),
            val: stats(valIdxs),
            test: stats(testIdxs)
        }
    };
    
    appendLog(`Split sizes: train=${split.stats.train.size} (churn=${(100 * split.stats.train.churnRate).toFixed(1)}%), val=${split.stats.val.size} (${(100 * split.stats.val.churnRate).toFixed(1)}%), test=${split.stats.test.size} (${(100 * split.stats.test.churnRate).toFixed(1)}%)`);
    STATE.split = split;
    return split;
}

// Build preprocessing maps from train
function buildPreprocessingMaps(trainRows) {
    const categoryMaps = {};
    for (const c of CATEGORICAL_FEATURES) {
        const set = new Set();
        for (const r of trainRows) {
            const v = r[c] == null ? 'Unknown' : String(r[c]);
            set.add(v);
        }
        categoryMaps[c] = Array.from(set.values());
    }
    
    const numStats = {};
    for (const n of NUMERIC_FEATURES) {
        const vals = trainRows.map(r => Number(r[n])).filter(v => isFinite(v));
        const mean = vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length);
        const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, vals.length));
        numStats[n] = { mean, std: std === 0 ? 1 : std };
    }
    
    const featureOrder = [];
    for (const n of NUMERIC_FEATURES) featureOrder.push(n);
    for (const c of CATEGORICAL_FEATURES) {
        for (const cat of categoryMaps[c]) featureOrder.push(`${c}__${cat}`);
    }
    
    const maps = { categoryMaps, numStats, featureOrder, numeric: NUMERIC_FEATURES, categorical: CATEGORICAL_FEATURES };
    STATE.maps = maps;
    appendLog(`Built preprocessing maps. One-hot dims: ${featureOrder.length - NUMERIC_FEATURES.length}. Total features: ${featureOrder.length}.`);
    return maps;
}

// Transform rows
function transformRows(rows, maps) {
    const n = rows.length;
    const d = maps.featureOrder.length;
    const X = new Float32Array(n * d);
    const y = new Float32Array(n);
    const meta = { monthly: new Float32Array(n), ids: [], raw: rows };
    const idxMap = {};
    maps.featureOrder.forEach((f, i) => idxMap[f] = i);
    
    for (let i = 0; i < n; i++) {
        const r = rows[i];
        const id = i * d;
        
        for (const num of maps.numeric) {
            const j = idxMap[num];
            let v = Number(r[num]);
            if (!isFinite(v)) v = maps.numStats[num].mean;
            v = (v - maps.numStats[num].mean) / maps.numStats[num].std;
            X[id + j] = v;
        }
        
        for (const catCol of maps.categorical) {
            const catVals = maps.categoryMaps[catCol];
            const v = r[catCol] == null ? 'Unknown' : String(r[catCol]);
            const idx = catVals.indexOf(v);
            if (idx !== -1) {
                const featName = `${catCol}__${catVals[idx]}`;
                const j = idxMap[featName];
                X[id + j] = 1;
            }
        }
        
        y[i] = Number(r._y ?? (r[TARGET_COL] === 'Yes' ? 1 : 0));
        meta.monthly[i] = Number(r.MonthlyCharges) || 0;
        meta.ids.push(r[ID_COL] || String(i));
    }
    
    return { X, y, meta, nRows: n, nCols: d };
}

// Models
function buildLogisticModel(inputDim, l2) {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelRegularizer: tf.regularizers.l2({ l2 }),
        inputShape: [inputDim]
    }));
    return model;
}

function buildMLPModel(inputDim, layerSizes, dropout, l2, learningRate) {
    const model = tf.sequential();
    let first = true;
    
    for (let i = 0; i < layerSizes.length; i++) {
        const units = layerSizes[i];
        model.add(tf.layers.dense({
            units,
            activation: 'relu',
            inputShape: first ? [inputDim] : undefined,
            kernelRegularizer: tf.regularizers.l2({ l2 })
        }));
        first = false;
        if (dropout > 0) model.add(tf.layers.dropout({ rate: dropout }));
    }
    
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    const opt = tf.train.adam(learningRate);
    model.compile({ optimizer: opt, loss: 'binaryCrossentropy', metrics: ['binaryAccuracy'] });
    return model;
}

function compileModel(model, learningRate) {
    const opt = tf.train.adam(learningRate);
    model.compile({ optimizer: opt, loss: 'binaryCrossentropy', metrics: ['binaryAccuracy'] });
}

function computeSampleWeights(yTrain) {
    let pos = 0, neg = 0;
    for (let i = 0; i < yTrain.length; i++) (yTrain[i] === 1 ? pos++ : neg++);
    const wPos = pos > 0 ? neg / pos : 1;
    const w = new Float32Array(yTrain.length);
    for (let i = 0; i < yTrain.length; i++) w[i] = yTrain[i] === 1 ? wPos : 1.0;
    appendLog(`Computed sample weights: posWeight=${wPos.toFixed(3)}, nPos=${pos}, nNeg=${neg}.`);
    return w;
}

async function fitModel(model, XTrain, yTrain, XVal, yVal, sampleWeights, epochs, batchSize, tag = 'Model') {
    const nTrain = yTrain.length;
    const nVal = yVal.length;
    const d = XTrain.length / nTrain;
    const xTrainT = tf.tensor2d(XTrain, [nTrain, d]);
    const yTrainT = tf.tensor2d(yTrain, [nTrain, 1]);
    const xValT = tf.tensor2d(XVal, [nVal, d]);
    const yValT = tf.tensor2d(yVal, [nVal, 1]);

    const historyContainer = { name: `${tag} Training`, tab: 'Training' };
    let hist;
    
    try {
        hist = await model.fit(xTrainT, yTrainT, {
            epochs,
            batchSize,
            validationData: [xValT, yValT],
            callbacks: [
                tfvis.show.fitCallbacks(historyContainer, ['loss', 'val_loss', 'binaryAccuracy', 'val_binaryAccuracy'], { callbacks: ['onEpochEnd'] }),
                tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 8, minDelta: 1e-4 })
            ],
            sampleWeight: tf.tensor1d(sampleWeights)
        });
    } catch (err) {
        appendLog('Warning: sampleWeight not supported in this environment. Training without weights. ' + err.message);
        hist = await model.fit(xTrainT, yTrainT, {
            epochs,
            batchSize,
            validationData: [xValT, yValT],
            callbacks: [
                tfvis.show.fitCallbacks(historyContainer, ['loss', 'val_loss', 'binaryAccuracy', 'val_binaryAccuracy'], { callbacks: ['onEpochEnd'] }),
                tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 8, minDelta: 1e-4 })
            ]
        });
    } finally {
        xTrainT.dispose();
        yTrainT.dispose();
        xValT.dispose();
        yValT.dispose();
    }
    return hist;
}

function predictProba(model, X) {
    const n = X.length ? (X.length / STATE.maps.featureOrder.length) : 0;
    if (n === 0) return new Float32Array([]);
    const d = STATE.maps.featureOrder.length;
    
    return tf.tidy(() => {
        const xT = tf.tensor2d(X, [n, d]);
        const preds = model.predict(xT);
        const probs = preds.dataSync();
        xT.dispose();
        preds.dispose();
        return Float32Array.from(probs);
    });
}

// Metrics
function rocCurve(probs, labels) {
    const pairs = [];
    for (let i = 0; i < probs.length; i++) pairs.push([probs[i], labels[i]]);
    pairs.sort((a, b) => b[0] - a[0]);
    
    let P = 0, N = 0;
    for (const p of pairs) (p[1] === 1 ? P++ : N++);
    
    let tp = 0, fp = 0;
    const tpr = [0], fpr = [0];
    let lastProb = Infinity;
    
    for (let i = 0; i < pairs.length; i++) {
        const [prob, y] = pairs[i];
        if (prob !== lastProb) {
            tpr.push(tp / (P || 1));
            fpr.push(fp / (N || 1));
            lastProb = prob;
        }
        if (y === 1) tp++;
        else fp++;
    }
    
    tpr.push(1);
    fpr.push(1);
    return { fpr, tpr };
}

function prCurve(probs, labels) {
    const pairs = [];
    for (let i = 0; i < probs.length; i++) pairs.push([probs[i], labels[i]]);
    pairs.sort((a, b) => b[0] - a[0]);
    
    let tp = 0, fp = 0, P = labels.reduce((a, b) => a + b, 0);
    const precision = [], recall = [];
    let lastProb = Infinity;
    
    for (let i = 0; i < pairs.length; i++) {
        const [prob, y] = pairs[i];
        if (prob !== lastProb) {
            const prec = tp + fp === 0 ? 1 : tp / (tp + fp);
            const rec = P === 0 ? 0 : tp / P;
            precision.push(prec);
            recall.push(rec);
            lastProb = prob;
        }
        if (y === 1) tp++;
        else fp++;
    }
    
    precision.push(tp + fp === 0 ? 1 : tp / (tp + fp));
    recall.push(P === 0 ? 0 : tp / P);
    return { precision, recall };
}

function aucFromCurve(x, y) {
    let auc = 0;
    for (let i = 1; i < x.length; i++) {
        const dx = x[i] - x[i - 1];
        const h = (y[i] + y[i - 1]) / 2;
        auc += dx * h;
    }
    return auc;
}

function confusionMatrixAtThreshold(probs, labels, t) {
    let TP = 0, FP = 0, TN = 0, FN = 0;
    for (let i = 0; i < probs.length; i++) {
        const pred = probs[i] >= t ? 1 : 0;
        const y = labels[i];
        if (pred === 1 && y === 1) TP++;
        else if (pred === 1 && y === 0) FP++;
        else if (pred === 0 && y === 1) FN++;
        else TN++;
    }
    return { TP, FP, TN, FN };
}

function brierScore(probs, labels) {
    let s = 0;
    for (let i = 0; i < probs.length; i++) {
        const e = probs[i] - labels[i];
        s += e * e;
    }
    return s / (probs.length || 1);
}

function basicMetricsAtThreshold(probs, labels, t) {
    const { TP, FP, TN, FN } = confusionMatrixAtThreshold(probs, labels, t);
    const acc = (TP + TN) / (TP + TN + FP + FN || 1);
    const prec = TP / (TP + FP || 1);
    const rec = TP / (TP + FN || 1);
    const f1 = (2 * prec * rec) / (prec + rec || 1);
    const fpr = FP / (FP + TN || 1);
    const tpr = rec;
    return { acc, prec, rec, f1, TP, FP, TN, FN, fpr, tpr };
}

function thresholdSearchF1(probs, labels) {
    let best = { thr: 0.5, f1: -1, acc: 0, prec: 0, rec: 0 };
    for (let t = 0.01; t < 0.99; t += 0.01) {
        const m = basicMetricsAtThreshold(probs, labels, t);
        if (m.f1 > best.f1) best = { thr: +t.toFixed(2), f1: m.f1, acc: m.acc, prec: m.prec, rec: m.rec };
    }
    return best;
}

function thresholdSearchYouden(probs, labels) {
    let best = { thr: 0.5, J: -2, tpr: 0, fpr: 0 };
    for (let t = 0.01; t < 0.99; t += 0.01) {
        const m = basicMetricsAtThreshold(probs, labels, t);
        const J = m.tpr - m.fpr;
        if (J > best.J) best = { thr: +t.toFixed(2), J, tpr: m.tpr, fpr: m.fpr };
    }
    return best;
}

function decileLift(probs, labels) {
    const n = probs.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => probs[b] - probs[a]);
    
    const decileSize = Math.max(1, Math.floor(n / 10));
    const rates = [];
    const cumCapture = [];
    const totalPos = labels.reduce((a, b) => a + b, 0);
    let cumPos = 0;
    
    for (let d = 0; d < 10; d++) {
        const start = d * decileSize;
        const end = d === 9 ? n : start + decileSize;
        const slice = idx.slice(start, end);
        const pos = slice.reduce((a, i) => a + labels[i], 0);
        const rate = pos / (slice.length || 1);
        rates.push({ x: `D${d + 1}`, y: rate });
        cumPos += pos;
        cumCapture.push({ x: `D${d + 1}`, y: totalPos === 0 ? 0 : cumPos / totalPos });
    }
    
    return { rates, cumCapture };
}

function calibrationBins(probs, labels, nBins = 10) {
    const pairs = probs.map((p, i) => ({ p, y: labels[i] })).sort((a, b) => a.p - b.p);
    const bins = [];
    
    for (let b = 0; b < nBins; b++) {
        const start = Math.floor(b * pairs.length / nBins);
        const end = Math.floor((b + 1) * pairs.length / nBins);
        const slice = pairs.slice(start, end);
        const avgP = slice.reduce((a, s) => a + s.p, 0) / (slice.length || 1);
        const avgY = slice.reduce((a, s) => a + s.y, 0) / (slice.length || 1);
        bins.push({ bin: b + 1, avgP, avgY, size: slice.length });
    }
    
    const ece = bins.reduce((a, bin) => a + (bin.size / (probs.length || 1)) * Math.abs(bin.avgP - bin.avgY), 0);
    return { bins, ece };
}

// EDA
async function runEDA(rows) {
    tfvis.visor().open();
    const tab = 'EDA';
    const churnYes = rows.reduce((a, r) => a + (r._y === 1 ? 1 : 0), 0);
    
    await tfvis.render.barchart({ name: 'EDA: Churn Distribution', tab }, {
        values: [{ x: 'No', y: rows.length - churnYes }, { x: 'Yes', y: churnYes }],
        series: ['count']
    }, { xLabel: 'Churn', yLabel: 'Count' });

    const cols = Object.keys(rows[0] || {});
    const miss = cols.map(c => {
        const m = rows.reduce((a, r) => a + ((r[c] === null || r[c] === undefined || r[c] === '') ? 1 : 0), 0);
        return { x: c, y: m / rows.length };
    });
    
    await tfvis.render.barchart({ name: 'EDA: Missingness', tab }, { values: miss, series: ['Missing%'] }, { xLabel: 'Column', yLabel: 'Fraction Missing', height: 300 });

    for (const n of NUMERIC_FEATURES) {
        const vals = rows.map(r => Number(r[n]) || 0);
        await tfvis.render.histogram({ name: `EDA: ${n} histogram`, tab }, vals, { xLabel: n, yLabel: 'Freq', height: 200 });
    }

    for (const c of CATEGORICAL_FEATURES) {
        const agg = {};
        for (const r of rows) {
            const v = r[c] == null ? 'Unknown' : String(r[c]);
            if (!agg[v]) agg[v] = { cnt: 0, pos: 0 };
            agg[v].cnt++;
            agg[v].pos += (r._y === 1 ? 1 : 0);
        }
        const bars = Object.keys(agg).map(k => ({ x: k, y: agg[k].pos / (agg[k].cnt || 1) }));
        await tfvis.render.barchart({ name: `EDA: Churn rate by ${c}`, tab }, { values: bars, series: ['Churn rate'] }, { xLabel: c, yLabel: 'Churn rate', height: 280 });
    }
    
    appendLog(`EDA complete. Churn rate overall: ${(100 * churnYes / rows.length).toFixed(1)}%.`);
}

// Business
function expectedValuePerCustomer(p, monthlyCharge, params) {
    const { offerCost, saveRate, retentionMonths, margin } = params;
    const m = isFinite(monthlyCharge) ? monthlyCharge : 0;
    return (p * saveRate * margin * m * retentionMonths) - offerCost;
}

function businessProfitThresholdSweep(probs, labels, monthly, params) {
    const results = [];
    for (let t = 0.1; t <= 0.9; t += 0.01) {
        let targeted = 0, profit = 0, tp = 0, pos = labels.reduce((a, b) => a + b, 0);
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] >= t) {
                targeted++;
                profit += expectedValuePerCustomer(probs[i], monthly[i], params);
                if (labels[i] === 1) tp++;
            }
        }
        const precision = targeted === 0 ? 0 : tp / targeted;
        const recall = pos === 0 ? 0 : tp / pos;
        results.push({ t: +t.toFixed(2), targeted, profit, precision, recall });
    }
    const best = results.reduce((a, b) => b.profit > a.profit ? b : a, { profit: -Infinity });
    return { results, best };
}

function businessProfitTopK(probs, labels, monthly, params, kMin = 5, kMax = 40) {
    const n = probs.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
    const results = [];
    
    for (let k = kMin; k <= kMax; k += 1) {
        const m = Math.floor(n * (k / 100));
        let profit = 0, tp = 0, targeted = m;
        let pos = labels.reduce((a, b) => a + b, 0);
        
        for (let i = 0; i < m; i++) {
            const j = idx[i];
            profit += expectedValuePerCustomer(probs[j], monthly[j], params);
            if (labels[j] === 1) tp++;
        }
        
        const precision = targeted === 0 ? 0 : tp / targeted;
        const recall = pos === 0 ? 0 : tp / pos;
        results.push({ k, targeted, profit, precision, recall });
    }
    
    const best = results.reduce((a, b) => b.profit > a.profit ? b : a, { profit: -Infinity });
    return { results, best };
}

// Permutation importance
async function permutationImportance(valRows, maps, model, maxN = 2000) {
    const n = Math.min(maxN, valRows.length);
    const sample = valRows.slice(0, n);
    const base = transformRows(sample, maps);
    const baseProbs = predictProba(model, base.X);
    const basePR = prCurve(baseProbs, base.y);
    const basePRAUC = aucFromCurve(basePR.recall, basePR.precision);

    const feats = [...NUMERIC_FEATURES, ...CATEGORICAL_FEATURES];
    const importance = [];
    
    for (const f of feats) {
        const perm = sample.map(r => ({ ...r }));
        const vals = perm.map(r => r[f]);
        const rng = mulberry32(STATE.seed + 123);
        
        for (let i = vals.length - 1; i > 0; i--) {
            const j = Math.floor(rng() * (i + 1));
            const tmp = vals[i];
            vals[i] = vals[j];
            vals[j] = tmp;
        }
        
        for (let i = 0; i < perm.length; i++) perm[i][f] = vals[i];
        const t = transformRows(perm, maps);
        const probs = predictProba(model, t.X);
        const pr = prCurve(probs, t.y);
        const auc = aucFromCurve(pr.recall, pr.precision);
        importance.push({ feature: f, delta: basePRAUC - auc });
    }
    
    importance.sort((a, b) => b.delta - a.delta);
    return { importance, basePRAUC };
}

// Segment metrics
function segmentMetrics(rows, probs, labels) {
    const segments = {
        Contract: Array.from(new Set(rows.map(r => r.Contract || 'Unknown'))),
        InternetService: Array.from(new Set(rows.map(r => r.InternetService || 'Unknown'))),
        SeniorCitizen: Array.from(new Set(rows.map(r => String(r.SeniorCitizen || '0'))))
    };
    
    const results = [];
    
    function segmentAUC(subProbs, subLabels) {
        const roc = rocCurve(subProbs, subLabels);
        const pairs = roc.fpr.map((x, i) => [x, roc.tpr[i]]).sort((a, b) => a[0] - b[0]);
        return aucFromCurve(pairs.map(p => p[0]), pairs.map(p => p[1]));
    }
    
    function segmentECE(subProbs, subLabels) {
        return calibrationBins(subProbs, subLabels, 10).ece;
    }
    
    for (const [key, values] of Object.entries(segments)) {
        for (const v of values) {
            const idx = rows.map((r, i) => ({ match: String(r[key] || 'Unknown') === String(v), i }))
                .filter(o => o.match).map(o => o.i);
            const subProbs = idx.map(i => probs[i]);
            const subLabels = idx.map(i => labels[i]);
            if (subProbs.length < 10) continue;
            results.push({
                segment: `${key}=${v}`,
                size: subProbs.length,
                auc: segmentAUC(subProbs, subLabels),
                ece: segmentECE(subProbs, subLabels)
            });
        }
    }
    
    return results.sort((a, b) => a.segment.localeCompare(b.segment));
}

// Rendering
async function renderCurves(name, probs, labels, tab = 'Evaluation') {
    const roc = rocCurve(probs, labels);
    const rocPairs = roc.fpr.map((x, i) => ({ x, y: roc.tpr[i] })).sort((a, b) => a.x - b.x);
    const rocAUC = aucFromCurve(rocPairs.map(p => p.x), rocPairs.map(p => p.y));
    
    await tfvis.render.linechart({ name: `${name}: ROC (AUC=${rocAUC.toFixed(3)})`, tab }, { values: rocPairs, series: ['ROC'] }, { xLabel: 'FPR', yLabel: 'TPR', width: 400, height: 300 });

    const pr = prCurve(probs, labels);
    const prPairs = pr.recall.map((x, i) => ({ x, y: pr.precision[i] })).sort((a, b) => a.x - b.x);
    const prAUC = aucFromCurve(prPairs.map(p => p.x), prPairs.map(p => p.y));
    
    await tfvis.render.linechart({ name: `${name}: PR (AUC=${prAUC.toFixed(3)})`, tab }, { values: prPairs, series: ['PR'] }, { xLabel: 'Recall', yLabel: 'Precision', width: 400, height: 300 });

    return { rocAUC, prAUC };
}

async function renderCalibration(name, probs, labels, tab = 'Evaluation') {
    const { bins, ece } = calibrationBins(probs, labels, 10);
    const points = bins.map(b => ({ x: b.avgP, y: b.avgY }));
    
    await tfvis.render.scatterplot({ name: `${name}: Calibration (ECE=${ece.toFixed(3)})`, tab },
        { values: points, series: ['bins'] },
        { xLabel: 'Predicted', yLabel: 'Observed', xAxisDomain: [0, 1], yAxisDomain: [0, 1], width: 400, height: 300 });
    
    const bs = brierScore(probs, labels);
    appendLog(`${name} Calibration: Brier=${bs.toFixed(4)}, ECE=${ece.toFixed(4)}.`);
    return { brier: bs, ece };
}

async function renderDecileLift(name, probs, labels, tab = 'Evaluation') {
    const { rates, cumCapture } = decileLift(probs, labels);
    
    await tfvis.render.barchart({ name: `${name}: Decile churn rate`, tab }, { values: rates, series: ['rate'] }, { xLabel: 'Decile (High->Low risk)', yLabel: 'Churn rate', height: 300 });
    await tfvis.render.linechart({ name: `${name}: Cumulative capture`, tab }, { values: cumCapture.map(p => ({ x: parseInt(p.x.replace('D', '')), y: p.y })), series: ['capture'] }, { xLabel: 'Decile', yLabel: 'Cumulative capture', height: 300 });
}

function renderMetricsTable(containerId, metrics) {
    const el = document.getElementById(containerId);
    const fmt = x => typeof x === 'number' ? x.toFixed(4) : x;
    
    el.innerHTML = `
        <div><b>Model Comparison (Test)</b></div>
        <div>ROC AUC — Baseline: ${fmt(metrics.baseline.rocAUC)}, NN:
