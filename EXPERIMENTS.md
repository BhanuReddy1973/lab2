# Lab 2 - Experiment Checklist

**Student:** Bhanu Reddy  
**Roll Number:** 2022bcd0026

## Experiments to Run

### ✅ Experiment LR-1 (DONE)
- Model: LinearRegression
- Test Size: 0.20
- Scaling: False
- Feature Selection: None
- Commit: "Experiment LR-1: LinearRegression, no scaling, all features, 80/20 split"

---

### 📝 Experiment LR-2
**Configuration:**
```python
MODEL_TYPE = 'LinearRegression'
TEST_SIZE = 0.30
USE_SCALING = True
FEATURE_SELECTION = None
```
**Commit Message:** `"Experiment LR-2: LinearRegression, with scaling, all features, 70/30 split"`

---

### 📝 Experiment LR-3
**Configuration:**
```python
MODEL_TYPE = 'LinearRegression'
TEST_SIZE = 0.20
USE_SCALING = False
FEATURE_SELECTION = 8
```
**Commit Message:** `"Experiment LR-3: LinearRegression, no scaling, top 8 features, 80/20 split"`

---

### 📝 Experiment LR-4
**Configuration:**
```python
MODEL_TYPE = 'LinearRegression'
TEST_SIZE = 0.25
USE_SCALING = True
FEATURE_SELECTION = 6
```
**Commit Message:** `"Experiment LR-4: LinearRegression, with scaling, top 6 features, 75/25 split"`

---

### 📝 Experiment RF-1
**Configuration:**
```python
MODEL_TYPE = 'RandomForest'
TEST_SIZE = 0.20
USE_SCALING = False
FEATURE_SELECTION = None
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 10
```
**Commit Message:** `"Experiment RF-1: RandomForest, 50 trees, depth=10, all features, 80/20 split"`

---

### 📝 Experiment RF-2
**Configuration:**
```python
MODEL_TYPE = 'RandomForest'
TEST_SIZE = 0.30
USE_SCALING = False
FEATURE_SELECTION = None
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
```
**Commit Message:** `"Experiment RF-2: RandomForest, 100 trees, depth=15, all features, 70/30 split"`

---

### 📝 Experiment RF-3
**Configuration:**
```python
MODEL_TYPE = 'RandomForest'
TEST_SIZE = 0.20
USE_SCALING = False
FEATURE_SELECTION = 8
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
```
**Commit Message:** `"Experiment RF-3: RandomForest, 100 trees, no depth limit, top 8 features, 80/20 split"`

---

### 📝 Experiment RF-4
**Configuration:**
```python
MODEL_TYPE = 'RandomForest'
TEST_SIZE = 0.25
USE_SCALING = False
FEATURE_SELECTION = 6
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
```
**Commit Message:** `"Experiment RF-4: RandomForest, 200 trees, depth=20, top 6 features, 75/25 split"`

---

## 📸 Required Screenshots for PDF

After all experiments:

1. **Main Workflow Page** - All 8 successful runs listed
2. **Job Summary** - Any one run showing student name, roll number, and metrics
3. **Artifacts Page** - Show downloadable model.pkl and results.json
4. **Individual Run Details** - Expand one run to show the full log

## 📊 Results Comparison

After all runs, create a table comparing all experiments.
