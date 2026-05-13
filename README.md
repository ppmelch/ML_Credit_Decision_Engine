# Final Credit Model
### Credit Models project

**Authors:**
- José Armando Melchor Soto — 745697
- Rolando Fortanell Canedo — 744872
- David Campos Ambriz — 7444435
 
**Course:** Credit Models

**Institution:** ITESO Universidad Jesuita de Guadalajara

**Date:** May 13, 2026
 
---
 
## Table of Contents

---
 
## Overview
 

---
 
## Architecture
 
### Project Structure
 
```mermaid
flowchart TB

    ROOT["Final_Credit_Model/"]

    %% =========================
    %% ROOT
    %% =========================

    ROOT --> DATA[data/]
    ROOT --> BACKEND[backend/]
    ROOT --> FRONTEND[frontend/]
    ROOT --> NOTEBOOKS[notebooks/]
    ROOT --> DOCS[docs/]

    ROOT --> MAIN[main.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> README[README.md]

    %% =========================
    %% DATA
    %% =========================

    DATA --> DATASET[dataset_modelado_final.csv]
    DATA --> RESULTS[results.csv]

    %% =========================
    %% BACKEND
    %% =========================

    BACKEND --> SRC

    subgraph SRC[src/]

        subgraph DATAR[data/]
            dp[data_preparation.py]
            ds[data_splitter.py]
        end

        subgraph MODELING[modeling/]
            bm[base_model.py]
            bl[business_logic.py]
            cm[classification_model.py]
            cfg[config.py]
            me[model_evaluation.py]
            mod[model.py]
            rc[risk_calculator.py]
            geo[geospatial_risk.py]
            th[threshold_optimization.py]
            er[experiment_runner.py]
            opt[optuna_optimizer.py]
            ml[model_loader.py]
        end

        subgraph VIS[visualization/]
            vz[viz.py]
        end

        subgraph UT[utils/]
            pr[prints.py]
            ut[utils.py]
        end

        subgraph MODS[models/]
            rf[random_forest.pkl]
        end

    end

    %% =========================
    %% SCRIPTS
    %% =========================

    BACKEND --> SCRIPTS[Scripts/]
    SCRIPTS --> EXT[External Data Source Scripts]

    %% =========================
    %% FRONTEND
    %% =========================

    subgraph FRONTEND[frontend/]

        idx[index.html]
        svg[jalisco.svg]

        subgraph FDATA[data/]
            risk[risk_data.json]
            jal[Jalisco.json]
            dash[dashboard_data.json]
        end

        subgraph JS[js/]
            app[app.js]
        end

        subgraph CSS[css/]
            cssfile[style.css]
        end

    end

    %% =========================
    %% NOTEBOOKS
    %% =========================

    subgraph NOTEBOOKS[notebooks/]
        da[data_analysis.ipynb]
        fa[feature_analysis.ipynb]
        dev[Credit Model Development.ipynb]
    end

    %% =========================
    %% DOCS
    %% =========================

    subgraph DOCS[docs/]
        pdf[Final_Credit_Model.pdf]
    end
```
 
### Functional Architecture
 
```mermaid

```
 
### OOP Architecture
 
```mermaid


```
 

 

---
 
## Methodology

 

### Model
 
 
---
 
## Results
 
### Model Performance

### Key Findings
 


---
 
## Limitations & Assumptions
 
### Future Improvements
 

---
 
## Conclusions
 

---
 
## Installation
 
```bash
# 1. Clone the repository
git clone https://github.com/ppmelch/Final_Credit_Model.git
cd Credit_Model
 
# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
---
 
## Usage
 
```bash
python main.py
```
 
---
 
## Output
 

---
 
## Documentation
 
The full project report is available at:
 
- [Credit Model Report](docs/Final_Credit_Model.pdf)
 
---
 
## License
 
This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
