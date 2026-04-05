# Supplementary Material - Spatial Decision Robustness Mapping for geo-distributed data center siting in Europe

## Open in Colab

**Notebook 1 – Build and validate the Eurostat decision matrix**  
[![Open Notebook 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/myresearchbvp/EU-DataCenter-SDRM/blob/main/notebooks/01_build_eurostat_decision_matrix.ipynb)

**Notebook 2 – Run the spatial decision robustness dashboard**  
[![Open Notebook 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/myresearchbvp/EU-DataCenter-SDRM/blob/main/notebooks/02_spatial_decision_robustness_dashboard.ipynb)

## About this repository

This repository contains the reproducible analytical workflow used in our study on spatial decision support for geo-distributed data center siting across 24 selected European NUTS-2 regions.

The workflow is organized in two notebooks:
- **Notebook 1** builds and validates the Eurostat decision matrix
- **Notebook 2** runs the deterministic and stochastic MCDA analysis and generates the tables, charts, and spatial figures used in the study

The study is based on five Eurostat criteria:
- ICT specialists share in total employment
- hourly labour cost
- renewable electricity share
- non-household electricity price
- population density

## Repository structure

- `notebooks/01_build_eurostat_decision_matrix.ipynb`  
  Upstream notebook for Eurostat extraction, filtering, completeness checks, and decision matrix construction.

- `notebooks/02_spatial_decision_robustness_dashboard.ipynb`  
  Downstream notebook for preprocessing, deterministic MCDA, stochastic robustness analysis, local Dirichlet sensitivity, and static spatial visualization.

- `data/Decision_Matrix_24x5_Eurostat_2022.csv`  
  Validated reference matrix used for the 2022 experiment reported in the article.

- `figures/`  
  Publication figures generated from the 2022 experiment by the downstream notebook.

- `requirements.txt`  
  Python package requirements.

- `docs/ENVIRONMENT.md`  
  Short execution and environment notes.

## How to run

### Option A – direct replication of the published 2022 experiment
1. Open **Notebook 2** in Colab.
2. Run all cells.
3. Upload or use the bundled file `Decision_Matrix_24x5_Eurostat_2022.csv`.
4. Reproduce the analytical tables, charts, and spatial outputs reported in the study.

### Option B – rebuild the decision matrix first
1. Open **Notebook 1** in Colab.
2. Run the extraction and validation workflow for a year with complete coverage under the study filters.
3. Export the generated decision matrix as CSV.
4. Open **Notebook 2** in Colab and load that CSV for analysis.

## Reproducibility note

The repository is based entirely on public Eurostat data and a fixed notebook workflow.  
The bundled 2022 matrix is provided for immediate replication of the reported results, while the upstream notebook allows validated reconstruction for other years when complete five-criterion coverage is available under the applied filters.

## Supplementary material note

This repository is provided as supplementary material for transparency, reproducibility, and review.
