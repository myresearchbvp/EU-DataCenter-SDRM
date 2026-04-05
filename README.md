# Supplementary Material – Spatial Decision Robustness Mapping for Geo-Distributed Data Center Siting in Europe

[![Open upstream notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINK_NOTEBOOK_1)
[![Open dashboard notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINK_NOTEBOOK_2)

## About this repository

This repository contains the reproducible analytical workflow used in our study on spatial decision support for geo-distributed data center siting across 24 selected European NUTS-2 regions.

The repository includes:
- an upstream notebook for Eurostat extraction and validation
- a frozen dashboard notebook for deterministic and stochastic MCDA analysis
- the validated 2022 decision matrix used in the paper
- a minimal environment specification

The study is based on five Eurostat criteria:
- ICT specialists share in total employment
- hourly labour cost
- renewable electricity share
- non-household electricity price
- population density

## Repository structure

- `notebooks/01_build_eurostat_decision_matrix.ipynb`  
  Builds and validates the Eurostat decision matrix.

- `notebooks/02_spatial_decision_robustness_dashboard.ipynb`  
  Runs the deterministic and stochastic analysis and generates tables and figures.

- `data/Decision_Matrix_24x5_Eurostat_2022.csv`  
  Bundled reference matrix used in the article.

- `requirements.txt`  
  Python package list.

- `docs/ENVIRONMENT.md`  
  Short runtime note.

## How to run

### Option A – direct replication of the published experiment
1. Open the dashboard notebook in Colab.
2. Run all cells.
3. Use the bundled `Decision_Matrix_24x5_Eurostat_2022.csv`.
4. Inspect the tables, figures and map outputs.

### Option B – rebuild the matrix first
1. Open the upstream extraction notebook in Colab.
2. Run the extraction for a year with complete coverage under the study filters.
3. Export the generated CSV.
4. Load that CSV into the dashboard notebook.

## Reproducibility note

The repository is based entirely on public Eurostat data and a fixed notebook workflow.  
The bundled 2022 matrix is provided for immediate replication of the reported results.

## Peer-review note

This repository is provided as supplementary material for reproducibility and review purposes.
