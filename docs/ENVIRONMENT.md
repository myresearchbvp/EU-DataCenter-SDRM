# Environment Notes

## Recommended environment

The workflow was developed and tested in Google Colab and standard Python notebook environments.

## Main dependencies

The repository uses the packages listed in `requirements.txt`.

## Execution logic

Two usage paths are supported:

### Option A: full workflow
1. Run `01_build_eurostat_decision_matrix.ipynb`
2. Export or save the validated CSV decision matrix
3. Run `02_spatial_decision_robustness_dashboard.ipynb`

### Option B: direct replication from the validated matrix
1. Open `02_spatial_decision_robustness_dashboard.ipynb`
2. Upload `data/Decision_Matrix_24x5_Eurostat_2022.csv`
3. Run the full notebook

## Notes

- The 2022 matrix is included as a ready-to-use validated input
- Static figures are generated at publication quality resolution
- The study focuses on complete-coverage years under the applied Eurostat filters