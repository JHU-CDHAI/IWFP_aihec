# Indigenous Wholistic Factors Project (IWFP)

**Lead Author**: Valentín Quiroz Sierra, PhD, MSW <br>
**Tribal Affiliation**: Yo'eme <br>
**Academic Affiliation**: Johns Hopkins Bloomberg School of Public Health, Center for Indigenous Health <br>
**Last Updated**: July 2025

## Overview

The Indigenous Wholistic Factors Project (IWFP) is a community-centered, data-driven initiative aimed at advancing culturally grounded understandings of suicide risk and protective factors among Native American youth in California. This repository supports the manuscript:

**"Predicting Suicidal Ideation Among Native American High Schoolers in California"**  

This study uses data from the 2019–2020 California Healthy Kids Survey (CHKS) and is guided by Indigenous Wholistic Theory and the Indigenous Computational Approach. A LASSO logistic regression model was used to identify predictors of past-year suicidal ideation among Native American high school students.

## Project Goals

- Advance culturally grounded knowledge of risk and protective factors to support suicide prevention among Native American youth in California
- Apply community-centered machine learning methods through the Indigenous Computational Approach to support ethical, culturally grounded public health research
- Promote open science practices that align with Indigenous Data Sovereignty principles

## Repository Contents

```
/data/                                # Placeholder for analysis-ready data (not included)
/raw_data/                            # Placeholder for raw CHKS data (not included)
/documentation/
  ├── calschls-2019-20-crosswalk.pdf  # Survey item crosswalk
  ├── cschls_researchsum2018.pdf      # CHKS codebook and methodology
  └── appendixa.pdf                   # Construction of predictor variables

/output/
  ├── appendixb.pdf                   # School-level clustering sensitivity analysis
  └── appendixc.pdf                   # Missing data imputation sensitivity analysis

/paper/                               # Final manuscript (to be uploaded upon acceptance)

/scripts/
  ├── primary_analysis.do             # LASSO model and predictors (Stata)
  ├── sensitivity_clustered.do        # Clustered schools sensitivity analysis (Stata)
  ├── sensitivity_imputed.do          # Missing data imputation analysis (Stata)
  ├── primary_analysis.py             # LASSO model and predictors (Python)
  ├── sensitivity_clustered.py        # Clustered schools sensitivity analysis (Python)
  ├── sensitivity_imputed.py          # Missing data imputation analysis (Python)
  ├── test_data_loading.py            # Data loading diagnostic tool (Python)
  ├── requirements.txt                # Python dependencies
  └── README_python.md               # Python scripts documentation

README.md                             # Main project documentation
```

## Software Requirements

### Stata Version
Analyses require **Stata/SE 17.0 or higher** and the following user-written packages:

- `lasso2` – Penalized regression
- `cvauroc` – AUC estimation
- `calibrationbelt` – Model calibration assessment
- `moremata` – Matrix and math utilities

Install them in Stata using:

```stata
ssc install lasso2
ssc install cvauroc
ssc install calibrationbelt
ssc install moremata
```

### Python Version
Alternatively, you can run the analyses using **Python 3.7+** with the following packages:

- `pandas` – Data manipulation and analysis
- `numpy` – Numerical computing
- `scikit-learn` – Machine learning (includes LASSO regression)
- `matplotlib` – Plotting and visualization
- `seaborn` – Statistical data visualization
- `scipy` – Statistical functions

Install Python dependencies using:

```bash
cd scripts/
pip install -r requirements.txt
```

Quick start with Python:

```bash
# Test data loading first
python test_data_loading.py

# Run primary analysis
python primary_analysis.py
```

For detailed Python usage instructions, see `scripts/README_python.md`.

## Data Availability

The California Healthy Kids Survey (CHKS) data are **not publicly shared** in this repository due to privacy protections for youth and school districts. Researchers may request access from WestEd or local education authorities under subject of IRB approval.

## OSF Link

All study materials — including supplemental documentation, appendices, and pre-registration — are archived at the Open Science Framework (OSF): 
🔗 [https://osf.io/4dpwt/](https://osf.io/4dpwt/)

## Contributing

This repository is intended to support transparent, ethical, and collaborative computational research in Indigenous health. If you'd like to adapt or contribute to this project:

1. Fork the repository and submit a pull request
2. Include a description of your proposed changes
3. Ensure your contributions respect Indigenous research ethics and Indigenous Data Sovereignty principles

## Citation

If you use or adapt any portion of this repository, please cite:

**Sierra, V. Q.** (2025). *Predicting Suicidal Ideation Among Native American High Schoolers in California*. Archives of Suicide Research, 1–18. [https://doi.org/10.1080/13811118.2025.2490154](https://doi.org/10.1080/13811118.2025.2490154).


## Contact

For questions, collaboration opportunities, or data requests, contact:  
**Valentín Quiroz Sierra, PhD, MSW** (Yo'eme) <br>
Postdoctoral Fellow <br>
Johns Hopkins Bloomberg School of Public Health <br>
Center for Indigenous Health <br>
[vsierra4@jh.edu](mailto:vsierra4@jh.edu)
