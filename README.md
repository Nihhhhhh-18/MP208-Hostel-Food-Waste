# MP208 - Hostel Mess Food Waste Prediction

## Project Overview
This repository contains the end-to-end MLOps pipeline for predicting hostel mess food waste (`waste_kg`) for Kochi University Hostel Office, developed as part of SCSE3040 MLOps Mini Project.

- **Group ID:** G182
- **Problem ID:** MP208
- **Batch:** 85
- **Faculty:** Richa Sharma

## Team Members
- **Nihal Devarashetty** - S24CSEU2236
- **Arth Agrawal** - S24CSEU2226
- **Shivam Verma** - S24CSEU2256

## Repository Layout
```
├── README.md                          # Group ID, Problem ID, Member details
├── requirements.txt                   # Pinned project dependencies
├── params.yaml                        # Central configuration read via yaml.safe_load
├── data/
│   └── raw/
│       ├── train.csv.dvc              # DVC tracking for messy training data
│       ├── later.csv.dvc              # DVC tracking for subsequent operational data
│       └── holdout_features.csv       # Test features for final evaluation
├── src/
│   └── food_waste/                    # Modular source package
│       ├── __init__.py
│       ├── config.py                  # YAML configuration loader
│       └── ...
├── app/
│   └── main.py                        # FastAPI prediction & health service
├── models/
│   └── model.joblib                   # Production champion model
├── reports/
│   └── findings.yaml                  # Structured audit findings & results
├── predictions/
│   └── holdout_predictions.csv        # Final 250 predictions
├── tests/
│   └── test_*.py                      # Pytest unit & TestClient integration tests
├── Dockerfile                         # Container build definition (python:3.12-slim)
├── .dockerignore                      # Docker ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml                     # Continuous integration pipeline on push
└── k8s/
    ├── deployment.yaml                # Kubernetes Deployment (3 replicas)
    └── service.yaml                   # Kubernetes Service (NodePort/ClusterIP)
```
