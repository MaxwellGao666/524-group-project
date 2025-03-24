```markdown
# Diabetes Risk Assessment System 🩺

A machine learning-based web application for diabetes risk prediction, featuring ensemble models and a multilingual (CN/EN) Streamlit interface.

## Features ✨
- **Data Preprocessing**: Handles missing values, creates feature interactions
- **Feature Engineering**: BMI×Age and Glucose×Insulin interaction terms
- **Model Training**:
  - 6 base models (Logistic Regression, Random Forest, XGBoost, etc.)
  - 2 ensemble methods (Voting & Stacking)
  - Comprehensive metrics tracking (Accuracy, AUC-ROC, F1-score, etc.)
- **Interpretability**: Feature importance visualization, model comparison ROC curves
- **Clinical UI**:
  - Bilingual interface (Chinese/English)
  - Interactive input with clinical validation
  - Risk stratification guidelines

## Requirements 📦
- Python 3.7+
- Required packages:
  ```bash
  pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib
  ```

## Dataset Preparation 📁
1. Place your diabetes dataset named `diabetes.csv` in the project root
2. Expected columns (with units):
   - Pregnancies, Glucose(mg/dL), BloodPressure(mmHg), SkinThickness(mm)
   - Insulin(μU/mL), BMI(kg/m²), DiabetesPedigreeFunction, Age(years), Outcome

## Usage 🚀
1. **Train Models**:
   ```bash
   python diabetes_detection.py
   ```
   This will:
   - Generate feature importance plot
   - Train and save models (.pkl files)
   - Create evaluation reports (CSV and PNG)

2. **Run Web App**:
   ```bash
   streamlit run diabetes_detection.py
   ```
   - Select language in sidebar
   - Fill patient information and clinical metrics
   - Click assessment button for risk prediction

## Key Files 🔑
- `scaler.pkl`: Saved feature scaler
- `*_model.pkl`: Serialized model files
- `feature_importance.png`: Feature ranking
- `roc_curve_comparison.png`: Model performance comparison
- `model_evaluation_results.csv`: Detailed metrics table

## Clinical Guidelines 📈
| Risk Level | Probability | Recommendation |
|------------|-------------|----------------|
| Low        | <30%        | Annual screening |
| Medium     | 30-60%      | Lifestyle intervention |
| High       | >60%        | HbA1c/OGTT testing |

## Input Specifications 🔍
| Parameter | Range | Clinical Standard |
|-----------|-------|-------------------|
| Glucose | 50-300 mg/dL | Normal: <100 mg/dL |
| BMI | 10.0-60.0 | Obese: ≥30 |
| BP | 50-200 mmHg | Hypertension: ≥140/90 |

## Troubleshooting 🛠️
- If seeing "model not found" errors:
  1. Ensure you've run `python diabetes_detection.py` first
  2. Verify generated .pkl files exist in project directory
- For Streamlit issues:
  - Check port conflicts with `--server.port` flag
  - Use Chrome/Firefox for best compatibility

## Disclaimer ⚠️
This system is for **clinical decision support** only. Always combine predictions with:
- Patient medical history
- Physical examination findings
- Laboratory test results

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

This README provides comprehensive guidance while maintaining clinical relevance. It highlights both technical implementation details and medical application considerations.
