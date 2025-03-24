import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, 
                             StackingClassifier, 
                             VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# 1. 数据加载与预处理
def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_columns] = df[zero_columns].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    df[zero_columns] = imputer.fit_transform(df[zero_columns])
    
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['Glucose_Insulin'] = df['Glucose'] * df['Insulin']
    
    return df

# 2. 特征选择与数据分割
def feature_selection_and_split(df):
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    plt.figure(figsize=(10,6))
    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feat_importances.nlargest(8).plot(kind='barh')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    selected_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Insulin']
    X = X[selected_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, 'scaler.pkl')
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 模型训练与评估
def train_and_evaluate(X_train, X_test, y_train, y_test):
    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, max_iter=1000))
    ]
    
    meta_model = LogisticRegression(max_iter=1000)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, max_iter=1000),
        'Voting': VotingClassifier(estimators=base_models, voting='soft'),
        'Stacking': StackingClassifier(estimators=base_models, final_estimator=meta_model)
    }
    
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            metrics = {
                'Model': name,
                'Accuracy': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan,
                'AUC-ROC': np.nan,
                'CrossVal Score': np.nan
            }
            results.append(metrics)
            continue
        
        y_pred = model.predict(X_test)
        
        # 处理不同模型的概率输出
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
            'CrossVal Score': cross_val_score(model, X_train, y_train, cv=5).mean()
        }
        results.append(metrics)

        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{name.replace(" ", "_").lower()}_confusion_matrix.png')
    
    return models, pd.DataFrame(results)

# 4. 主流程
def main():
    df = load_and_preprocess("diabetes.csv")
    X_train, X_test, y_train, y_test = feature_selection_and_split(df)
    models, results_df = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("\nModel Evaluation Results:")
    print(results_df)
    
    plt.figure(figsize=(10,8))
    for name, model in models.items():
        if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
            y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig('roc_curve_comparison.png')

# UI界面部分
LANGUAGES = {
    "CN": {
        "title": "🩺 糖尿病风险评估系统",
        "desc": "**说明**：本系统基于机器学习模型预测糖尿病风险，用于辅助临床决策。建议结合其他临床检查结果综合判断。",
        "patient_info": "患者信息",
        "medical_id": "病历号",
        "age": "年龄",
        "gender": ["男性", "女性", "其他"],
        "pregnancies": "怀孕次数（女性）",
        "clinical_metrics": "📌 临床指标输入",
        "glucose": "血糖 (mg/dL)",
        "insulin": "胰岛素 (μU/mL)",
        "bmi": "BMI",
        "bp": "血压 (mmHg)",
        "skin": "皮褶厚度 (mm)",
        "pedigree": "遗传函数值",
        "risk_prob": "风险概率",
        "high_risk": "高风险：建议进行OGTT检测",
        "low_risk": "低风险：建议保持常规监测",
        "guide_title": "**参考指南**：",
        "guide_content": """- 高风险 (>60%)：安排糖化血红蛋白检测
- 中风险 (30-60%)：建议生活方式干预
- 低风险 (<30%)：常规年度筛查""",
        "metrics_desc": "📚 临床指标说明",
        "operation_guide": "🖥️ 系统操作指南"
    },
    "EN": {
        "title": "🩺 Diabetes Risk Assessment System",
        "desc": "**Note**: This system predicts diabetes risk using machine learning models for clinical decision support. Results should be interpreted in conjunction with other clinical findings.",
        "patient_info": "Patient Information",
        "medical_id": "Medical ID",
        "age": "Age",
        "gender": ["Male", "Female", "Other"],
        "pregnancies": "Pregnancies (Female)",
        "clinical_metrics": "📌 Clinical Metrics Input",
        "glucose": "Glucose (mg/dL)",
        "insulin": "Insulin (μU/mL)",
        "bmi": "BMI",
        "bp": "Blood Pressure (mmHg)",
        "skin": "Skin Thickness (mm)",
        "pedigree": "Diabetes Pedigree Function",
        "risk_prob": "Risk Probability",
        "high_risk": "High Risk: Recommend OGTT test",
        "low_risk": "Low Risk: Recommend routine monitoring",
        "guide_title": "**Clinical Guidelines**:",
        "guide_content": """- High Risk (>60%): Schedule HbA1c test
- Medium Risk (30-60%): Lifestyle intervention
- Low Risk (<30%): Annual screening""",
        "metrics_desc": "📚 Clinical Metrics Description",
        "operation_guide": "🖥️ User Manual"
    }
}

if 'language' not in st.session_state:
    st.session_state.language = "CN"

st.set_page_config(
    page_title="Diabetes Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    model = joblib.load('voting_model.pkl')
except FileNotFoundError:
    st.error("未找到 'voting_model.pkl' 文件，请先运行主流程训练并保存模型。")
    model = None
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("未找到 'scaler.pkl' 文件，请先运行主流程训练并保存模型。")
    scaler = None

with st.sidebar:
    lang = st.radio("Language/语言", ["CN", "EN"], index=0 if st.session_state.language == "CN" else 1)
    st.session_state.language = lang
    
    text = LANGUAGES[lang]
    st.header(text["patient_info"])
    patient_id = st.text_input(text["medical_id"], max_chars=10)
    age = st.number_input(text["age"], 18, 100, 35)
    gender = st.radio(text["gender"][0], text["gender"])
    pregnancies = st.number_input(text["pregnancies"], 0, 20, 0)

text = LANGUAGES[lang]
st.title(text["title"])
st.markdown(text["desc"])

with st.expander(text["clinical_metrics"]):
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.slider(text["glucose"], 50, 300, 120)
        insulin = st.number_input(text["insulin"], 0, 1000, 80)
        bmi = st.number_input(text["bmi"], 10.0, 60.0, 25.0, format="%.1f")
    with col2:
        bp = st.number_input(text["bp"], 50, 200, 80)
        skin_thickness = st.number_input(text["skin"], 0, 100, 20)
        pedigree = st.number_input(text["pedigree"], 0.0, 2.5, 0.5, step=0.01)

if st.button("进行风险评估" if lang == "CN" else "Start Risk Assessment"):
    if model is None or scaler is None:
        st.error("模型或scaler加载失败，请先运行主流程训练并保存模型。")
    else:
        features = pd.DataFrame([[glucose, bmi, age, pedigree, insulin]],
                                columns=['Glucose', 'BMI', 'Age', 
                                         'DiabetesPedigreeFunction', 'Insulin'])
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)[0][1]
        
        st.subheader("评估结果" if lang == "CN" else "Assessment Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(text["risk_prob"], f"{proba * 100:.1f}%")
            if prediction[0] == 1:
                st.error(text["high_risk"])
            else:
                st.success(text["low_risk"])
        with col2:
            st.markdown(f"{text['guide_title']}\n{text['guide_content']}")

with st.expander(text["metrics_desc"]):
    st.markdown("""
    ### 指标定义标准
    1. **血糖 (Glucose)**  
       空腹血浆葡萄糖检测标准：  
       - 正常：<100 mg/dL  
       - 糖尿病前期：100-125 mg/dL  
       - 糖尿病：≥126 mg/dL

    2. **BMI 分类**  
       WHO标准：  
       - 体重不足：<18.5  
       - 正常：18.5-24.9  
       - 超重：25-29.9  
       - 肥胖：≥30

    3. **血压标准**  
       - 正常：<120/80 mmHg  
       - 高血压前期：120-139/80-89 mmHg  
       - 高血压：≥140/90 mmHg
    """ if lang == "CN" else """
    ### Clinical Metrics Standards
    1. **Glucose**  
       Fasting Plasma Glucose:  
       - Normal: <100 mg/dL  
       - Prediabetes: 100-125 mg/dL  
       - Diabetes: ≥126 mg/dL

    2. **BMI Classification**  
       WHO Standards:  
       - Underweight: <18.5  
       - Normal: 18.5-24.9  
       - Overweight: 25-29.9  
       - Obese: ≥30

    3. **Blood Pressure**  
       - Normal: <120/80 mmHg  
       - Elevated: 120-139/80-89 mmHg  
       - Hypertension: ≥140/90 mmHg
    """)

with st.expander(text["operation_guide"]):
    st.markdown("""
    ### 使用步骤
    1. 在左侧栏输入患者基本信息
    2. 填写全部临床指标参数
    3. 点击"进行风险评估"按钮
    4. 查看并解读评估结果
    
    ### 注意事项
    - 所有输入字段均为必填项
    - 数值范围已根据临床标准设置限制
    - 结果应结合临床判断使用
    """ if lang == "CN" else """
    ### User Guide
    1. Input patient information in sidebar
    2. Fill all clinical metrics
    3. Click "Start Risk Assessment" button
    4. View and interpret results
    
    ### Notes
    - All fields are required
    - Value ranges are set based on clinical standards
    - Results should be used with clinical judgment
    """)

if __name__ == "__main__":
    main()