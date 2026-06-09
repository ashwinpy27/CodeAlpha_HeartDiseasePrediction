# ── CELL 1: Import Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
# ── CELL 2: Load & Preview Data ───────────────────────────────────
df = pd.read_csv('heart.csv')

print("Shape:", df.shape)          # rows x columns
print("\nFirst 5 rows:")
display(df.head())

print("\nColumn Info:")
df.info()

print("\nBasic Statistics:")
display(df.describe())
# ── CELL 3: Check for Missing Values & Target Distribution ────────
print("Missing values per column:")
print(df.isnull().sum())

print("\nTarget value counts (0 = No Disease, 1 = Disease):")
print(df['target'].value_counts())

# Visualize class balance
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Disease vs No Disease Count')
plt.xticks([0,1], ['No Disease', 'Disease'])
plt.show()