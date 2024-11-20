import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import skfuzzy as fuzz
import skfuzzy.membership as mf
import sklearn.metrics


# Reading Dataset

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

i = 0
for f in csv_files:
    person = pd.read_csv(f)
    if(i==0):
        df = person.mean(axis=0, skipna=True, level=None).to_frame().T
        i = 1
    else:
        df = pd.concat([df, person.mean(axis=0, skipna=True, level=None).to_frame().T], ignore_index=True)

X = df.drop(columns=['sirs', 'qsofa', 'sepsis_icd'])

# Output Selection

Y = df["sepsis_icd"]

# Feature Selection Based on Missing Values of Features

msno.matrix(X)

X = X.drop(columns=['fio2', 'ph', 'pco2', 'po2'])

# Feature Selection Based on Correlation Between Features

sns.heatmap(X.corr())

X = X.drop(columns=['bp_systolic', 'bp_diastolic', 'resp', 'temp', 'spo2', 'wbc', 'bun', 'bicarbonate', 'hemoglobin', 'hematocrit', 'potassium', 'chloride', 'age'])

# Variable Ranges

x_heart_rate = np.arange(0, 140.1, 0.1)
x_map = np.arange(0, 140.1, 0.1)
x_bilirubin = np.arange(0, 80.1, 0.1)
x_creatinine = np.arange(0, 11.6, 0.1)
x_lactate = np.arange(0, 14.1, 0.1)
x_platelets = np.arange(0, 750.1, 0.1)
x_gcs = np.arange(3, 16, 1)
y_risk = np.arange(0, 1.1, 0.1)

# Membership Functions

heart_rate_low = mf.trapmf(x_heart_rate, [0, 0, 0, 60])
heart_rate_mid = mf.trapmf(x_heart_rate, [0, 60, 100, 140])
heart_rate_high = mf.trapmf(x_heart_rate, [100, 140, 140, 140])

map_low = mf.trapmf(x_map, [0, 0, 0, 60])
map_mid = mf.trapmf(x_map, [0, 60, 100, 140])
map_high = mf.trapmf(x_map, [100, 140, 140, 140])

bilirubin_mid = mf.trapmf(x_bilirubin, [0, 5.1, 20.5, 68.4])
bilirubin_high = mf.trapmf(x_bilirubin, [20.5, 68.4, 80, 80])

creatinine_mid = mf.trapmf(x_creatinine, [0, 6.2, 11.5, 11.5])
creatinine_high = mf.trapmf(x_creatinine, [4.4, 11.5, 11.5, 11.5])

lactate_mid = mf.trapmf(x_lactate, [0, 0.5, 2, 4])
lactate_high = mf.trapmf(x_lactate, [2, 4, 13, 13])

platelets_low = mf.trapmf(x_platelets, [0, 0, 100, 150])
platelets_mid = mf.trapmf(x_platelets, [100, 150, 450, 750])

gcs_very_low = mf.trapmf(x_gcs, [3, 3, 8, 9])
gcs_low = mf.trapmf(x_gcs, [8, 9, 12, 13])
gcs_mid = mf.trapmf(x_gcs, [12, 13, 15, 15])

risk_not = mf.trapmf(y_risk, [0, 0, 0.2, 0.3])
risk_low = mf.trapmf(y_risk, [0.2, 0.3, 0.5, 0.6])
risk_mid = mf.trapmf(y_risk, [0.5, 0.6, 0.8, 0.9])
risk_high = mf.trapmf(y_risk, [0.8, 0.9, 1, 1])

fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(nrows = 8, figsize =(10, 25))

ax0.plot(x_heart_rate, heart_rate_low, label='Low')
ax0.plot(x_heart_rate, heart_rate_mid, label='Normal')
ax0.plot(x_heart_rate, heart_rate_high, label='High')
ax0.set_title('Heart Rate')
ax0.legend()

ax1.plot(x_map, map_low, label='Low')
ax1.plot(x_map, map_mid, label='Normal')
ax1.plot(x_map, map_high, label='High')
ax1.set_title('Map')
ax1.legend()

ax2.plot(x_bilirubin, bilirubin_mid, label='Normal')
ax2.plot(x_bilirubin, bilirubin_high, label='High')
ax2.set_title('Bilirubin')
ax2.legend()

ax3.plot(x_creatinine, creatinine_mid, label='Normal')
ax3.plot(x_creatinine, creatinine_high, label='High')
ax3.set_title('Creatinine')
ax3.legend()

ax4.plot(x_lactate, lactate_mid, label='Normal')
ax4.plot(x_lactate, lactate_high, label='High')
ax4.set_title('Lactate')
ax4.legend()

ax5.plot(x_platelets, platelets_low, label='Low')
ax5.plot(x_platelets, platelets_mid, label='Normal')
ax5.set_title('Platelets')
ax5.legend()

ax6.plot(x_gcs, gcs_very_low, label='Very Low')
ax6.plot(x_gcs, gcs_low, label='Low')
ax6.plot(x_gcs, gcs_mid, label='Normal')
ax6.set_title('GCS')
ax6.legend()

ax7.plot(y_risk, risk_not, label='Not')
ax7.plot(y_risk, risk_low, label='Low')
ax7.plot(y_risk, risk_mid, label='High')
ax7.plot(y_risk, risk_high, label='Very High')
ax7.set_title('Risk')
ax7.legend()

plt.tight_layout()

# Fuzzy Inference System

def fuzzy_inference_system(X, Y):
    
    result_array = []
    
    for i in range(X.shape[0]):
        
        input_heart_rate = X.loc[i, 'heart_rate']
        input_map = X.loc[i, 'map']
        input_bilirubin = X.loc[i, 'bilirubin']
        input_creatinine = X.loc[i, 'creatinine']
        input_lactate = X.loc[i, 'lactate']
        input_platelets = X.loc[i, 'platelets']
        input_gcs = X.loc[i, 'gcs']
        
        # Fuzzification

        heart_rate_fit_low = fuzz.interp_membership(x_heart_rate, heart_rate_low, input_heart_rate)
        heart_rate_fit_mid = fuzz.interp_membership(x_heart_rate, heart_rate_mid, input_heart_rate)
        heart_rate_fit_high = fuzz.interp_membership(x_heart_rate, heart_rate_high, input_heart_rate)
        
        map_fit_low = fuzz.interp_membership(x_map, map_low, input_map)
        map_fit_mid = fuzz.interp_membership(x_map, map_mid, input_map)
        map_fit_high = fuzz.interp_membership(x_map, map_high, input_map)

        bilirubin_fit_mid = fuzz.interp_membership(x_bilirubin, bilirubin_mid, input_bilirubin)
        bilirubin_fit_high = fuzz.interp_membership(x_bilirubin, bilirubin_high, input_bilirubin)

        creatinine_fit_mid = fuzz.interp_membership(x_creatinine, creatinine_mid, input_creatinine)
        creatinine_fit_high = fuzz.interp_membership(x_creatinine, creatinine_high, input_creatinine)

        lactate_fit_mid = fuzz.interp_membership(x_lactate, lactate_mid, input_lactate)
        lactate_fit_high = fuzz.interp_membership(x_lactate, lactate_high, input_lactate)

        platelets_fit_low = fuzz.interp_membership(x_platelets, platelets_low, input_platelets)
        platelets_fit_mid = fuzz.interp_membership(x_platelets, platelets_mid, input_platelets)

        gcs_fit_very_low = fuzz.interp_membership(x_gcs, gcs_very_low, input_gcs)
        gcs_fit_low = fuzz.interp_membership(x_gcs, gcs_low, input_gcs)
        gcs_fit_mid = fuzz.interp_membership(x_gcs, gcs_mid, input_gcs)
        
        # Rules
        
        rule0 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(heart_rate_fit_mid, map_fit_mid), bilirubin_fit_mid), creatinine_fit_mid), lactate_fit_mid), platelets_fit_mid), gcs_fit_mid), risk_not)
        rule1 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(heart_rate_fit_low, map_fit_mid), bilirubin_fit_mid), creatinine_fit_mid), lactate_fit_mid), platelets_fit_mid), gcs_fit_mid), risk_not)
        rule2 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(heart_rate_fit_mid, map_fit_high), bilirubin_fit_mid), creatinine_fit_mid), lactate_fit_mid), platelets_fit_mid), gcs_fit_mid), risk_not)
        rule3 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(heart_rate_fit_low, map_fit_high), bilirubin_fit_mid), creatinine_fit_mid), lactate_fit_mid), platelets_fit_mid), gcs_fit_mid), risk_not)
        
        rule4 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(heart_rate_fit_high, map_fit_mid), bilirubin_fit_mid), creatinine_fit_mid), lactate_fit_mid), platelets_fit_mid), gcs_fit_mid), risk_low)
        rule5 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(heart_rate_fit_high, map_fit_mid), bilirubin_fit_mid), creatinine_fit_mid), lactate_fit_mid), platelets_fit_mid), gcs_fit_low), risk_low)
        
        rule6 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_mid), creatinine_fit_high), lactate_fit_mid), platelets_fit_low), risk_mid)
        rule7 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_high), creatinine_fit_mid), lactate_fit_mid), platelets_fit_low), risk_mid)
        rule8 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_high), creatinine_fit_high), lactate_fit_mid), platelets_fit_mid), risk_mid)
        rule9 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_high), creatinine_fit_high), lactate_fit_mid), platelets_fit_low), risk_mid)
        
        rule10 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_mid), creatinine_fit_high), lactate_fit_high), platelets_fit_low), risk_high)
        rule11 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_high), creatinine_fit_mid), lactate_fit_high), platelets_fit_low), risk_high)
        rule12 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_high), creatinine_fit_high), lactate_fit_high), platelets_fit_mid), risk_high)
        rule13 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_mid, bilirubin_fit_high), creatinine_fit_high), lactate_fit_high), platelets_fit_low), risk_high)        
        rule14 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_mid), creatinine_fit_high), lactate_fit_mid), platelets_fit_low), risk_high)
        rule15 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_high), creatinine_fit_mid), lactate_fit_mid), platelets_fit_low), risk_high)
        rule16 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_high), creatinine_fit_high), lactate_fit_mid), platelets_fit_mid), risk_high)
        rule17 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_high), creatinine_fit_high), lactate_fit_mid), platelets_fit_low), risk_high)        
        rule18 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_mid), creatinine_fit_high), lactate_fit_high), platelets_fit_low), risk_high)
        rule19 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_high), creatinine_fit_mid), lactate_fit_high), platelets_fit_mid), risk_high)
        rule20 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_high), creatinine_fit_high), lactate_fit_high), platelets_fit_low), risk_high)
        rule21 = np.fmin(np.fmin(np.fmin(np.fmin(np.fmin(map_fit_low, bilirubin_fit_high), creatinine_fit_high), lactate_fit_high), platelets_fit_low), risk_high)        
        
        # Inference
        
        out_not = np.fmax(np.fmax(np.fmax(rule0, rule1), rule2), rule3)
        out_low = np.fmax(rule4, rule5)
        out_mid = np.fmax(np.fmax(np.fmax(rule6, rule7), rule8), rule9)
        out_high = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(rule10, rule11), rule12), rule13), rule14), rule15), rule16), rule17), rule18), rule19), rule20), rule21)
        
        # Defuzzification
        
        out_risk = np.fmax(np.fmax(np.fmax(out_not, out_low), out_mid), out_high)
        defuzzified  = fuzz.defuzz(y_risk, out_risk, 'mom')
        result = fuzz.interp_membership(y_risk, out_risk, defuzzified)
        
        
        # Risk Results
        if(result > 0.4):
            result_array.append(1.0)
        else:
            result_array.append(0.0)
        
    return result_array

Y_result = fuzzy_inference_system(X, Y)

# Results

sklearn.metrics.confusion_matrix(Y, Y_result)

sklearn.metrics.accuracy_score(Y, Y_result)

sklearn.metrics.f1_score(Y, Y_result)

sklearn.metrics.roc_auc_score(Y, Y_result)
