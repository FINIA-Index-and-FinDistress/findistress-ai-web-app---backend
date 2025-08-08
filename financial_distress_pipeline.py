import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import shap
from imblearn.over_sampling import ADASYN
import statsmodels.api as sm
from joblib import dump
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import os

# Define sector mappings
mappings = {
    'Construction': '1', 'Retail': '2', 'Manufacturing': '3', 'Other Services': '4',
    'Other Manufacturing': '5', 'Food': '6', 'Garments': '7', 'Hotels': '8',
    'Services': '9', 'Rest of Universe': '10', 'IT & IT Services': '11', 'Textiles': '12',
    'Machinery & Equipment': '13', 'Textiles & Garments': '14',
    'Basic Metals/Fabricated Metals/Machinery & Equip.': '15',
    'Chemicals, Plastics & Rubber': '16', 'Chemicals & Chemical Products': '17',
    'Machinery & Equipment & Electronics': '18', 'Leather Products': '19',
    'Furniture': '20', 'Motor Vehicles & Transport Equip.': '21',
    'Fabricated Metal Products': '22', 'Hospitality & Tourism': '23',
    'Motor Vehicles': '24', 'Electronics': '25', 'Services of Motor Vehicles/Wholesale/Retail': '26',
    'Food/Leather/Wood/Tobacco/Rubber Products': '27', 'Professional Activities': '28',
    'Non-Metallic Mineral Products': '29', 'Hotels & Restaurants': '30',
    'Electronics & Communications Equip.': '31', 'Transport, Storage, & Communications': '32',
    'Services of Motor Vehicles': '33', 'Rubber & Plastics Products': '34',
    'Basic Metals & Metal Products': '35', 'Wholesale': '36', 'Basic Metals': '37',
    'Electrical & Computer Products': '38', 'Minerals, Metals, Machinery & Equipment': '39',
    'Wood Products': '40', 'Printing & Publishing': '41', 'Petroleum products, Plastics & Rubber': '42',
    'Wood products, Furniture, Paper & Publishing': '43',
    'Machinery & Equipment, Electronics & Vehicles': '44', 'Transport': '45',
    'Textiles, Garments & Leather': '46', 'Restaurants': '47',
    'Wholesale, Including of Motor Vehicles': '48', 'Publishing, Telecommunications & IT': '49',
    'Wholesale & Retail': '50', 'Mining Related Manufacturing': '51',
    'Pharmaceuticals & Medical Products': '52', 'Wood Products & Furniture': '53',
    'Computer, Electronic & Optical Products': '54', 'Retail & IT': '55',
    'Metals, Machinery, Computers & Electronics': '56', 'Manufacturing Panel': '57',
    'Retail Panel': '58', 'Other Services Panel': '59',
    'Chemicals, Non-Metallic Mineral, Plastics & Rubber': '60',
    'Textiles, Garments, Leather & Paper': '61', 'Pharmaceutical, Chemicals & Chemical Products': '62',
    'Wholesale of Agri Inputs & Equipment': '63'
}

# Load and preprocess data
def load_data(file_path='DF_2025.xlsx'):
    df = pd.read_excel(file_path)
    
    # Select features
    df2 = df[['idstd', 'country2', 'region', 'year', 'car1', 'fin1', 'fin2', 'fin3', 'fin4', 'fin5',
              'fin16', 'fin33', 'gend2', 'gend3', 'gend4', 'gend6', 'wk14', 'car3', 'car2', 'car6',
              'obst9', 'tr15', 't10', 't2', 'corr4', 'obst11', 'infor1', 'perf1', 'obst1', 'stra_sector',
              'GDP', 'Credit', 'MarketCap', 'WUI', 'GPR', 'PRIME', 'WSI', 'size2']].copy()

    # Rename columns
    df2 = df2.rename(columns={
        'fin1': 'Fin_int', 'fin2': 'Fin_bank', 'fin3': 'Fin_supplier', 'fin4': 'Fin_equity',
        'fin5': 'Fin_other', 'gend2': 'Fem_wf', 'gend3': 'Fem_Wf_Non_Prod', 'gend4': 'Fem_CEO',
        'gend6': 'Fem_Own', 'car3': 'For_Own', 'car2': 'Pvt_Own', 'car6': 'Con_Own',
        'obst9': 'Edu', 'tr15': 'Exports', 't10': 'Innov', 't2': 'Transp', 'corr4': 'Gifting',
        'obst11': 'Pol_Inst', 'infor1': 'Infor_Comp', 'size2': 'Size'
    })

    # Map sector
    df2['Sector'] = df2['stra_sector'].map(mappings)
    # df2['Sector'] = df2['Sector'].astype(float)  

    # Create regions
    # african_countries = [
    #     'Angola', 'Bangladesh', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
    #     'Central African Republic', 'Chad', 'Congo', "Cote d'Ivoire", 'DRC', 'Djibouti', 'Egypt',
    #     'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
    #     'Lebanon', 'Lesotho', 'Liberia', 'Guineabissau', 'Kenya', 'Madagascar', 'Malawi',
    #     'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
    #     'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Sudan',
    #     'Southafrica', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
    # ]
    # df2['region2'] = df2['country2'].apply(lambda x: 'AFR' if x in african_countries else 'ROW')
    
    # # Create df3 for AFR and ROW
    # df3_afr = df2[df2['region2'] == 'AFR'][['idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33',
    #                                         'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
    #                                         'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Exports', 'Edu', 'Innov', 'Transp',
    #                                         'Gifting', 'Pol_Inst', 'Infor_Comp', 'Sector', 'Credit', 'WSI', 'WUI',
    #                                         'GDP', 'PRIME']].copy()
    # df3_row = df2[df2['region2'] == 'ROW'][['idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33',
    #                                         'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
    #                                         'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp',
    #                                         'Gifting', 'Pol_Inst', 'Infor_Comp', 'Credit', 'Sector', 'WUI', 'WSI',
    #                                         'PRIME', 'MarketCap', 'GPR']].copy()

    df2.loc[(df2['country2'].isin(['Angola','Bangladesh','Benin', 'Botswana','Burkina Faso', 'Burundi','Cameroon','Central African Republic', 'Chad',
                              'Congo',"Cote d'Ivoire", 'DRC','Djibouti','Egypt','Equatorial Guinea','Eswatini', 'Ethiopia',
                               'Gabon', 'Gambia','Ghana','Guinea','Lebanon','Lesotho', 'Liberia', 'Guineabissau','Kenya', 'Madagascar', 'Malawi',
                               'Mali','Mauritania', 'Mauritius', 'Morocco','Mozambique','Namibia', 'Niger', 'Nigeria','Rwanda','Senegal', 'Seychelles', 
                               'Sierra Leone','South Sudan', 'Southafrica', 'Sudan','Tanzania','Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'])), 
    'region2'] = 'AFR'

    df2.loc[(df2['country2'].isin(['Afghanistan', 'Albania', 'Antiguaandbarbuda','Argentina', 'Armenia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
                               'Barbados', 'Belarus', 'Belgium','Belize','Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cabo Verde', 
                               'Cambodia', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica','Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Dominica',
                               'Dominican Republic', 'Ecuador', 'El Salvador','Eritrea', 'Estonia',  'Fiji', 'Finland', 'France','Georgia', 'Germany','Greece', 'Grenada', 
                               'Guatemala','Guyana', 'Honduras', 'Hong Kong SAR China', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iraq', 'Ireland',
                               'Israel', 'Italy', 'Jamaica', 'Jordan', 'Kazakhstan', 'Korea Republic', 'Kosovo', 'Kyrgyz Republic', 'Lao PDR', 'Latvia',
                               'Lithuania', 'Luxembourg',  'Malaysia','Malta','Mexico', 'Micronesia, Fed. Sts.', 'Moldova', 'Mongolia', 'Montenegro', 
                               'Myanmar','Nepal', 'Netherlands', 'New Zealand', 'Nicaragua','North Macedonia', 'Pakistan', 'Panama', 'Papua New Guinea', 
                               'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia','Samoa', 'Saudi Arabia','Serbia', 'Singapore',
                               'Slovak Republic', 'Slovenia', 'Solomon Islands','Spain', 'SriLanka', 'Stkittsandnevis', 'Stlucia',
                               'Stvincentandthegrenadines','Suriname', 'Sweden', 'Taiwan China', 'Tajikistan','Thailand', 'Timor-Leste', 'Togo', 'Tonga',
                               'Trinidad and Tobago','Turkiye', 'Turkmenistan','Ukraine', 'United Kingdom','United States', 'Uruguay', 'Uzbekistan', 
                               'Vanuatu', 'Venezuela', 'Viet Nam', 'West Bank And Gaza', 'Yemen' ])), 'region2'] = 'ROW'
    
    df2_afr = df2[df2['region2'] == "AFR"]
    df2_row = df2[df2['region2'] != "AFR"]

    df3_afr = df2_afr[['idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33', 'Fin_bank', \
                       'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own',\
                          'Exports', 'Edu', 'Innov', 'Transp', 'Gifting', 'Pol_Inst', 'Infor_Comp', 'Sector', \
                            'Credit', 'WSI', 'WUI', 'GDP', 'PRIME']].copy()
    
    df3_row = df2_row[['idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33', 'Fin_bank', \
                       'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own',\
                          'Edu', 'Exports', 'Innov', 'Transp', 'Gifting', 'Pol_Inst', 'Infor_Comp', 'Size', \
                            'Credit', 'Sector', 'WUI', 'WSI', 'PRIME', 'MarketCap', 'GPR', 'GDP']].copy()

    # Create distress and startup variables
    df3_afr['distress'] = np.where((df3_afr['perf1'] < 0) & ((df3_afr['obst1'] == 100) |
                                   (df3_afr['fin33'] == 1) | (df3_afr['fin16'] == 1)), 1, 0)
    df3_afr['startup'] = np.where((df3_afr['wk14'] < 5) & (df3_afr['car1'] < 5), 1, 0)
    df3_row['distress'] = np.where((df3_row['perf1'] < 0) & ((df3_row['obst1'] == 100) |
                                   (df3_row['fin33'] == 1) | (df3_row['fin16'] == 1)), 1, 0)
    df3_row['startup'] = np.where((df3_row['wk14'] < 5) & (df3_row['car1'] < 5), 1, 0)

    # Fill NaN with 0
    df3_afr.fillna(0, inplace=True)
    df3_row.fillna(0, inplace=True)

    # Scale percentage features
    percentage_cols = ['Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO',
                      'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp', 'Gifting',
                      'Pol_Inst', 'Infor_Comp', 'Credit', 'PRIME', 'GDP']
    for col in percentage_cols:
        df3_afr[col] = df3_afr[col].apply(lambda x: x / 100 if x > 1 else x)
        df3_afr[col] = df3_afr[col].apply(lambda x: 0 if x < 0 else x)
        if col in df3_row.columns:
            df3_row[col] = df3_row[col].apply(lambda x: x / 100 if x > 1 else x)
            df3_row[col] = df3_row[col].apply(lambda x: 0 if x < 0 else x)
    
    df3_row['MarketCap'] = df3_row['MarketCap'].apply(lambda x: x / 100 if x > 1 else x)
    df3_row['MarketCap'] = df3_row['MarketCap'].apply(lambda x: 0 if x < 0 else x)

    # Define feature sets
    afr_features = ['startup', 'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
                    'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp',
                    'Gifting', 'Pol_Inst', 'Infor_Comp', 'Credit', 'WSI', 'WUI', 'GDP', 'PRIME']
    row_features = afr_features + ['Size', 'MarketCap', 'GPR']

    # Prepare X and y
    x_afr = df3_afr[afr_features]
    y_afr = df3_afr['distress']
    x_row = df3_row[row_features]
    y_row = df3_row['distress']

    # Add constant for SMOTE compatibility
    x_afr_c = sm.add_constant(x_afr)
    x_row_c = sm.add_constant(x_row)

    # Apply ADASYN
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    x_ada_afr, y_ada_afr = adasyn.fit_resample(x_afr, y_afr)
    x_ada_row, y_ada_row = adasyn.fit_resample(x_row, y_row)

    # Split data
    X_train_afr, X_test_afr, y_train_afr, y_test_afr = train_test_split(x_ada_afr, y_ada_afr, test_size=0.25, random_state=16)
    X_train_row, X_test_row, y_train_row, y_test_row = train_test_split(x_ada_row, y_ada_row, test_size=0.25, random_state=16)

    return df3_afr, df3_row, X_train_afr, X_test_afr, y_train_afr, y_test_afr, X_train_row, X_test_row, y_train_row, y_test_row

# Define pipeline
def create_pipeline():
    # Define features for preprocessing
    afr_features = ['startup', 'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
                    'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp',
                    'Gifting', 'Pol_Inst', 'Infor_Comp', 'Credit', 'WSI', 'WUI', 'GDP', 'PRIME']
    row_features = afr_features + ['Size', 'MarketCap', 'GPR']

    # Preprocessor
    preprocessor_afr = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', IterativeImputer(max_iter=30, random_state=42)),
                ('scaler', StandardScaler())
            ]), afr_features)
        ]
    )
    preprocessor_row = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', IterativeImputer(max_iter=30, random_state=42)),
                ('scaler', StandardScaler())
            ]), row_features)
        ]
    )

    # Models
    models = {
        'LGB_afr': lgb.LGBMClassifier(learning_rate=0.2, n_estimators=150, num_leaves=31, random_state=42),
        'LGB_row': lgb.LGBMClassifier(learning_rate=0.2, n_estimators=150, num_leaves=31, random_state=42)
    }

    return preprocessor_afr, preprocessor_row, models

# Evaluate models and select the best
def evaluate_models(X_train, X_test, y_train, y_test, preprocessor, model, region):
    results = {}
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Fit model
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation accuracy
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    # SHAP analysis
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test)
    
    # Store results
    results = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'confusion_matrix': cm,
        'classification_report': clf_report,
        'shap_values': shap_values,
        'X_test': X_test
    }

    print(f"\n{region} - {model.__class__.__name__}:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot SHAP summary
    shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns, show=False)

    return results, pipeline

# Main function
def main():
    # Load and preprocess data
    df_afr, df_row, X_train_afr, X_test_afr, y_train_afr, y_test_afr, X_train_row, X_test_row, y_train_row, y_test_row = load_data()

    # Create pipeline
    preprocessor_afr, preprocessor_row, models = create_pipeline()

    # Evaluate for AFR
    print("\nEvaluating model for AFR region:")
    afr_results, afr_best_pipeline = evaluate_models(X_train_afr, X_test_afr, y_train_afr, y_test_afr,
                                                    preprocessor_afr, models['LGB_afr'], 'AFR')

    # Evaluate for ROW
    print("\nEvaluating model for ROW region:")
    row_results, row_best_pipeline = evaluate_models(X_train_row, X_test_row, y_train_row, y_test_row,
                                                    preprocessor_row, models['LGB_row'], 'ROW')

    # # Save pipelines
    # dump(afr_best_pipeline, 'afr_best_pipeline.joblib')
    # dump(row_best_pipeline, 'row_best_pipeline.joblib')
    

    # Save pipelines
    os.makedirs('ml_pipeline/trained_models', exist_ok=True)
    dump(afr_best_pipeline, 'ml_pipeline/trained_models/afr_best_pipeline.joblib')
    dump(row_best_pipeline, 'ml_pipeline/trained_models/row_best_pipeline.joblib')
    return afr_best_pipeline, row_best_pipeline

if __name__ == '__main__':
    afr_pipeline, row_pipeline = main()