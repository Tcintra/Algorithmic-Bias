import pandas as pd
import numpy as np
import os

def clean():

    # Use CleanCSV to clean the compas-scores.csv file from ProPublica's github.
    path = os.path.join("..", "Data")
    df = pd.read_csv(os.path.join(path, 'compas-scores.csv'), index_col='name')

    # Drop uninformative columns.
    df.drop(["id", "first", "last", "compas_screening_date", "dob",  "days_b_screening_arrest", 'c_jail_in', 'c_jail_out', "c_case_number", "c_arrest_date", "c_offense_date",  "r_case_number", "r_charge_degree", "r_days_from_arrest", "r_offense_date", "r_charge_desc", "vr_case_number", "vr_charge_degree", "vr_charge_desc", "num_vr_cases", 'c_charge_desc', 'num_r_cases', 'r_jail_in', 'r_jail_out', 'num_vr_cases', 'vr_offense_date', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date', 'type_of_assessment', 'decile_score.1', 'screening_date'], axis = 1, inplace = True)

    # Combine recidivism and violent recidivism into one column for simplicity's sake.
    def recid(row):
        if row['is_recid'] == 1:
            val = '1'
        elif row['is_violent_recid'] == 1:
            val = '1'
        else: 
            val = '0'
        return val
    df['is_recid'] = df.apply(recid, axis = 1)
    df.drop(['is_violent_recid'], axis = 1, inplace = True)

    # Drop entries where recidivism scores are nonsensical, assume problems with data collection.
    indexNames = df[ df['decile_score'] < 1 ].index
    df.drop(indexNames,inplace=True)

    # Drop entries where the compas assessment occured 30 days after the initial arrest. Again, assume problems with data collection.
    df= df.loc[df['c_days_from_compas'] <= 30]
    df.drop(['c_days_from_compas'], axis = 1, inplace = True)

    # Rorder columns
    df = df[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'decile_score', 'score_text', 'is_recid']]

    # Rename Columns
    df.columns = ['Gender', 'Age', 'Age Category', 'Race', 'Juvenile Felony Count', 'Juvenile Misdemeanor Count', 'Juvenile (other) Count','Priors Count', "Charge Degree", 'Threat Score', 'Assessment', 'Recidivism']

    # export it to Compas.csv
    path = "../Data/"
    df.to_csv(path + 'Compas.csv')

if __name__ == '__main__':
    clean()