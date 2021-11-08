from __future__ import absolute_import
from __future__ import print_function

import csv
import numpy as np
import os
import pandas as pd
from pandas.core.algorithms import diff
from pandas.tseries.offsets import Nano
from tqdm import tqdm
import datetime

from mimic3benchmark.util import dataframe_from_csv


def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']] # select a subset of all columns. 
    pats.DOB = pd.to_datetime(pats.DOB)
    # this isn't good. this line address the overflow of int64 in line82 if the DOB is too early. there are some patient was born earlier than 1900/1/1
    # TODO: figure out an way to address the above issue. this is just a temporarily fix
    # pats['DOB'].loc[(pats['DOB'] < pd.to_datetime("1900-01-1 00:00:00"))] = pd.to_datetime("1910-1-1 00:00:00")
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats

def read_full_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats.DOB = pd.to_datetime(pats.DOB)
    # this isn't good. this line address the overflow of int64 in line82 if the DOB is too early. there are some patient was born earlier than 1900/1/1
    # TODO: figure out an way to address the above issue. this is just a temporarily fix
    # pats['DOB'].loc[(pats['DOB'] < pd.to_datetime("1900-01-1 00:00:00"))] = pd.to_datetime("1910-1-1 00:00:00")
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats

def read_full_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def read_icd_diagnoses_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses

def read_chartevents(mimic3_path):
    events = dataframe_from_csv(os.path.join(mimic3_path, 'CHARTEVENTS.csv'))
    return events


def get_ventilation_classification(mimic3_path):
    # read 
    chartevents = read_chartevents_vent(mimic3_path)
    # we need icustay_id, charttime, MechVent, OxygenTherapy, Extubated, SelfExtubated from chartevents_vent
    # def isEmpty(x):
    #     if x is not None:
    #         return 1
    #     else:
    #         return 0
    # merged['TRACH'] = merged.ICUSTAY_ID.apply(isEmpty)
    def calMechVent(row):
        itemId_vent_set = [
        445, 448, 449, 450, 1340, 1486, 1600, 224687 # minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 # tidal volume
        , 218,436,535,444,459,224697,224695,224696,224746,224747 # High/Low/Peak/Mean/Neg insp force ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 # Insp pressure
        , 543 # PlateauPressure
        , 5865,5866,224707,224709,224705,224706 # APRV pressure
        , 60,437,505,506,686,220339,224700 # PEEP
        , 3459 # high pressure relief
        , 501,502,503,224702 # PCV
        , 223,667,668,669,670,671,672 # TCPCV
        , 224701 # PSVlevel
        ]
        item_id = row['ITEMID']
        value = row['VALUE']
        # print("... item_id: ", item_id, value, type(item_id), type(value))
        return_val = 0
        if item_id == None or item_id == 'null':
            return_val = return_val | 0
        if item_id == 720 and value != 'Other/Remarks':
            return_val = return_val | 1
        if item_id == 223848 and value != 'Other':
            return_val = return_val | 1
        if item_id == 223849:
            return_val = return_val | 1
        if item_id == 467 and value == 'Ventilator':
            return_val = return_val | 1
        if item_id in itemId_vent_set:
            return_val = return_val | 1
        return return_val
    def calOxygenTherapy(row):
        item_id = row['ITEMID']
        value = row['VALUE']
        return_val = 0
        observations_set_226732 = [
          'Nasal cannula', # 153714 observations
          'Face tent', # 24601 observations
          'Aerosol-cool', # 24560 observations
          'Trach mask ', # 16435 observations
          'High flow neb', # 10785 observations
          'Non-rebreather', # 5182 observations
          'Venti mask ', # 1947 observations
          'Medium conc mask ', # 1888 observations
          'T-piece', # 1135 observations
          'High flow nasal cannula', # 925 observations
          'Ultrasonic neb', # 9 observations
          'Vapomist' # 3 observations
        ]
        
        observations_set_467 = [
          'Cannula', # 278252 observations
          'Nasal Cannula', # 248299 observations
          #-- 'None', -- 95498 observations
          'Face Tent', # 35766 observations
          'Aerosol-Cool', # 33919 observations
          'Trach Mask', # 32655 observations
          'Hi Flow Neb', # 14070 observations
          'Non-Rebreather', # 10856 observations
          'Venti Mask', # 4279 observations
          'Medium Conc Mask', # 2114 observations
          'Vapotherm', # 1655 observations
          'T-Piece', # 779 observations
          'Hood', # 670 observations
          'Hut', # 150 observations
          'TranstrachealCat', # 78 observations
          'Heated Neb', # 37 observations
          'Ultrasonic Neb' # 2 observations
        ]
        
        if item_id == 226732 and value in observations_set_226732:
            return_val = return_val | 1
        if item_id == 467 and value in observations_set_467:
            return_val = return_val | 1
            
        return return_val
    
    def calExtubated(row):
        item_id = row['ITEMID']
        value = row['VALUE']
        return_val = 0
        if item_id is None or item_id == 'null' or value == 'null' or value == None:
            return_val = return_val | 0
        if item_id == 640 and (value == 'Extubated' or value == 'Self Extubation'):
            return_val = return_val | 1
        return return_val
            
    def calSelfExtubated(row):
        item_id = row['ITEMID']
        value = row['VALUE']
        return_val = 0
        if item_id is None or item_id == 'null' or value == 'null' or value == None:
            return_val = return_val | 0
        if item_id == 640 and value == 'Self Extubation':
            return_val = return_val | 1
        return return_val
        
    mechVent = chartevents.apply(lambda row: calMechVent(row), axis = 1)
    oxygenTherapy = chartevents.apply(lambda row: calOxygenTherapy(row), axis = 1)
    extubated = chartevents.apply(lambda row: calExtubated(row), axis = 1)
    selfExtubated = chartevents.apply(lambda row: calSelfExtubated(row), axis = 1)
    
    
    chartevents = chartevents[['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'SUBJECT_ID', 'HADM_ID']]
    chartevents['MECHVENT'] = mechVent
    chartevents['OXYGENTHERAPY'] = oxygenTherapy
    chartevents['EXTUBATED'] = extubated
    chartevents['SELFEXTUBATED'] = selfExtubated
    
    procedureevents_mv = dataframe_from_csv(os.path.join(mimic3_path, 'PROCEDUREEVENTS_MV.csv'))
    procedureevents_mv_id_set = [
          227194 # "Extubation"
        , 225468 # "Unplanned Extubation (patient-initiated)"
        , 225477 # "Unplanned Extubation (non-patient initiated)"
    ]
    procedureevents_mv = procedureevents_mv[(procedureevents_mv.ITEMID.isin(procedureevents_mv_id_set))]
    procedureevents_mv['CHARTTIME'] = pd.to_datetime(procedureevents_mv.STARTTIME)
    procedureevents_mv['MECHVENT'] = 0
    procedureevents_mv['OXYGENTHERAPY'] = 0
    procedureevents_mv['EXTUBATED'] = 1
    def calProcedureevents_mvSelfExtubated(row):
        itemId = row['ITEMID']
        if id == 225468:
            return 1
        return 0
    procedureevents_mv['SELFEXTUBATED'] = procedureevents_mv.apply(lambda row: calProcedureevents_mvSelfExtubated(row), axis = 1)
    procedureevents_mv = procedureevents_mv[['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'MECHVENT', 'OXYGENTHERAPY', 'EXTUBATED', 'SELFEXTUBATED', 'HADM_ID', 'SUBJECT_ID']]
    
    events = pd.concat([chartevents, procedureevents_mv], ignore_index=True,)
    
    return events

def get_ventilation_durations(ventilation_events):
    # print(ventilation_events.to_string())
    ventilation_events = ventilation_events.sort_values(by=['ICUSTAY_ID', 'CHARTTIME', 'MECHVENT'])
    # ventilation_events_with_duration = ventilation_events[(ventilation_events.MECHVENT.isin([1]))]
    ventilation_events_with_duration = ventilation_events

    ventilation_events_with_duration.CHARTTIME = pd.to_datetime(ventilation_events_with_duration.CHARTTIME)
    
    
    ventilation_events_with_duration['VENTDURATION_LAG'] = ventilation_events_with_duration.groupby(['ICUSTAY_ID', 'MECHVENT']).CHARTTIME.diff() 
    ventilation_events_with_duration['VENTDURATION'] = ventilation_events_with_duration['VENTDURATION_LAG']
    # def filterDuration(x):
    #     if x is 'Nat' or x < pd.to_timedelta('0 days 00:00:00'):
    #         return -1
    #     else:
    #         x = x / np.timedelta64(1, 's')
    #         x = x / 60.  # ventilation duralation in minutes
    #         return x

    # print(ventilation_events_with_duration.to_string())
    def filterDuration(row):
        if row['MECHVENT'] == 0:
            row['VENTDURATION'] = -1
        else:
            x = row['VENTDURATION'] 
            if x is 'NaN' or x is 'NaT' or x is None or x < pd.to_timedelta('0 days 00:00:00'):
                row['VENTDURATION'] = -1
            else:
                x = x / np.timedelta64(1, 's')
                x = x / 60.  # ventilation duralation in minutes
                row['VENTDURATION'] = x
        return row

    ventilation_events_with_duration = ventilation_events_with_duration.apply(lambda row: filterDuration(row), axis = 1)
    
    ventilation_events_with_duration = ventilation_events_with_duration.sort_values(by=['ICUSTAY_ID', 'CHARTTIME'])  
    # newVentFlag = ventilation_events_with_duration.loc[(ventilation_events_with_duration.EXTUBATED == 1) | (ventilation_events_with_duration.SELFEXTUBATED == 1)]
    
    # newVentFlag2 = ventilation_events_with_duration.loc[(ventilation_events_with_duration.OXYGENTHERAPY == 1) & (ventilation_events_with_duration.MECHVENT == 0)]
    
    ventilation_events_with_duration = ventilation_events_with_duration.loc[ventilation_events_with_duration.VENTDURATION > 0]
    # print(".... test: \n", test.to_string(), test.shape)
    # print(".... test2: \n", newVentFlag2.to_string())
    ventilation_events_with_duration['ENDTIME'] = pd.to_datetime(ventilation_events_with_duration.CHARTTIME)
    ventilation_events_with_duration['STARTTIME'] = ventilation_events_with_duration.ENDTIME - pd.to_timedelta(ventilation_events_with_duration.VENTDURATION_LAG)
    
    ventilation_events_with_duration = ventilation_events_with_duration[['ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'VENTDURATION', 'VENTDURATION_LAG', 'SUBJECT_ID', 'HADM_ID']]
    
    
    # print(">>>> extubatedLag: ", extubatedLag.to_string())
    # print(ventilation_events_with_duration.to_string())
    # print(ventilation_events_with_duration.shape)
    # test = ventilation_events_with_duration.loc[ventilation_events_with_duration.groupby('ICUSTAY_ID').CHARTTIME.idxmax()]
    # test = ventilation_events_with_duration.groupby('ICUSTAY_ID').EXTUBATED.sum()
    
    # print(">>>>> test\n", test.to_string())
    ventilation_events_with_duration['ITEMID'] = 999999 # special itemId for ventilation duration
    ventilation_events_with_duration['CHARTTIME'] = ventilation_events_with_duration['STARTTIME']
    ventilation_events_with_duration['VALUE'] = ventilation_events_with_duration['VENTDURATION']
    ventilation_events_with_duration['VALUEUOM'] = "mins"
    ventilation_events_with_duration = ventilation_events_with_duration[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']]
    ventilation_events_with_duration['ICUSTAY_ID'] = ventilation_events_with_duration.ICUSTAY_ID.astype('int')
    # print("<><><> start_time: \n", ventilation_events_with_duration.to_string(), ventilation_events_with_duration.shape)
    
    return ventilation_events_with_duration


def read_procedureevents_mv_vent(mimic3_path):
    ventilation_item_id_set = [
         227194 # "Extubation"
        , 225468 # "Unplanned Extubation (patient-initiated)"
        , 225477 # "Unplanned Extubation (non-patient initiated)"
    ]
    return None

def read_chartevents_vent(mimic3_path):
    events = dataframe_from_csv(os.path.join(mimic3_path, 'CHARTEVENTS.csv'))
    ventilation_item_id_set = [
    # the below are settings used to indicate ventilation
      720, 223849 # vent mode
    , 223848 # vent type
    , 445, 448, 449, 450, 1340, 1486, 1600, 224687 # minute volume
    , 639, 654, 681, 682, 683, 684,224685,224684,224686 # tidal volume
    , 218,436,535,444,224697,224695,224696,224746,224747 # High/Low/Peak/Mean ("RespPressure")
    , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 # Insp pressure
    , 543 # PlateauPressure
    , 5865,5866,224707,224709,224705,224706 # APRV pressure
    , 60,437,505,506,686,220339,224700 # PEEP
    , 3459 # high pressure relief
    , 501,502,503,224702 # PCV
    , 223,667,668,669,670,671,672 # TCPCV
    , 224701 # PSVlevel

    # the below are settings used to indicate extubation
    , 640 # extubated

    # the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
    , 468 # O2 Delivery Device#2
    , 469 # O2 Delivery Mode
    , 470 # O2 Flow (lpm)
    , 471 # O2 Flow (lpm) #2
    , 227287 # O2 Flow (additional cannula)
    , 226732 # O2 Delivery Device(s)
    , 223834 # O2 Flow

    # used in both oxygen + vent calculation
    , 467 # O2 Delivery Device
    ]
   
    events = events[(events.ITEMID.isin(ventilation_item_id_set) & events.VALUE.notnull() & (~events.ERROR.isin([1.0])))] 
    events = events[['ICUSTAY_ID', "CHARTTIME","ITEMID", 'VALUE', 'VALUENUM', 'VALUEUOM', 'STORETIME', 'SUBJECT_ID', 'HADM_ID']]
    events.STORETIME = pd.to_datetime(events.STORETIME)
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html
    # Whether elements in Series are contained in values.
    # print("after: events: ", events.shape)
    # print(events.shape)
    # print(events.to_string())

    return events
    
def get_ventilation_events_with_label(events, d_items):
    merged = events.merge(d_items, how='inner', left_on=['ITEMID'], right_on=['ITEMID'])
    
    merged = merged[['ICUSTAY_ID', "ITEMID", 'VALUE', 'VALUENUM', 'VALUEUOM', 'STORETIME', 'LABEL']]
    merged = merged[(~merged.VALUE.isin(['None', None]))]
    return merged    

def read_d_items(mimic3_path):
    d_items = dataframe_from_csv(os.path.join(mimic3_path, 'D_ITEMS.csv'))
    return d_items

def read_events_table_by_row(mimic3_path, table):
    nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219, 'ventilation': 330712484}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]


def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes[codes.COUNT > 0]
    if output_path:
        codes.to_csv(output_path, index_label='ICD9_CODE')
    return codes.sort_values('COUNT', ascending=False).reset_index()


def remove_icustays_with_transfers(stays):
    stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]


def get_mpwr_trach(icustays, chartevents):
    chartevents = chartevents[(chartevents.ITEMID.isin([687,688,690,691,692,224831,224829,224830,224864,225590,227130]) 
                               & (chartevents.ERROR.notnull())
                               & (chartevents.VALUE.notnull()))]
    
    merged = icustays.merge(chartevents, how='left', left_on=['ICUSTAY_ID'], right_on=['ICUSTAY_ID'])
    merged.INTIME = pd.to_datetime(merged.INTIME)
    merged.CHARTTIME = pd.to_datetime(merged.CHARTTIME)
    
    merged = merged[(merged.CHARTTIME > merged.INTIME) & (merged.CHARTTIME.subtract(merged.INTIME) < pd.to_timedelta('3 days'))]
    merged = merged[['ICUSTAY_ID', 'CHARTTIME']]
    
    def isEmpty(x):
        if x is not None:
            return 1
        else:
            return 0
    merged['TRACH'] = merged.ICUSTAY_ID.apply(isEmpty)
    return merged

def get_mpwr_cohort(icustays, admissions, mpwr_trach):
    return None
    
def merge_on_subject(table1, table2):
    # print(table1.shape)
    # print(table2.shape)
    merged = table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])
    # print(merged.shape)
    return merged


def merge_on_subject_admission(table1, table2):
    merged = table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
    return merged


def add_age_to_icustays(stays):    
    
    stays['AGE'] = (stays.INTIME.subtract(stays.DOB)).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    # stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays


def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)

def break_up_ventilation_by_subject(vent, output_path, subjects=None):
    subjects = vent.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        vent[vent.SUBJECT_ID == subject_id].sort_values(by='CHARTTIME').to_csv(os.path.join(dn, 'vent.csv'),
                                                                              index=False)



def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUENUM', 'VALUEUOM']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219, 'ventilation': 330712484}
    print('teble: ', table)
    nb_rows = nb_rows_dict[table.lower()]
    
    
    for row, row_no, _ in tqdm(read_events_table_by_row(mimic3_path, table), total=nb_rows,
                                                        desc='Processing {} table'.format(table)):
        if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['ITEMID'] not in items_to_keep):
            continue
        
        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUENUM': '' if 'VALUENUM' not in row else row['VALUENUM'],
                   'VALUEUOM': row['VALUEUOM']}
        
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()
        
        
