import pandas as pd

def preprocess(employee_data):
    # Import dataset

    # Filling NA values in 'rating_tahun_lalu' column with median
    employee_data['rating_tahun_lalu'] = employee_data['rating_tahun_lalu'].fillna(employee_data['rating_tahun_lalu'].median(skipna=True))

    # Filling NA values in 'pendidikan' column with most frequent data
    employee_data['pendidikan'] = employee_data['pendidikan'].fillna("Bachelor's")

    employee_data.drop(['departemen','wilayah','pendidikan','jenis_kelamin','rekrutmen'], axis=1, inplace=True)

    # Dropping column which had less influence to promotion
    employee_data.drop(['id_karyawan','jumlah_training','umur','masa_kerja'], axis=1, inplace=True)

    return employee_data.iloc[:,1:5]

