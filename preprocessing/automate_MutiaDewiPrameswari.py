import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
import os


def preprocess_data(
    input_path: str,
    output_path: str,
    pipeline_path: str
):
    """
    Fungsi untuk melakukan preprocessing otomatis pada dataset
    Breast Cancer Wisconsin dan mengembalikan dataset siap dilatih.
    """

    #Load dataset
    df = pd.read_csv(input_path)

    #Drop kolom id karena tidak relevan
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    #Encode target
    encoder = LabelEncoder()
    df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

    #Pisahkan fitur & target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    #Pipeline preprocessing
    preprocessing_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    X_scaled = preprocessing_pipeline.fit_transform(X)

    #Gabungkan kembali
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed['diagnosis'] = y.values

    #Simpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    dump(preprocessing_pipeline, pipeline_path)

    print("Preprocessing selesai.")
    print(f"Dataset tersimpan di: {output_path}")
    print(f"Pipeline tersimpan di: {pipeline_path}")

    return df_processed


if __name__ == "__main__":
    preprocess_data(
        input_path="../breast-cancer_raw/breast-cancer_raw.csv",
        output_path="../preprocessing/breast-cancer_preprocessing.csv",
        pipeline_path="../preprocessing/preprocessing_pipeline.joblib"
    )
