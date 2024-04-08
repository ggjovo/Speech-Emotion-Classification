import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy.stats import skew, kurtosis, linregress
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from specs import *



DATA_PATH = os.path.join("data/raw")

speakers = os.listdir(DATA_PATH)

NUM_MFCCS = 12

columns = ["speaker_id", "target"]

base_columns = ["zcr", "rms_energy", "F0"]

base_columns.extend([f"mfcc_{i}" for i in range(1, NUM_MFCCS+1)])

functionals = ["min", "max", "mean", "median", "q1", "q3", "std", "skewness", "kurtosis", "iqr", "range", "slope", "intercept", "mse"]

for col in base_columns:
    for func in functionals:
        columns.append(f"{col}_{func}")
        columns.append(f"delta_{col}_{func}")
        
        
def calculate_functionals(data, delta_data):
    
    time = np.array([i for i in range(len(data))], dtype=np.float64)
    slope, intercept, _, _, std_err = linregress(time, data)
    data_pred = intercept + slope * time.astype(int)
    mse = np.mean((data - data_pred)**2)
    
    delta_time = np.array([i for i in range(len(delta_data))], dtype=np.float64)
    delta_slope, delta_intercept, _, _, _ = linregress(delta_time, delta_data)
    delta_data_pred = delta_intercept + delta_slope * delta_time.astype(int)
    delta_mse = np.mean((delta_data - delta_data_pred)**2)
    
    return [
        data.min(),
        delta_data.min(),
        data.max(),
        delta_data.max(),
        data.mean(),
        delta_data.mean(),
        np.median(data),
        np.median(delta_data),
        np.percentile(data, 25),
        np.percentile(delta_data, 25),
        np.percentile(data, 75),
        np.percentile(delta_data, 75),
        np.std(data),
        np.std(delta_data),
        skew(data),
        skew(delta_data),
        kurtosis(data),
        kurtosis(delta_data),
        np.percentile(data, 75) - np.percentile(data, 25),
        np.percentile(delta_data, 75) - np.percentile(delta_data, 25),
        np.max(data) - np.min(data),
        np.max(delta_data) - np.min(delta_data),
        slope,
        delta_slope,
        intercept,
        delta_intercept,
        mse,
        delta_mse
    ]
    

def process_audio_file(file_path, speaker_id, emotion):
    audio, sr = librosa.load(file_path)
    
    if len(audio) == 0:
        return np.array([None for _ in range(len(columns))])
    
    feature_vector = [speaker_id, emotion]
    
    
    #CALCLULATING ZCR AND DELTA_ZCR FUNCTIONALS
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    delta_zcr = librosa.feature.delta(zcr)
    
    feature_vector.extend(calculate_functionals(zcr, delta_zcr))
    
    
    #CALCULATING RMS ENERGY FUNCTIONALS
    rms_energy = librosa.feature.rms(y=audio)[0]
    delta_rms_energy = librosa.feature.delta(rms_energy)
    
    feature_vector.extend(calculate_functionals(rms_energy, delta_rms_energy))
    
    
    #CALCULATING F0 FUNCTIONALS
    f0, vid, vpd = librosa.pyin(audio, sr = sr, fmin = librosa.note_to_hz('C2'), fmax= librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    delta_f0 = librosa.feature.delta(f0)
    
    feature_vector.extend(calculate_functionals(f0, delta_f0))
    
    
    #CALCULATING MFCC FUNCTIONALS
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCCS)
    delta_mfccs = librosa.feature.delta(mfccs)
    
    for mfcc, delta_mfcc in zip(mfccs, delta_mfccs):
        feature_vector.extend(calculate_functionals(mfcc, delta_mfcc))
        
    if sum(np.array(feature_vector) == None) > 0:
        print(file_path)
        print(zcr)
        print(delta_zcr)
        print(rms_energy)
        print(delta_rms_energy)
        print(f0)
        print(delta_f0)
            
    return np.array(feature_vector)


def main():
    df_rows = []
    for speaker_id in tqdm(speakers, desc="Processing: "):
        actor_path = os.path.join(DATA_PATH, speaker_id)
        emotions = os.listdir(actor_path)

        # Skip actors without all 5 emotions recorded
        if len(emotions) != 5:
            continue

        with parallel_backend('threading', n_jobs=-1):
            rows = Parallel()(delayed(process_audio_file)(os.path.join(actor_path, emotion, audio_file), speaker_id, emotion)
                              for emotion in emotions
                              for audio_file in os.listdir(os.path.join(actor_path, emotion)))

            df_rows.extend(row for row in rows if row is not None)

    df = pd.DataFrame(df_rows, columns=columns)
    df.dropna(inplace=True)
    df.to_csv("data/processed/data_12mfccs.csv", index=False)

if __name__ == "__main__":
    main()

