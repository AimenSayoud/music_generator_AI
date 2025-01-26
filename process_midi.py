import pretty_midi
import numpy as np
import glob
import pickle
from pathlib import Path
import json
import sys
import os

def extract_features(midi_file):
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        piano_roll = pm.get_piano_roll(fs=16)
        piano_roll = piano_roll.T
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return None

def process_dataset(midi_dir):
    processed_data = []
    midi_files = glob.glob(f"{midi_dir}/*.mid") + glob.glob(f"{midi_dir}/*.midi")
    
    total_files = len(midi_files)
    processed_files = 0
    
    for midi_file in midi_files:
        features = extract_features(midi_file)
        if features is not None and len(features) > 0:
            for i in range(0, len(features) - 64, 32):
                sequence = features[i:i + 64]
                processed_data.append(sequence)
        
        processed_files += 1
        # Report progress
        progress = (processed_files / total_files) * 100
        print(json.dumps({"progress": progress}), flush=True)

    return np.array(processed_data)

def main():
    params = json.loads(sys.argv[1])
    midi_dir = params['upload_dir']
    output_dir = params['output_dir']
    
    try:
        processed_data = process_dataset(midi_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'processed_music_data.pkl')
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(json.dumps({
            "status": "success",
            "sequences": len(processed_data),
            "shape": processed_data.shape
        }))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()


# python3 src/python/process_midi.py '{"upload_dir": "/Users/aymen/Documents/music_magic/music_magic/input", "output_dir": "/Users/aymen/Documents/music_magic/music_magic/output