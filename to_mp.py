import os
import sys
import json
from pathlib import Path
import subprocess
import glob

def convert_midi_to_mp3(midi_file, output_mp3, soundfont_path=None):
    """
    Convert a MIDI file to MP3 using fluidsynth and ffmpeg
    """
    try:
        # Default soundfont path for different operating systems
        if soundfont_path is None:
            if sys.platform == 'darwin':  # macOS
                soundfont_paths = [
                    '/usr/local/share/soundfonts/default.sf2',
                    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                    '/usr/share/sounds/sf2/default.sf2'
                ]
            elif sys.platform == 'linux':
                soundfont_paths = [
                    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                    '/usr/share/soundfonts/default.sf2'
                ]
            else:  # Windows or others
                soundfont_paths = ['./soundfonts/default.sf2']
            
            # Find the first available soundfont
            soundfont_path = next((path for path in soundfont_paths if os.path.exists(path)), None)
            
            if soundfont_path is None:
                raise Exception("No soundfont found. Please install a soundfont or specify its path.")

        # Create a temporary WAV file
        wav_file = midi_file.with_suffix('.wav')
        
        # Convert MIDI to WAV using fluidsynth
        fluidsynth_cmd = [
            'fluidsynth',
            '-ni',
            soundfont_path,
            midi_file,
            '-F', str(wav_file),
            '-r', '44100'
        ]
        
        subprocess.run(fluidsynth_cmd, check=True, capture_output=True)
        
        # Convert WAV to MP3 using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', str(wav_file),
            '-codec:a', 'libmp3lame',
            '-qscale:a', '2',
            str(output_mp3),
            '-y'  # Overwrite output file if it exists
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Clean up temporary WAV file
        if wav_file.exists():
            wav_file.unlink()
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(json.dumps({
            "status": "error",
            "error": f"Conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
        }))
        return False
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))
        return False

def main():
    try:
        # Parse command line arguments if provided, otherwise use defaults
        if len(sys.argv) > 1:
            params = json.loads(sys.argv[1])
        else:
            params = {}
        
        # Get base directory
        base_dir = Path(params.get('base_dir', os.getcwd()))
        
        # Set up paths
        midi_dir = Path(params.get('midi_dir', base_dir / 'public/generated'))
        mp3_dir = Path(params.get('mp3_dir', base_dir / 'public/mp3'))
        soundfont_path = params.get('soundfont_path', None)
        
        # Create mp3 directory if it doesn't exist
        mp3_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all MIDI files in the directory
        midi_files = list(midi_dir.glob('*.mid'))
        
        if not midi_files:
            print(json.dumps({
                "status": "error",
                "error": f"No MIDI files found in {midi_dir}"
            }))
            return
        
        # Convert each MIDI file
        for midi_file in midi_files:
            print(json.dumps({
                "status": "progress",
                "progress": 30,
                "message": f"Converting {midi_file.name}..."
            }))
            sys.stdout.flush()
            
            # Create output MP3 path
            output_mp3 = mp3_dir / f"{midi_file.stem}.mp3"
            
            # Convert the file
            success = convert_midi_to_mp3(midi_file, output_mp3, soundfont_path)
            
            if success:
                print(json.dumps({
                    "status": "complete",
                    "file": str(output_mp3),
                    "original": str(midi_file)
                }))
            
            # Optionally remove the MIDI file after successful conversion
            # if success and params.get('remove_original', False):
            #     midi_file.unlink()
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))

if __name__ == "__main__":
    main()