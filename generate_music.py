import torch
import pretty_midi
import numpy as np
import pickle
import zlib
from pathlib import Path
import random
import argparse

class MusicLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=384):
        super(MusicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Bidirectional LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = torch.nn.Dropout(0.2)

        self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = torch.nn.Dropout(0.2)

        self.lstm3 = torch.nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.dropout3 = torch.nn.Dropout(0.2)

        self.note_layer = torch.nn.Linear(hidden_size, input_size)
        self.velocity_layer = torch.nn.Linear(hidden_size, input_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        notes, velocities = x[:, :, :self.input_size], x[:, :, self.input_size:]
        x = torch.cat([notes, velocities], dim=-1)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = self.dropout3(x)

        note_out = self.sigmoid(self.note_layer(x))
        velocity_out = self.sigmoid(self.velocity_layer(x))

        return note_out, velocity_out

class MusicGenerator:
    def __init__(self, fs=32, min_active_notes=1, max_active_notes=8):
        self.fs = fs
        self.min_active_notes = min_active_notes
        self.max_active_notes = max_active_notes
        self.activity_window = []
        self.activity_window_size = 32

        # Historical context
        self.note_history = set()
        self.last_notes = []
        self.note_repeat_threshold = 4

    def get_dynamic_threshold(self, step, beat_position, recent_activity):
        """Dynamic threshold based on musical position and context"""
        if beat_position == 0:  # Bar start
            base_threshold = 0.35
        elif beat_position % self.fs == 0:  # Beat start
            base_threshold = 0.40
        elif beat_position % (self.fs // 2) == 0:  # Eighth notes
            base_threshold = 0.45
        else:
            base_threshold = 0.50

        # Adjust based on recent activity
        if recent_activity < 2:
            base_threshold *= 0.8
        elif recent_activity > 6:
            base_threshold *= 1.2

        return base_threshold

    def select_notes(self, probs, k):
        """Select notes with diversity control"""
        top_k = min(max(k * 3, 6), probs.shape[-1])  # Get more candidates
        top_values, top_indices = torch.topk(probs, k=top_k)

        # Convert to numpy for manipulation
        indices = top_indices.cpu().numpy()
        values = top_values.cpu().numpy()

        # Avoid recent notes unless they have high probability
        if len(self.last_notes) > 0:
            for idx in indices:
                if idx in self.last_notes[-self.note_repeat_threshold:]:
                    values[indices == idx] *= 0.8

        # Select with probability weighting
        selected = np.random.choice(
            indices,
            size=min(k, len(indices)),
            replace=False,
            p=values/np.sum(values)
        )

        return selected

    def generate(self, model, seed_sequence, seed_velocities, length=1024, device='cuda'):
        model.eval()
        with torch.no_grad():
            current_sequence = torch.cat([seed_sequence, seed_velocities], dim=-1).to(device)
            output_notes = seed_sequence.clone()
            output_velocities = seed_velocities.clone()

            print(f"\nGeneration Config:")
            print(f"Sampling rate: {self.fs} steps/second")
            print(f"Sequence length: {length} steps")
            print(f"Initial active notes: {torch.sum(seed_sequence[0, -1] > 0.5)}")

            for step in range(length):
                # Get predictions
                pred_notes, pred_vels = model(current_sequence)
                next_notes = pred_notes[:, -1:, :]
                next_vels = pred_vels[:, -1:, :]

                # Log raw predictions periodically
                if step % 100 == 0:
                    print(f"\nStep {step} Predictions:")
                    print(f"Note range: {next_notes.min():.3f} to {next_notes.max():.3f}")
                    print(f"Velocity range: {next_vels.min():.3f} to {next_vels.max():.3f}")

                # Calculate musical position and threshold
                bar_position = step % (self.fs * 4)
                beat_position = bar_position % self.fs
                recent_activity = np.mean(self.activity_window[-4:]) if len(self.activity_window) > 4 else 2

                threshold = self.get_dynamic_threshold(step, beat_position, recent_activity)

                # Generate initial notes
                note_probs = torch.sigmoid(next_notes[0, 0])
                initial_notes = (note_probs > threshold).float()
                active_notes = torch.sum(initial_notes)

                # Adjust note density
                if active_notes < self.min_active_notes:
                    selected_indices = self.select_notes(note_probs, self.min_active_notes)
                    initial_notes.zero_()
                    initial_notes[selected_indices] = 1
                elif active_notes > self.max_active_notes:
                    selected_indices = self.select_notes(note_probs, self.max_active_notes)
                    initial_notes.zero_()
                    initial_notes[selected_indices] = 1

                # Update note history
                active_indices = torch.where(initial_notes > 0)[0]
                self.last_notes.extend(active_indices.cpu().numpy())
                if len(self.last_notes) > self.note_repeat_threshold:
                    self.last_notes = self.last_notes[-self.note_repeat_threshold:]

                # Enhance velocities
                active_indices = torch.where(initial_notes > 0)[0]
                if len(active_indices) > 0:
                    # Scale velocities based on position
                    if beat_position == 0:  # Bar start
                        velocity_boost = 1.4
                    elif beat_position % self.fs == 0:  # Beat start
                        velocity_boost = 1.3
                    elif beat_position % (self.fs // 2) == 0:  # Eighth notes
                        velocity_boost = 1.2
                    else:
                        velocity_boost = 1.1

                    next_vels[0, 0, active_indices] *= velocity_boost
                    next_vels[0, 0, active_indices] = torch.clamp(next_vels[0, 0, active_indices], 0.3, 1.0)

                # Update tracking
                self.activity_window.append(float(len(active_indices)))
                if len(self.activity_window) > self.activity_window_size:
                    self.activity_window.pop(0)

                # Add to sequences
                output_notes = torch.cat([output_notes, initial_notes.unsqueeze(0).unsqueeze(0)], dim=1)
                output_velocities = torch.cat([output_velocities, next_vels], dim=1)

                # Update current sequence
                current_sequence = torch.cat([
                    output_notes[:, -seed_sequence.size(1):, :],
                    output_velocities[:, -seed_sequence.size(1):, :]
                ], dim=-1)

                # Log progress
                if (step + 1) % 100 == 0:
                    avg_vel = float(next_vels[0, 0, active_indices].mean()) if len(active_indices) > 0 else 0
                    print(f"\nGeneration progress: {step + 1}/{length}")
                    print(f"Active notes: {len(active_indices)}")
                    print(f"Recent activity: {recent_activity:.1f}")
                    print(f"Current threshold: {threshold:.2f}")
                    print(f"Average velocity: {avg_vel:.2f}")

        return output_notes.cpu().numpy(), output_velocities.cpu().numpy()

def create_midi_file(notes, velocities, output_file='generated_music.mid', fs=32, initial_tempo=100):
    """Create MIDI file with improved dynamics"""
    if len(notes.shape) == 3:
        notes = notes[0]
        velocities = velocities[0]

    print("\nMIDI Creation Analysis:")
    print(f"Input shape: notes {notes.shape}, velocities {velocities.shape}")
    print(f"Note range: {notes.min():.3f} to {notes.max():.3f}")
    print(f"Velocity range: {velocities.min():.3f} to {velocities.max():.3f}")

    time_per_step = 1/fs
    pm = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    piano = pretty_midi.Instrument(program=0)

    note_states = np.zeros(128, dtype=int)
    note_start_times = np.zeros(128)
    note_velocities = np.zeros(128)

    note_statistics = {
        'total_activations': 0,
        'note_durations': [],
        'velocity_values': [],
        'active_notes_per_step': []
    }

    min_duration = 1/(fs*2)  # Minimum note duration

    for step in range(notes.shape[0]):
        current_time = step * time_per_step
        bar_position = step % (fs * 4)
        beat_position = bar_position % fs

        step_active_notes = 0

        for note_num in range(notes.shape[1]):
            is_active = notes[step, note_num] > 0.5
            current_velocity = velocities[step, note_num]

            if is_active and note_states[note_num] == 0:
                step_active_notes += 1
                note_statistics['total_activations'] += 1

                # Calculate velocity with musical dynamics
                base_velocity = int(current_velocity * 127)
                if bar_position == 0:
                    velocity = min(127, int(base_velocity * 1.4))
                elif beat_position == 0:
                    velocity = min(127, int(base_velocity * 1.3))
                elif beat_position % (fs // 2) == 0:
                    velocity = min(127, int(base_velocity * 1.2))
                else:
                    velocity = base_velocity

                velocity = max(50, min(127, velocity))  # Ensure good velocity range
                note_statistics['velocity_values'].append(velocity)

                note_states[note_num] = 1
                note_start_times[note_num] = current_time
                note_velocities[note_num] = velocity

            elif not is_active and note_states[note_num] == 1:
                duration = current_time - note_start_times[note_num]
                note_statistics['note_durations'].append(duration)

                if duration >= min_duration:
                    if duration < 1/fs:
                        end_time = note_start_times[note_num] + 1/fs
                    else:
                        end_time = current_time

                    note = pretty_midi.Note(
                        velocity=int(note_velocities[note_num]),
                        pitch=note_num,
                        start=note_start_times[note_num],
                        end=end_time
                    )
                    piano.notes.append(note)

                note_states[note_num] = 0

        note_statistics['active_notes_per_step'].append(step_active_notes)

    # Handle still-active notes
    final_time = notes.shape[0] * time_per_step
    active_at_end = 0
    for note_num in range(128):
        if note_states[note_num] == 1:
            active_at_end += 1
            duration = final_time - note_start_times[note_num]
            note_statistics['note_durations'].append(duration)

            if duration >= min_duration:
                note = pretty_midi.Note(
                    velocity=int(note_velocities[note_num]),
                    pitch=note_num,
                    start=note_start_times[note_num],
                    end=final_time
                )
                piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(output_file)

    # Print detailed statistics
    print("\nGeneration Statistics:")
    print(f"Total notes created: {len(piano.notes)}")
    print(f"Duration: {final_time:.1f} seconds")
    print(f"Notes per second: {len(piano.notes)/final_time:.1f}")

    if note_statistics['note_durations']:
        print("\nNote duration statistics:")
        print(f"Mean duration: {np.mean(note_statistics['note_durations']):.3f}s")
        print(f"Min duration: {np.min(note_statistics['note_durations']):.3f}s")
        print(f"Max duration: {np.max(note_statistics['note_durations']):.3f}s")

    if note_statistics['velocity_values']:
        print("\nVelocity statistics:")
        print(f"Mean velocity: {np.mean(note_statistics['velocity_values']):.1f}")
        print(f"Range: {np.min(note_statistics['velocity_values'])} to {np.max(note_statistics['velocity_values'])}")

    active_notes = np.array(note_statistics['active_notes_per_step'])
    print("\nActive notes statistics:")
    print(f"Mean: {np.mean(active_notes):.1f}")
    print(f"Range: {np.min(active_notes)} to {np.max(active_notes)}")

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Music Generation Parameters')
    
    # Model parameters
    parser.add_argument('--hidden-size', type=int, default=384,
                      help='Hidden size for LSTM layers')
    parser.add_argument('--model-path', type=str, 
                      default='music_generator_model.pth',
                      help='Path to the model checkpoint')
    parser.add_argument('--data-path', type=str,
                      default='processed_music_data.pkl',
                      help='Path to the processed music data')
    

    # Generation parameters
    parser.add_argument('--fs', type=int, default=32,
                      help='Sampling rate (steps per second)')
    parser.add_argument('--sequence-length', type=int, default=1024,
                      help='Length of the generated sequence')
    parser.add_argument('--min-active-notes', type=int, default=1,
                      help='Minimum number of active notes')
    parser.add_argument('--max-active-notes', type=int, default=8,
                      help='Maximum number of active notes')
    parser.add_argument('--seed-length', type=int, default=128,
                      help='Length of seed sequence')
    
    # MIDI parameters
    parser.add_argument('--tempo', type=int, default=100,
                      help='Initial tempo for the MIDI file')
    parser.add_argument('--output-file', type=str, default='generated_music.mid',
                      help='Output MIDI file path')
    
    # Other parameters
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for computation')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        
        # Set random seed if provided
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.device == 'cuda':
                torch.cuda.manual_seed(args.seed)
        
        device = torch.device(args.device)
        print(f"Using device: {device}")

        # Load processed data
        print(f"\nLoading data from {args.data_path}...")
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)

        # Load model
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path,
                              map_location=device,
                              weights_only=True)
        input_size = checkpoint['input_size']

        # Initialize model
        model = MusicLSTM(input_size=input_size, hidden_size=args.hidden_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print("Model loaded successfully")

        # Select and prepare seed data
        print("\nPreparing seed sequence...")
        best_seed_idx = None
        best_note_count = 0

        for _ in range(5):
            seed_idx = np.random.randint(len(data['sequences']))
            seed_seq = np.frombuffer(zlib.decompress(data['sequences'][seed_idx]),
                                   dtype=np.uint8).reshape(-1, 128)
            note_count = np.sum(seed_seq)
            if note_count > best_note_count:
                best_seed_idx = seed_idx
                best_note_count = note_count

        seed_idx = best_seed_idx
        print(f"Selected seed {seed_idx} with {best_note_count} notes")

        # Decompress seed data
        seed_seq = np.frombuffer(zlib.decompress(data['sequences'][seed_idx]),
                               dtype=np.uint8).reshape(-1, 128)
        seed_vel = np.frombuffer(zlib.decompress(data['velocities'][seed_idx]),
                               dtype=np.float32).reshape(-1, 128)

        # Make arrays writable and prepare tensors
        seed_seq = seed_seq.copy()
        seed_vel = seed_vel.copy()
        seed_sequence = torch.FloatTensor(seed_seq[:args.seed_length]).unsqueeze(0).to(device)
        seed_velocities = torch.FloatTensor(seed_vel[:args.seed_length]).unsqueeze(0).to(device)

        # Generate music
        print("\nGenerating music...")
        generator = MusicGenerator(
            fs=args.fs,
            min_active_notes=args.min_active_notes,
            max_active_notes=args.max_active_notes
        )

        generated_notes, generated_velocities = generator.generate(
            model=model,
            seed_sequence=seed_sequence,
            seed_velocities=seed_velocities,
            length=args.sequence_length,
            device=device
        )

        # Create MIDI file
        print(f"\nCreating MIDI file: {args.output_file}")
        if create_midi_file(generated_notes, generated_velocities, 
                          args.output_file, args.fs, args.tempo):
            print(f"\nMusic successfully saved to {args.output_file}")

            # Print summary statistics
            total_notes = (generated_notes > 0.5).sum()
            duration = len(generated_notes[0]) / args.fs

            print("\nGeneration Summary:")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Total notes: {total_notes}")
            print(f"Average density: {total_notes / duration:.1f} notes per second")
        else:
            raise Exception("Failed to create MIDI file")

    except Exception as e:
        print(f"\nError during generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
