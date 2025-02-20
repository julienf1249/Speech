import datasets

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import pickle
import librosa
import torch
import torchgmm.bayes as bayes

def group_samples_by_speaker(input_dataset):
    """
    Groups the input dataset by speaker, creating a dictionary mapping speaker_id to a list of samples.

    Parameters:
      - input_dataset: A Hugging Face Dataset 

    Returns:
      - A dictionary mapping each speaker_id to the list of samples for that speaker.
    """
    speaker_to_samples = {}
    for sample in input_dataset:
        spk = sample["speaker_id"]
        speaker_to_samples.setdefault(spk, []).append(sample)
    return speaker_to_samples

def create_balanced_duration_dataset_from_groups(speaker_to_samples, target_duration=3600):
    """
    Creates a balanced duration dataset (up to target_duration seconds) using pre-grouped samples by speaker.
    Samples are added whole (no cutting) using a round-robin strategy over the speakers.

    Parameters:
      - speaker_to_samples: A dictionary mapping speaker_id to lists of samples.
      - target_duration: Desired total duration in seconds (e.g., 3600 for 1h, 18000 for 5h).

    Returns:
      - A new Hugging Face Dataset containing the selected samples.
    """
    # Initialize pointers for each speaker group.
    pointers = {spk: 0 for spk in speaker_to_samples}
    selected_samples = []
    total_duration = 0.0

    # Get a sorted list of speaker IDs (you can randomize this order if desired).
    speaker_ids = sorted(speaker_to_samples.keys())

    # Round-robin selection loop.
    while total_duration < target_duration:
        added_in_round = False  # Track whether any sample was added in this round.
        for spk in speaker_ids:
            idx = pointers[spk]
            samples = speaker_to_samples[spk]
            # Try to find a sample from this speaker that can be added without exceeding target_duration.
            while idx < len(samples):
                sample = samples[idx]
                audio_info = sample["audio"]
                waveform = audio_info["array"]
                sample_rate = audio_info["sampling_rate"]
                sample_duration = len(waveform) / sample_rate

                if total_duration + sample_duration <= target_duration:
                    selected_samples.append(sample)
                    total_duration += sample_duration
                    pointers[spk] = idx + 1  # Advance the pointer for this speaker.
                    added_in_round = True
                    break  # Move on to the next speaker.
                else:
                    # This sample is too long to add; skip it.
                    idx += 1
                    pointers[spk] = idx

        # If no sample was added in a complete round, then no further samples fit; break the loop.
        if not added_in_round:
            break

    print(f"Total accumulated duration: {total_duration:.2f} seconds")

    # Create a new Hugging Face Dataset from the selected samples.
    new_dataset = datasets.Dataset.from_list(selected_samples)
    return new_dataset


def compute_mfcc(audio_array, sample_rate, n_mfcc=13, n_fft=400, hop_length=160):
    """
    Compute MFCC features from an audio waveform.
    
    Parameters:
      - audio_array (np.array): The audio waveform.
      - sample_rate (int): The sampling rate (in Hz).
      - n_mfcc (int): Number of MFCC coefficients to return.
      - n_fft (int): The FFT window size (in samples). For 25 ms at 16 kHz, n_fft ≈ 400.
      - hop_length (int): The hop length (in samples). For 10 ms at 16 kHz, hop_length ≈ 160.
    
    Returns:
      - mfcc (np.array): MFCC features of shape (n_frames, n_mfcc).
    """
    mfcc = librosa.feature.mfcc(
        y=audio_array,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Transpose so that each row corresponds to one time frame
    mfcc = mfcc.T  # shape: (n_frames, n_mfcc)
    
    # Apply Cepstral Mean and Variance Normalization (CMVN)
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    mfcc_norm = (mfcc - mean) / std
    
    return mfcc_norm



def add_mfcc_to_sample(sample):
    """
    Processes a single sample: computes the MFCC features for the audio and adds
    them as a new field "mfcc" in the sample.
    """
    audio_info = sample["audio"]
    # Ensure the waveform is a NumPy array.
    waveform = np.array(audio_info["array"])
    sample_rate = audio_info["sampling_rate"]

    # Compute MFCC features.
    mfcc = compute_mfcc(waveform, sample_rate)
    
    # Add MFCC to the sample (converted to list for serialization compatibility).
    sample["mfcc"] = mfcc.tolist()
    return sample


def save_dataset_with_mfcc(dataset, name):
    """
    Adds MFCC features to all samples in the dataset and saves the updated dataset to a new file.
    """
    # Add MFCC features to all samples in the dataset.
    dataset_with_mfcc = dataset.map(add_mfcc_to_sample)
    
    # Save the updated dataset to a new file.
    with open(f"{name}.pkl", "wb") as f:
      pickle.dump(dataset_with_mfcc, f)
    print(f"Dataset with MFCC features saved as {name}")

def load_dataset_with_mfcc(name):
    """
    Loads a dataset with MFCC features from a file.
    """
    with open(f"{name}.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset

