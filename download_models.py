import os
import gdown

# File ID and target path mapping
files = {
    "audio_model.keras": "1qBEIYEU4HKPysMaFwzOOBz_67gnpsESV",
    "best_video_model.pt": "1tE5ISadseu_EKa-gUe2tp9kfmuKTJwiE",
    "image_model.pth": "1wuzqk1IgRGH3QZ5Y7yvpEVDB1qaTsrRp",
    "model_video.hdf5": "1hjkgzZReNnZNKzHzrMS9MTz6E2qufz0d"
}

def download_models():
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    download_models()
