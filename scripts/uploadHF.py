from huggingface_hub import login, upload_folder
from huggingface_hub import HfApi
import os

# (optional) Login with your Hugging Face credentials
login()

api = HfApi()

# Define your local folder path and repository ID
local_folder_path = "./SN7_YOLO_dataset"
repo_id = "Eponitaxd/SN7_YOLO_dataset" # e.g., "Wauplin/Docmatix"

# Optional: For maximum performance, set the HF_XET_HIGH_PERFORMANCE environment variable
# This uses the Rust-based hf_xet backend which is enabled by default in huggingface_hub
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# Upload the folder
api.upload_large_folder(
    folder_path=local_folder_path,
    repo_id=repo_id,
    repo_type="dataset",
)
