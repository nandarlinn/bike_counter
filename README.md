# Create Virtual Environment
python3 -m vevn counterenv

# Activate virtual environment
source counterenv/bin/activate

# Download weight file
Download yolov9-e-modify-trained.pt at https://github.com/tuansunday05/fe8kv9?tab=readme-ov-file

# Add data folder to predict
Replace the folder name at SOURCE_FOLDER = "frames/"    

# Run program
python detect.py
