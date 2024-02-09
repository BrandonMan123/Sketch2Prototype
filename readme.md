# Sketch2Prototype
Sketch2Prototype is a framework that converts abstract sketches into 3D-printable prototypes through GPT-4 and DALL-E 3. 

# Updates
9 Feb 2024 - Initial release

# How to run

1. Download our milk frother dataset here : https://www.dropbox.com/scl/fo/2vpqrqkdu6o7uo95it18h/h?rlkey=95c256efvaphhgy5xg4cw4fwo&dl=0 and rename the downloaded folder to dataset_full
2. Run `pip install -r requirements.txt`
3. Create an OpenAI api key. Inside `api_key.py` put your API key into `api_key`
4. Run `python3 sketch2prototype.py --input_dir data --output_dir augmented_dataset`
5. Run `python3 compute_clip.py` for metrics

For our generated dataset, you can have a look here: https://www.dropbox.com/home/decode_lab/Datasets/sketch2prototype. Note that you can generate at most 100 images per day as of writing since OpenAI only allows 100 requests to GPT4-V.
