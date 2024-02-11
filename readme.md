# Sketch2Prototype
Sketch2Prototype is a framework that converts abstract sketches into 3D-printable prototypes through GPT-4 and DALL-E 3. 

# Updates
9 Feb 2024 - Initial release

# How to run

1. Download our milk frother dataset here : https://www.dropbox.com/scl/fo/2vpqrqkdu6o7uo95it18h/h?rlkey=95c256efvaphhgy5xg4cw4fwo&dl=0 and rename the downloaded folder to `dataset_full`
2. Run `pip install -r requirements.txt`
3. Create an OpenAI api key. Inside `api_key.py` put your API key into `api_key`
4. Run `python3 sketch2prototype.py --input_dir data --output_dir augmented_dataset`

## Running Metrics
1. Move sketch_drawings.csv into you generated dataset
2. Run `python3 compute_clip.py`. The `clip_scores` directory will be generated and results will be stored there.

For our generated dataset, you can have a look here: https://www.dropbox.com/home/decode_lab/Datasets/sketch2prototype. Note that you can generate at most 100 images per day as of writing since OpenAI only allows 100 requests to GPT4-V.

# About the dataset
## Base milk frother dataset
The unprocessed milk frother dataset consists of 1089 images denoted as Page-X.png where X is the sample ID. The dataset also consists of a csv file named sketch_drawing.csv that includes Image_ID, that denotes the id of the image, and text, that denotes the text description of the image. The other fields in the dataset do not matter for the purposes of this project.

## Augmented milk frother dataset
After running the sketch2prototype framework on the dataset, we obtain an augmented dataset. Each sample has its own directory with the following contents:
1. Directory of 4 generated images
2. dalle_response.json - a log of the prompts used to generate the images
3. original.png - the original milk frother sketch used to generate the 4 images

Additionally, sketch_drawing.csv is in the folder.

# Acknowledgements
We would like to thank Scarlett Miller's group at Penn State for providing access to the milk frother dataset. https://sites.psu.edu/creativitymetrics/2018/08/24/summary-of-idea-repository/
