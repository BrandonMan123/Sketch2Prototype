# How to run

1. Download our milk frother dataset here : #insert dataset link here#
2. Run `pip install -r requirements.txt`
3. Create an OpenAI api key. Create a file named `api_key.py` and write `api_key = <YOUR API KEY>`
4. Run `python3 sketch2prototype.py --input_dir data --output_dir augmented_dataset`
5. Run `python3 compute_clip.py` for metrics

For our generated dataset, you can have a look here: #insert dataset link here#. Note that you can generate at most 100 images per day as of writing since OpenAI only allows 100 requests to GPT4-V.