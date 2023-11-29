def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def save_json(json_dict, filepath):
    json_obj = json.dumps(json_dict, indent=4)
    with open(filepath, "w") as f:
        f.write(json_obj) 

def load_sketch(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')