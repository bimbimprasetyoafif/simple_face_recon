import os
import json
import face_recognition

KNOWN_FACES_DIR = "Datasets"

datasets = {}

print(f"there are {len([name for name in os.listdir(KNOWN_FACES_DIR)])} data folder ready")
print()
for name in os.listdir(KNOWN_FACES_DIR):
    print(f"{name}")
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(f" --- {filename}")
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        try:
            encoding = face_recognition.face_encodings(image)[0]
            fitur = encoding.tolist()
            datasets.update({name: fitur})
        except Exception as theExcept:
            print(f" ! error encode in {name} = filename {filename} : {theExcept}")

with open('dataset.json', 'w') as fp:
    json.dump(datasets, fp, indent=4)
