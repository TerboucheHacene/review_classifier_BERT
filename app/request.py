import requests
import json

url = "http://localhost:5000/predict/"

reviews = [
    {"review": "I'm not happy"},
    {"review": "I am happy"},
    {"review": "not good and not bad"},
]

payload = json.dumps(reviews[0])
headers = {"Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)

results = [response.json()]
print(results)
for d in results:
    print(
        "Predicted Label {} with a probability of {}".format(
            d["predicted_label"], d["probability"]
        )
    )
