import requests
import json

url = "http://localhost:5000/predict/"

payload = json.dumps(
    {
        "reviews": [
            {"review": "I'm not happy"},
            {"review": "I am happy"},
            {"review": "not good and not bad"},
        ]
    }
)
headers = {'Content-Type': 'application/json'}

response = requests.request("POST", url, headers=headers, data=payload)

results = json.loads(response.json()['data'])
for d in results:
    print(
        "Predicted Label {} with a probability of {}".format(
            d["predicted_label"], d["predicted_label"]
        )
    )
