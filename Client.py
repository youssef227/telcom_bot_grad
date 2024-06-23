import requests

# URL of the Flask server endpoint
url = "https://telecombot-k2fv2pc7mq-uc.a.run.app"

# Payload to send to the server
payload = {
    "text": "محتاج اجدد الباقه"
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Make the POST request
response = requests.post(url, json=payload, headers=headers)

# Print the response
if response.status_code == 200:
    print("Generated Text:", response.json()["generated_text"])
else:
    print("Failed to get a response:", response.status_code, response.text)
