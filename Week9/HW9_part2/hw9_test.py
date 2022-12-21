import requests

#url = 'http://localhost:8080/Week9/'
url = "http://localhost:8080/2015-03-31/functions/function/invocations"
#url = "https://h3cizmdxnk.execute-api.eu-central-1.amazonaws.com/test/predict"
data = {'url' : 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}

result = requests.post(url, json = data).json()
print(result)