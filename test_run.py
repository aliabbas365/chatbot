import json
import pandas as pd
from server import app

client = app.test_client()

print('GET /')
r = client.get('/')
print(r.status_code, r.get_json())

print('\nGET /features')
r = client.get('/features')
print(r.status_code, r.get_json())

print('\nGET /model-info')
r = client.get('/model-info')
print(r.status_code, r.get_json())

# prepare sample payload from DATA.csv
row = pd.read_csv('DATA.csv').iloc[[0]]
payload = row.to_dict(orient='records')
print('\nSample payload prepared (first row)')
print(json.dumps(payload, indent=2))

print('\nPOST /preview')
r = client.post('/preview', json=payload)
print(r.status_code, r.get_json())

print('\nPOST /predict')
r = client.post('/predict', json=payload)
print(r.status_code, r.get_json())
