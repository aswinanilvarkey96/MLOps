import time
import requests
url = 'https://europe-west1-inlaid-goods-337908.cloudfunctions.net/mlops-cloud-function'
payload = {"message": "Hello, General Kenobi"}

for i in range(1000):
   r = requests.get(url, params=payload)
   if i%100 ==0:
       print(f'step{i}')
       print(r)