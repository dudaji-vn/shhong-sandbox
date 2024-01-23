import io
import random
import requests


def run(filename='img1.jpg'):
    port = random.randint(4999, 5000)
    resp = requests.post( 
            f"http://localhost:{port}/preprocess", 
            files={"file": open(f'./{filename}','rb')})
    buff = io.BytesIO(resp.content)

    headers = {'Content-Type': 'application/octet-stream'}
    resp = requests.post(
            "http://localhost:5000/predict",
            data=buff, 
            headers=headers)

if __name__=="__main__":
    run()
