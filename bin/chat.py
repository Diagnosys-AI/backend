import requests
from time import perf_counter

def get_stream(url, call):
    s = requests.Session()
    print("Calling:", url)

    try:
        with s.post(url, json=call, stream=True) as resp:
            resp.raise_for_status()  # Check for HTTP errors
            for line in resp.iter_lines():
                if line:
                    print(line.decode('utf-8'))  # Decode byte to string
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

url = "http://localhost:8000/api/chat"
call = {"messages": [{"role": "user", "content": "What are Dimensional Standards for Letters?"}]}

t1 = perf_counter()
get_stream(url, call)
t2 = perf_counter()
print(f"Response time: {t2 - t1} s")
