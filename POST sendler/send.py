import requests

files = {'file': open('image.tif', 'rb')}
r = requests.post("http://127.0.0.1:5000/5", files=files)
print(r.text)

# text = "Bob Ross lived in Florida"
# t = requests.post("http://127.0.0.1:5000/4", data=text)
# print(t.text)
