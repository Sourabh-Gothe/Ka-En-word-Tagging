import requests
import json
words = ['']
url = "http://www.google.com/transliterate/indic?tlqt=1&langpair=en|kn&text=madu,hogu,barbeda,madide,hogona&&tl_app=1"
r = requests.get(url);
buffer = r.text
buffer.replace(",\n]","]")
buffer = json.loads(buffer)
print(buffer[0]['hws'][0])
print(buffer[1]['hws'][0])
print(buffer[2]['hws'][0])
print(buffer[3]['hws'][0])

