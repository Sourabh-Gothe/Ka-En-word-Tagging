from googletrans import Translator

translator = Translator()
with open('new.txt', encoding="utf8") as f:
	for line in f:
		line = line[:-1]
		translations = translator.translate(line,dest='en')
		print(translations.origin, ' -> ', translations.text)
