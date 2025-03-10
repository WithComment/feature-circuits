from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/stsb-TinyBERT-L4')
scores = model.predict([('I like sandwich', 'I don\'t like toast'),
                       ('It is a beautiful day', 'It is a gloomy day')])

print(scores)
