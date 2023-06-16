# REST API
REST API for access model

### Request

`GET /`

    curl -X 'GET' \
      'https://model-api-uiwtfdqmea-et.a.run.app/' \
      -H 'accept: application/json'

### Response

"Chatbot API"

### Request

`POST /predict_text`

    curl -X 'POST' \
    'https://model-api-uiwtfdqmea-et.a.run.app/predict_text' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "text": "good morning"
    }'

### Response

"Good morning. I hope you had a good night's sleep. How are you feeling today? "
