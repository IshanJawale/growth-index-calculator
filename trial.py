from flask import Flask, request, jsonify

app = Flask(__name__)

def calFact(marks):
    fact = 1
    while marks > 1:
        fact *= marks
        marks -= 1
    return fact

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    value = int(data.get('marks_obtained', 0))
    
    
    fact = calFact(value)
    
    result = {
        'marks_obtained': value,
        'factorial': round(fact, 2)
    }
    return jsonify(result)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
