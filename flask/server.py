from flask import Flask, request, jsonify

app = Flask(__name__)

def calPer(marks, total_marks):
    return ((marks/total_marks)*100)

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    value = int(data.get('marks_obtained', 0))
    total = int(data.get('total_marks', 100))
    
    if total == 0:
        return jsonify({'error': 'Total cannot be zero'}), 400
    
    percentage = (value / total) * 100
    
    result = {
        'marks_obtained': value,
        'total_marks': total,
        'percentage': round(percentage, 2)
    }
    return jsonify(result)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
