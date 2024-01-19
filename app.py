# app.py

from flask import Flask, render_template, request
from ml_model import load_data, train_model, predict_probability

app = Flask(__name__)

# Load the data and train the model
df = load_data()
model = train_model(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if request.method == 'POST':
        # Get user input from the form
        parameters = {
            'Follicle No. (R)': float(request.form['Follicle_No_R']),
            'Follicle No. (L)': float(request.form['Follicle_No_L']),
            'Skin darkening (Y/N)': int(request.form['Skin_darkening']),
            'hair growth(Y/N)': int(request.form['hair_growth']),
            'Weight gain(Y/N)': int(request.form['Weight_gain']),
            'Cycle(R/I)': int(request.form['Cycle']),
            'Fast food (Y/N)': int(request.form['Fast_food']),
            'Pimples(Y/N)': int(request.form['Pimples']),
            'AMH(ng/mL)': float(request.form['AMH']),
            'Weight (Kg)': float(request.form['Weight'])
        }

        # Use the trained model to get the probability
        probability = predict_probability(model, parameters)

        # Set the color and text for the bar based on probability
        bar_color = 'green' if probability < 0.5 else 'red'
        probability_text = f'Probability: {probability:.2f}'

        return render_template('result.html', bar_color=bar_color, probability_text=probability_text)

if __name__ == '__main__':
    app.run(debug=True)



