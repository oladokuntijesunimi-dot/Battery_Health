from flask import Flask, render_template, request
import os
import numpy as np
import pickle
app = Flask('__name__')

with open('battery_percent_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('rec_action_model.pkl', 'rb') as f:
    class_model = pickle.load(f)

with open('background_encoder.pkl', 'rb') as f:
    background_enc = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        device_age = int(request.form.get('device_age'))
        battery_capacity = float(request.form.get('battery_capacity'))
        screen_time = float(request.form.get('screen_time'))
        charging_cycles = float(request.form.get('charging_cycles'))
        battery_temperature = float(request.form.get('battery_temperature'))
        fast_charging = float(request.form.get('fast_charging'))
        overnight_charging = int(request.form.get('overnight_charging'))
        gaming_hours = float(request.form.get('gaming_hours'))
        streaming_hours = float(request.form.get('streaming_hours'))
        background_app_usage = request.form.get('background_app_usage')
        charging_habit = int(request.form.get('charging_habit'))
        usage_intensity = int(request.form.get('usage_intensity'))
        stress_index = float(request.form.get('stress_index'))

        background_app_usage_encoded = background_enc.transform(np.array([background_app_usage])).flatten()

        datapoint = np.array([[device_age, battery_capacity, screen_time, charging_cycles, battery_temperature, fast_charging, overnight_charging, gaming_hours, streaming_hours, *background_app_usage_encoded, charging_habit, usage_intensity, stress_index]])
        battery_percent = reg_model.predict(datapoint)
        action = class_model.predict(datapoint)
        #print(battery_percent, action)
        return render_template('index.html', battery_percentage=round(battery_percent[0], 2), rec_action=action[0])
if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)
