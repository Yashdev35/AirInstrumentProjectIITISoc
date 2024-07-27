

from flask import Flask, render_template, redirect, url_for
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_script/<script_name>')
def run_script(script_name):
    try:
        script_path = os.path.join(os.getcwd(), script_name)
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{script_name} ran successfully")
            print(result.stdout)
        else:
            print(f"Error running {script_name}")
            print(result.stderr)
    except Exception as e:
        print(f"An error occurred: {e}")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)



