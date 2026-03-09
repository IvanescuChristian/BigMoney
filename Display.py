import gradio as gr
import os
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta

File_His = os.path.join(os.getcwd(),'historical_hourly')
File_Pred = os.path.join(os.getcwd(),'predicted_hourly')
File_S = os.path.join(os.getcwd(),'historical_hourly_Save')

def get_next_filename():
    #name of the predicted file will be the last date +1
    try:
        files = os.listdir(File_His)
        csv_files = [f for f in files if f.endswith('.csv')]
        dates = []
        for name in csv_files:
            try:
                base = os.path.splitext(name)[0]
                date = datetime.strptime(base,"%Y-%m-%d")#got date
                dates.append(date)
            except ValueError:
                continue
        if not dates:
            raise ValueError("No CSV valid in 'historical_hourly'.")
        next_day = max(dates)+timedelta(days=1)
        return next_day.strftime("%Y-%m-%d")+".csv"
    except Exception as e:
        return f"Couldn t identify file : {str(e)}"

def open_folder():
    os.startfile(File_His)
    return "Opened"

def run_prediction():
    try:
        file_name = get_next_filename()
        if "Eroare" in file_name:
            return file_name
        subprocess.run([sys.executable,"FillData.py"],check=True,capture_output=True,text=True)
        subprocess.run([sys.executable,"Predict.py"],check=True,capture_output=True,text=True)
        predicted_file_path = os.path.join(File_Pred,file_name)
        if not os.path.exists(predicted_file_path):
            return f"File {file_name} not found"

        df = pd.read_csv(predicted_file_path)
        return f"File '{file_name}' \n{df.to_string(index=False)}"
    except Exception as e:
        return f"Error {str(e)}"

def compare_files():
    try:
        file_name = get_next_filename()
        if "Eroare" in file_name:
            return file_name

        predicted_file_path = os.path.join(File_Pred, file_name)
        real_file_path = os.path.join(File_S, file_name)
        df_predicted = pd.read_csv(predicted_file_path)
        df_real = pd.read_csv(real_file_path)

        return f"Compared to real - \n{df_real.to_string(index=False)}"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("#CryptoScam")
    with gr.Row():
        btn_open_folder = gr.Button("Select Files (historical_hourly)")
        output_folder = gr.Textbox(label="Floder Status")
    with gr.Row():
        btn_run_pred = gr.Button("Prediction")
        output_pred = gr.Textbox(label="Pred Result",lines=20)
    with gr.Row():
        btn_compare = gr.Button("Compare")
        output_compare = gr.Textbox(label="Compare Result",lines=20)
    btn_open_folder.click(open_folder,outputs=output_folder)
    btn_run_pred.click(run_prediction,outputs=output_pred)
    btn_compare.click(compare_files, outputs=output_compare)
demo.launch()
