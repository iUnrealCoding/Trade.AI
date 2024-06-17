from flask import Flask, request, render_template, send_from_directory, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from src.pipeline.pre_pipeline import StockDataDownloader, StockDataPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.visualization_pipeline import RunVisPipeline
import pandas as pd

app = Flask(__name__)

# Ticker, period, and interval variables
ticker = 'AAPL'  # Default ticker
period = '3mo'   # Default period
interval = '1h'  # Default interval
refresh_task_enabled = False

# Initialize scheduler
scheduler = BackgroundScheduler(daemon=True)

def scheduled_job():
    if refresh_task_enabled:
        process_stock_data()
        print("Scheduled job executed")

def process_stock_data():
    global ticker, period, interval
    try:
        # Download stock data
        downloader = StockDataDownloader(ticker=ticker, period=period, interval=interval)
        downloader.download_data()

        # Process the downloaded data
        pipeline = StockDataPipeline(ticker, period, interval)
        pipeline.run_pipeline()

        # Train the model
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        # Visualize the data
        vis_pipeline = RunVisPipeline()
        vis_pipeline.run_pipeline(MF=1.0, x=None, y=None)

        graph_filename = 'plot.html'  # Adjust this based on your actual filename
        return jsonify({'graphUrl': f'/renderplot/{graph_filename}'})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index_new.html')  # Render the form

@app.route('/process', methods=['POST'])
def process():
    global ticker, period, interval
    try:
        data = request.json  # Access JSON data sent from frontend
        ticker = data.get('ticker', 'AAPL')
        period = data.get('period', '3mo')
        interval = data.get('interval', '1h')
        
        # Process once
        return process_stock_data()
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/toggle_refresh', methods=['POST'])
def toggle_refresh():
    global refresh_task_enabled
    refresh_task_enabled = not refresh_task_enabled
    if refresh_task_enabled:
        if not scheduler.get_job('scheduled_job'):
            scheduler.add_job(scheduled_job, 'interval', seconds=60, id='scheduled_job')
        scheduler.start()
    else:
        scheduler.remove_all_jobs()
    return jsonify({'status': 'success', 'refresh_task_enabled': refresh_task_enabled})

@app.route('/updateplot', methods=['POST'])
def update_plot():
    try:
        data = request.json
        MF = float(data.get('MF'))
        x = pd.to_datetime(data.get('x'), format='%Y-%m-%dT%H:%M', utc=True)
        y = pd.to_datetime(data.get('y'), format='%Y-%m-%dT%H:%M', utc=True)

        vis_pipeline = RunVisPipeline()
        vis_pipeline.run_pipeline(MF=MF, x=x, y=y)

        graph_filename = 'plot.html'  # Adjust this based on your actual filename
        return jsonify({'graphUrl': f'/renderplot/{graph_filename}'})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/renderplot/<path:plot_filename>')
def render_plot(plot_filename):
    try:
        # Serve the generated plot file from the templates directory
        return send_from_directory('templates', plot_filename)

    except Exception as e:
        print("Error:", e)
        return render_template('error.html', error="An error occurred while rendering the plot.")

@app.route('/form')
def home():
    return render_template('form_new.html')  # Render the home page or dashboard

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Mock data processing and prediction (replace with actual logic)
        data = CustomData(
            Year=int(request.form.get('year', 0)),
            Month=int(request.form.get('month', 0)),
            Day=int(request.form.get('day', 0)),
            Hour=int(request.form.get('hour', 0)),
            Minute=int(request.form.get('minute', 0)),
            EMASignal=int(request.form.get('emasignal', 0)),
            isPivot=int(request.form.get('ispivot', 0)),
            CHOCH_pattern_detected=int(request.form.get('choch_pattern_detected', 0)),
            fibonacci_signal=int(request.form.get('fibonacci_signal', 0)),
            SL=float(request.form.get('sl', 0)),
            TP=float(request.form.get('tp', 0)),
            MinSwing=float(request.form.get('minswing', 0)),
            MaxSwing=float(request.form.get('maxswing', 0)),
            LBD_detected=int(request.form.get('lbd_detected', 0)),
            LBH_detected=int(request.form.get('lbh_detected', 0)),
            SR_signal=int(request.form.get('sr_signal', 0)),
            isBreakOut=int(request.form.get('isbreakout', 0)),
            candlestick_signal=int(request.form.get('candlestick_signal', 0)),
            result=int(request.form.get('result', 0)),
            signal1=int(request.form.get('signal1', 0)),
            buy_signal=int(request.form.get('buy_signal', 0)),
            Position=int(request.form.get('position', 0)),
            sell_signal=int(request.form.get('sell_signal', 0)),
            fractal_high=float(request.form.get('fractal_high', 0)),
            fractal_low=float(request.form.get('fractal_low', 0)),
            buy_signal1=int(request.form.get('buy_signal1', 0)),
            sell_signal1=int(request.form.get('sell_signal1', 0)),
            fractals_high=int(request.form.get('fractals_high', 0)),
            fractals_low=int(request.form.get('fractals_low', 0)),
            VSignal=int(request.form.get('vsignal', 0)),
            PriceSignal=int(request.form.get('pricesignal', 0)),
            TotSignal=int(request.form.get('totsignal', 0)),
            SLSignal=int(request.form.get('slsignal', 0)),
            grid_signal=int(request.form.get('grid_signal', 0)),
            ordersignal=int(request.form.get('ordersignal', 0)),
            SLSignal_heiken=float(request.form.get('slsignal_heiken', 0)),
            EMASignal1=int(request.form.get('emasignal1', 0)),
            long_signal=int(request.form.get('long_signal', 0)),
            martiangle_signal=int(request.form.get('martiangle_signal', 0))
        )
        
        # Perform prediction
        predict_pipeline = PredictPipeline()
        input_data = data.get_data_as_array().astype(float)  # Convert to float array
        results = predict_pipeline.predict(input_data)
        print("Predicted Results:", results)

        return render_template('form_new.html', results=int(results[0]))

    except Exception as e:
        print("Error:", e)
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
