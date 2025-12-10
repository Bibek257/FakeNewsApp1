# app.py - Flask app for Fake News Detection
import sys
from flask import Flask, request, render_template
from src.exception import CustomException
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logging
# from src.agents.ai_agents import AIAgent   # ✅ Import class, not function
# from google.protobuf import message_factory as mf
# if not hasattr(mf.MessageFactory, 'GetPrototype'):
    # mf.MessageFactory.GetPrototype = mf.MessageFactory.GetMessageClass
# Initialize Flask app
app = Flask(__name__)

# Initialize PredictionPipeline once
predictor = PredictionPipeline()

#Initialize AI Agent once (loads sentence-transformer only once)
# agent = AIAgent()

@app.route("/", methods=["GET"])
def home() -> str:
    """
    Renders the home page with input form.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict() -> str:
    """
    Receives news text from the form, performs prediction, and returns the result with confidence percentage.
    """
    try:
        news_text: str = request.form.get("news_text", "").strip()
        if not news_text:
            logging.warning("No news text provided by the user")
            return render_template("index.html", prediction="Please enter some news text.", confidence=None)

        logging.info("Received news text for prediction")
        # Updated to unpack three values
        label, conf_fake, conf_real = predictor.predict(news_text)

        # Determine result and confidence text
        if label == 1:
            result = "Real News"
            confidence = f"{conf_real:.2f}% confident"
        else:
            result = "Fake News"
            confidence = f"{conf_fake:.2f}% confident"

        # ✅ Call the agent’s search method
        # related_articles = agent.search(news_text)

        logging.info(f"Prediction result: {result}, Confidence: {confidence}")
        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence
            # related_articles=related_articles
        )

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise CustomException(e, sys)

@app.route("/predict_batch", methods=["POST"])
def predict_batch() -> str:
    """
    Receives multiple news texts (comma separated) and predicts each one.
    """
    try:
        texts: str = request.form.get("batch_news_text", "").strip()
        if not texts:
            return render_template("index.html", prediction="Please enter news texts for batch prediction.", confidence=None)

        news_list = [t.strip() for t in texts.split(",") if t.strip()]
        results = predictor.predict_batch(news_list)

        formatted_results = []
        for text, (label, conf_fake, conf_real) in zip(news_list, results):
            if label == 1:
                res = f"Real News ({conf_real:.2f}% confident)"
            else:
                res = f"Fake News ({conf_fake:.2f}% confident)"
            formatted_results.append(f"{text[:50]}... : {res}")

        logging.info("Batch prediction completed")
        return render_template("index.html", batch_results=formatted_results)

    except Exception as e:
        logging.error(f"Error during batch prediction: {e}", exc_info=True)
        raise CustomException(e, sys)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

    