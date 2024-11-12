from flask import Flask, request, render_template
import random
import requests
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification,
    BartTokenizer,
    TFBartForConditionalGeneration,
)
import tensorflow as tf
import nltk
import os
import plotly.graph_objs as go
import plotly.io as pio
from markupsafe import Markup

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Ensure TensorFlow uses only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load user agents from file
with open("user_agents.txt", "r") as f:
    user_agents = [line.strip() for line in f.readlines()]

# Load proxy list from URL
"""proxy_url = (
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt"
)
proxy_list = requests.get(proxy_url).text.split("\n")
proxy_list = [proxy.strip() for proxy in proxy_list if proxy.strip()]"""

# Shuffle the user agents and proxy list to increase randomness
random.shuffle(user_agents)
# random.shuffle(proxy_list)


# Function to select a random user agent
def get_random_user_agent():
    return random.choice(user_agents)


# Function to select a random proxy
"""def get_random_proxy():
    return {"http": random.choice(proxy_list), "https": random.choice(proxy_list)}"""


# Set custom headers for requests
def get_custom_headers():
    return {
        "User-Agent": get_random_user_agent(),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }


# Function to validate TripAdvisor URL
def is_valid_tripadvisor_url(base_url):
    pattern = re.compile(
        r"https://www\.tripadvisor\.com/(Hotel_Review|Restaurant_Review)-g\d+-d\d+-Reviews-.*\.html"
    )
    return bool(pattern.match(base_url))


# Function to generate hotel page URL
def generate_pageurl_hotel(base_url, page_number):
    if page_number == 0:
        return base_url
    offset = page_number * 10
    return base_url.replace("Reviews-", f"Reviews-or{offset}-")


# Function to generate restaurant page URL
def generate_pageurl_resta(base_url, page_number):
    if page_number == 0:
        return base_url
    offset = page_number * 15
    return base_url.replace("Reviews-", f"Reviews-or{offset}-")


result = []


def generate_pie_chart(sentiment_summary):
    labels = ["Positive Reviews", "Negative Reviews"]
    values = [
        sentiment_summary["Positive Reviews"],
        sentiment_summary["Negative Reviews"],
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                hoverinfo="label+percent+value",
                textinfo="label+percent",
                textfont_size=20,
                marker=dict(line=dict(color="#000000", width=2)),
                pull=[0.1, 0],
                rotation=90,
                sort=False,  # Keep the order of sections
            )
        ]
    )

    fig.update_traces(
        pull=[0.1, 0],  # Emphasize positive reviews
        rotation=90,
        marker=dict(line=dict(color="#000000", width=2)),
        hoverinfo="label+percent+value",
    )

    fig.update_layout(
        title_text="Sentiment Analysis",
        title_x=0.5,
        title_font=dict(size=24, family="Montserrat"),
        legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            yanchor="bottom",
            y=-0.2,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="#f5f5f5",
        plot_bgcolor="#fff",
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    {
                        "label": "Reset View",
                        "method": "relayout",
                        "args": [{"showlegend": True}],
                    },
                ],
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
    )

    pie_chart_html = pio.to_html(fig, full_html=False)
    return pie_chart_html


# Function to scrape hotel reviews
def scrape_reviews_hotel(page_url):
    success = False
    for _ in range(5):  # Try 5 different proxies
        try:
            response = requests.get(page_url, headers=get_custom_headers())
            if response.status_code == 404:
                print(f"Page not found: {page_url}")
                return False
            soup = BeautifulSoup(response.text, "html.parser")
            reviews = soup.find_all("div", {"class": "fIrGe _T"})
            if not reviews:
                print(f"No reviews found on page: {page_url}")
                return False
            for review in reviews:
                review_text_span = review.find("span", {"class": "orRIx Ci _a C"})
                if review_text_span:
                    review_text = review_text_span.text.strip()
                    result.append(review_text)
            success = True
            break
        except Exception as e:
            print(f"Request failed with proxy. Error: {e}")
    return success


# Function to scrape restaurant reviews
def scrape_reviews_resta(page_url):
    success = False
    for _ in range(5):  # Try 5 different proxies
        try:
            response = requests.get(page_url, headers=get_custom_headers())
            if response.status_code == 404:
                print(f"Page not found: {page_url}")
                return False
            soup = BeautifulSoup(response.text, "html.parser")
            reviews = soup.find_all("div", {"class": "biGQs _P pZUbB KxBGd"})
            if not reviews:
                print(f"No reviews found on page: {page_url}")
                return False
            for review in reviews:
                review_text_span = review.find("span", {"class": "JguWG"})
                if review_text_span:
                    review_text = review_text_span.text.strip()
                    result.append(review_text)
            success = True
            break
        except Exception as e:
            print(f"Request failed with proxy. Error: {e}")
    return success


# Function to extract the name from the URL
def extract_name_from_url(url):
    match = re.search(r"Reviews-(.*)-", url)
    if match:
        return match.group(1)
    return "reviews"


# Function to preprocess text for sentiment analysis
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = " ".join(
        [stemmer.stem(word) for word in text.split() if word not in stop_words]
    )
    return text


# Function to perform sentiment analysis
def sentiment_analysis(reviews):
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = TFDistilBertForSequenceClassification.from_pretrained("sentiment")

    sentiments = []
    for review in reviews:
        inputs = tokenizer(
            review, return_tensors="tf", truncation=True, padding=True, max_length=512
        )
        outputs = model(inputs)
        prediction = tf.nn.softmax(outputs.logits, axis=1)
        sentiment = tf.argmax(prediction, axis=1).numpy()[0]
        sentiments.append(sentiment)

    return sentiments


# Function to summarize reviews
def summarize_reviews(df):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    reviews_text = " ".join(df["Review"].tolist())
    inputs = tokenizer(
        reviews_text, return_tensors="tf", max_length=1024, truncation=True
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=300,
        min_length=200,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        base_url = request.form["url"].strip()
        if not is_valid_tripadvisor_url(base_url):
            return render_template(
                "index_final.html",
                error="Invalid URL. Please enter a valid Tripadvisor Hotel or Restaurant URL.",
            )

        global result
        result = []
        name = extract_name_from_url(base_url)

        page_number = 0
        while True:
            if "Hotel_Review" in base_url:
                page_url = generate_pageurl_hotel(base_url, page_number)
                has_reviews = scrape_reviews_hotel(page_url)
            else:
                page_url = generate_pageurl_resta(base_url, page_number)
                has_reviews = scrape_reviews_resta(page_url)

            if not has_reviews:
                break

            page_number += 1
            time.sleep(2)  # Sleep to avoid being blocked by the server

        if not result:
            return render_template("index_final.html", error="No reviews found.")

        df = pd.DataFrame(result, columns=["Review"])
        df["Processed_Review"] = df["Review"].apply(preprocess_text)
        df["Sentiment"] = sentiment_analysis(df["Processed_Review"])

        positive_reviews = df[df["Sentiment"] == 1]
        negative_reviews = df[df["Sentiment"] == 0]

        sentiment_summary = {
            "Total Reviews": len(df),
            "Positive Reviews": len(positive_reviews),
            "Negative Reviews": len(negative_reviews),
            "Positive Percentage": len(positive_reviews) / len(df) * 100,
            "Negative Percentage": len(negative_reviews) / len(df) * 100,
        }

        summary = summarize_reviews(df)

        # Generate the pie chart
        pie_chart_html = generate_pie_chart(sentiment_summary)

        return render_template(
            "index_final.html",
            reviews=df.to_dict(orient="records"),
            summary=summary,
            sentiment_summary=sentiment_summary,
            pie_chart=Markup(pie_chart_html),  # Pass the pie chart to the template
        )

    return render_template("index_final.html")


# Hardcoded example route for demonstration
@app.route("/example", methods=["GET"])
def example():
    # Load the data from the CSV file
    df = pd.read_csv(r"C:\Users\Ishan\Downloads\output_sentiment.csv")
    # Assume CSV has 'Review' and 'Sentiment' columns
    positive_reviews = df[df["Sentiment"] == 1]
    negative_reviews = df[df["Sentiment"] == 0]

    sentiment_summary = {
        "Total Reviews": len(df),
        "Positive Reviews": len(positive_reviews),
        "Negative Reviews": len(negative_reviews),
        "Positive Percentage": len(positive_reviews) / len(df) * 100,
        "Negative Percentage": len(negative_reviews) / len(df) * 100,
    }

    # Hardcoded summary for demonstration
    summary = "Located in the middle of the mall road, this Chinese/Tibetan eatery is a must visit. They serve authentic, tasty and hygienic food with generous portions, the food seems slightly overpriced and the place is also a bit crammed. The only drawback is that its pricey but the potion they serve is good and the quality is good as well. The chicken in the chicken momos was hard as a rock. On complaining no reaction from the restaurant staff All good ordered was below average & is not worth the money. Do not visit this place we had the pleasure of eating at chopsticks in 2013 for the first time and we ended up at this Manali mall road restaurant again after 7 odd year and they did not disappoint on either occasion. a great place to dine, fantastic ambience, delicious food. Icing on the carke is their extremely courteous staff. you will definitely have a good experience. Located extremely good location just on Manali Mall road, a bit different than any other restaurants nearby. You can reach here by auto and it's on way to hadimba devi temple. We were there for almost 3 times to have lunch/dinner, one evening we had a get together with friends and we tried trout fish fry and roasted with some beers. It was amazing. Very well behaved and friendly stuff. Excellent location juston mall road itself. Only thing the space it small so you may have to wait a little bit during peak hours in peak season. It was neither too pricey nor very cheap. So, if you are a lover of Chinese/Asian cuisine, Chopsticks is the place. Not worthy of visiting. The food was authentic and tasty.  A very good place to get authentic Tibetan food though it is having lot of varieties. We were at the restaurant for almost three times to Have lunch/Dinner, and we were there with friends. We ordered veg wanton for starters and a glass of Rio wine to go with it. For our main course, I ordered prawns fried rice and my friend ordered ting-mo and mix vegetables curry. I loved the wanton and the prawn's fried rice quite delicious. The Rio wine was surprisingly tasty. The restaurant was very close to the kitchen and to the place where they keep the kitchen stuff. They have to cross the washroom from kitchen to the dining hall to bring the food. It look very unhygienics. The owner needs to do more than just serving a food to excel people experience."

    # Generate the pie chart
    pie_chart_html = generate_pie_chart(sentiment_summary)

    return render_template(
        "index_final.html",
        reviews=df.to_dict(orient="records"),
        summary=summary,
        sentiment_summary=sentiment_summary,
        pie_chart=Markup(pie_chart_html),  # Pass the pie chart to the template
    )


if __name__ == "__main__":
    app.run(debug=True)
