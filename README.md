# Sentiment Analysis of Product Reviews using BERT 

* Analyse and visualize the datasets to detect statistical bias 
* Build and train a RoBERTa model to classify the sentiment of the reviews as **positive**, **negative** or **neutral**
* Deploy the trained model using FastAPI and Docker
* Streamlit App to showcase the model

# How to run
* Clone this repo
* Install [poetry](https://python-poetry.org/docs/) (a tool for dependency management and packaging in Python) 
    
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    
* Install the dependecies && activate the virtual environment

        poetry install && poetry shell

# Data
The **[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)** dataset was used to train and test the model. It contains reviews written by customers. This dataset includes 23486 rows and 10 feature variables. However, only the column **Review Text** was used to predict the **Rating** column.

The target variable is a positive ordinal integer variable for the product score granted by the customer from **1** Worst, to **5** Best. It has been mapped to **3** categories:

* {1, 2} -> "negative"
* {3} -> "neutral"
* {4, 5} -> "positive"

# Modeling
A BERT based model is fine-tuned to predict the sentiment. We used the **[RoBERTa](https://arxiv.org/abs/1907.11692)** model (A Robustly Optimized BERT Pretraining)

This model is implemented is [Transformers Hugging Face library](https://huggingface.co/docs/transformers/master/en/model_doc/roberta#overview). 

# Streamlit App
You can run the app locally:

    streamlit run streamlit_app/model_dash.py

![Streamlit App screenshot](/images/streamlit.png)



# Deployment using FastAPI and Docker
We deploy the model as an HTTP endpoint using FastAPI, and then dockerize the code in a docker image.

We use this docker image as the base image (from the developper of the FastAPI package)

* To build the image:

        docker image build -t review_app .

* To run the container locally:

        docker container run -d -p 8080:80 --name myapp review_app

* To run the app using docker-compose:

        docker-compose up 

* To run the app using docker swarm:

        docker stack deploy -c docker-compose.yml MyApp


