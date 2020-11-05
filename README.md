# Chatbot
This is a Conversational Chatbot trained on Cornell Movie dataset.
Architecture: Seq2Seq model with Bidirectional LSTMs incorporated with Attention mechanism and Beam search decoding

## Environment setup
1. Prerequisite – Python 3.7.6
2. Create a folder (say chatbot) and ensure that the binaries are downloaded and saved in the same folder
3. Create a virtual environment and activate it
   -> python –m venv venv
   -> venv\Scripts\activate
4. Install requirements.txt
   -> pip install –r requirements.txt

## Train Chatbot

To train a model from python console:
1. Configure hparams.json file to required training hyperparameters.
2. Set console working directory to chatbot directory.
3. To train a new model, run the below:
   train.py –d datasets\cornell_movie_dialog
4. Trained model available at: https://drive.google.com/drive/folders/1sCfjFdKWb3IPSyEZGKGvzeeK8IEh3mm8 
5. Once model is trained or model is downloaded(from the above link), ensure that the trained model is 
   available under 'models\cornell_movie_dialog' location.
   
Note: Since the model is already trained, please skip this step

## Chat with Chatbot

1. For console chat, ensure that the trained model is available in ‘models\cornell_movie_dialog’ location

2. From the chatbot working directory, run the below:
	 chat.py models\cornell_movie_dialog\trained_model\best_weights_training.ckpt



	


