# email_service.py
import pandas as pd 
import numpy as np 
from domain.domain import EmailRequest, EmailResponse 
import joblib 
import os 
import warnings
import nltk
from gensim.models import Word2Vec
warnings.filterwarnings('ignore')
from pre_processing.text_processing import TextProcessor
from Models.word2vec_train import Word2VecTrainer

class EmailService: 
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(base_dir, "..", "artifacts", "best_model.pkl")
        self.path_word2vec_model = os.path.join(base_dir, "..", "artifacts", "word2vec_model.model")
        self.word2vec_trainer = Word2VecTrainer()
        self.word2vec_trainer.model = Word2Vec.load(self.path_word2vec_model)
        self.model = joblib.load(self.path_model)
        self.processor = TextProcessor(
            messages=pd.DataFrame({'message': []}),
            lemmatizer=nltk.stem.WordNetLemmatizer(),
            word2vec_trainer=self.word2vec_trainer
        )

    def preprocess_input(self, request: EmailRequest) -> pd.DataFrame:
        return pd.DataFrame({'message': [request.message]})
    
    def predict(self, request: EmailRequest) -> EmailResponse:
        input_df = self.preprocess_input(request)
        self.processor.messages = input_df
        self.processor.corpus = []
        self.processor.valid_indices = []
        self.processor.words = []
        self.processor.word_indices = []
        self.processor.final_indices = []
        self.processor.X = None
        self.processor.y = None
        self.processor.preprocess_corpus()
        self.processor.tokenize_words(min_len=1)
        self.processor.compute_features()
        if self.processor.X.shape[0] == 0:
            return EmailResponse(reply=0)
        prediction = self.model.predict(self.processor.X)[0]
        return EmailResponse(reply=int(prediction))

# if __name__ == '__main__':
#     email_service = EmailService()
#     request = EmailRequest(message="Congratulations! You won a free iPhone. Click here to claim now.")
#     response = email_service.predict(request)
#     print("Reply:", response.reply)
# (spam_email) C:\Learn_AI\Model\Email_Spam\services>python email_service.py 
# Số tài liệu trong corpus: 1
# Số chỉ số hợp lệ: 1
# Số tài liệu trong words: 1
# Số chỉ số trong word_indices: 1
# 100%|███████████████████████████████████████████████████████████████████| 1/1 [00:00<?, ?it/s] 
# X shape: (1, 100)
# Reply: 0


# if __name__ == '__main__':
#     email_service = EmailService()
#     request = EmailRequest(message="""
#         Dear Student,

#         This is a reminder that your Data Science 2025 class schedule will be updated starting next week.
#         Please check the LMS portal for more information.

#         Best regards,
#         Academic Office
#     """)
#     response = email_service.predict(request)
#     print("Reply:", response.reply)
# (spam_email) C:\Learn_AI\Model\Email_Spam\services>python email_service.py 
# Số tài liệu trong corpus: 1
# Số chỉ số hợp lệ: 1
# Số tài liệu trong words: 1
# Số chỉ số trong word_indices: 1
# 100%|███████████████████████████████████████████████████████████████████| 1/1 [00:00<?, ?it/s] 
# X shape: (1, 100)
# Reply: 1