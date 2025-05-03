from nltk.corpus import stopwords
import re
from gensim.utils import simple_preprocess
import numpy as np
from tqdm import tqdm
import pandas as pd

class TextProcessor:
    def __init__(self, messages, lemmatizer, word2vec_trainer=None, stopwords=None):
        """
        Khởi tạo TextProcessor.

        Parameters:
        - messages: DataFrame chứa cột 'message' (nội dung văn bản) và 'label' (nhãn).
        - lemmatizer: Đối tượng lemmatizer (ví dụ: WordNetLemmatizer).
        - word2vec_trainer: Đối tượng Word2VecTrainer hoặc None (mặc định: None, sẽ tạo mới).
        - stopwords: Danh sách stopwords (mặc định: stopwords tiếng Anh từ NLTK).
        """
        self.messages = messages
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords if stopwords is not None else stopwords.words('english')
        self.word2vec_trainer = word2vec_trainer if word2vec_trainer is not None else Word2VecTrainer()
        self.model = self.word2vec_trainer.get_model()
        self.corpus = []
        self.valid_indices = []
        self.words = []
        self.word_indices = []
        self.final_indices = []
        self.X = None
        self.y = None

    def preprocess_corpus(self):
        """Tạo corpus từ messages, loại bỏ ký tự không cần thiết, stopwords, và chuẩn hóa."""
        self.corpus = []
        self.valid_indices = []
        
        for i in range(len(self.messages)):
            review = re.sub('[^a-zA-Z]', ' ', self.messages['message'][i])
            review = review.lower()
            review = review.split()
            review = [self.lemmatizer.lemmatize(word) for word in review if word not in self.stopwords]
            review = ' '.join(review)
            if review.strip():
                self.corpus.append(review)
                self.valid_indices.append(i)
        
        print(f"Số tài liệu trong corpus: {len(self.corpus)}")
        print(f"Số chỉ số hợp lệ: {len(self.valid_indices)}")

    def tokenize_words(self, min_len=1):
        """
        Tạo words từ corpus bằng simple_preprocess.

        Parameters:
        - min_len: Độ dài tối thiểu của từ trong simple_preprocess (mặc định: 1).
        """
        self.words = []
        self.word_indices = []
        
        for i, sent in enumerate(self.corpus):
            processed_sent = simple_preprocess(sent, min_len=min_len)
            if processed_sent:
                self.words.append(processed_sent)
                self.word_indices.append(i)
        
        print(f"Số tài liệu trong words: {len(self.words)}")
        print(f"Số chỉ số trong word_indices: {len(self.word_indices)}")

    def ensure_word2vec_model(self):
        """Đảm bảo có mô hình Word2Vec, huấn luyện nếu chưa có."""
        if self.model is None:
            self.model = self.word2vec_trainer.train(self.words)
        else:
            print("Mô hình Word2Vec đã được cung cấp, bỏ qua huấn luyện.")

    def avg_word2vec(self, doc):
        """Tính vector trung bình cho một tài liệu bằng Word2Vec."""
        if not doc:
            return np.zeros(self.model.vector_size)
        vectors = [self.model.wv[word] for word in doc if word in self.model.wv]
        if not vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def compute_features(self):
        """Tạo X từ words bằng avg_word2vec."""
        self.X = []
        for i in tqdm(range(len(self.words))):
            vector = self.avg_word2vec(self.words[i])
            self.X.append(vector)
        
        self.X = np.array(self.X)
        print("X shape:", self.X.shape)

    def compute_labels(self):
        """Tạo y từ nhãn của messages, đồng bộ với X."""
        self.final_indices = [self.valid_indices[i] for i in self.word_indices]
        print(f"Số chỉ số trong final_indices: {len(self.final_indices)}")
        
        filtered_messages = self.messages.iloc[self.final_indices]
        self.y = pd.get_dummies(filtered_messages['label']).iloc[:, 0].values
        print("y shape:", self.y.shape)

    def process(self, min_len=1):
        """
        Thực hiện toàn bộ quy trình xử lý.

        Parameters:
        - min_len: Độ dài tối thiểu của từ trong simple_preprocess.

        Returns:
        - X: Mảng NumPy chứa vector đặc trưng.
        - y: Mảng NumPy chứa nhãn.
        """
        print("Bắt đầu xử lý...")
        self.preprocess_corpus()
        self.tokenize_words(min_len=min_len)
        self.ensure_word2vec_model()
        self.compute_features()
        self.compute_labels()
        
        if self.X.shape[0] == self.y.shape[0]:
            print("X và y đồng bộ, sẵn sàng cho học máy!")
        else:
            print("X và y không đồng bộ, kiểm tra lại!")
        
        return self.X, self.y

    def inspect_lost_documents(self):
        """Kiểm tra các tài liệu bị mất từ corpus khi tạo words."""
        lost_indices = [i for i in range(len(self.corpus)) if i not in self.word_indices]
        print(f"Số tài liệu bị mất: {len(lost_indices)}")
        if lost_indices:
            print("Các tài liệu bị mất:")
            for i in lost_indices:
                print(f"corpus[{i}]: {self.corpus[i]}")
        else:
            print("Không có tài liệu nào bị mất.")