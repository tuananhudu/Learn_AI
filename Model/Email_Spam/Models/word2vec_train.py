from gensim.models import Word2Vec

class Word2VecTrainer:
    def __init__(self , vector_size = 100 , window = 5 , min_count = 1 , workers = 4):
        self.vector_size = vector_size 
        self.window = window 
        self.min_count = min_count 
        self.workers = workers
        self.model = None 
    
    def train(self, sentences):
        print("Huấn luyện mô hình Word2Vec...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
            )
        print("Hoàn tất huấn luyện mô hình")
        return self.model  # Trả về mô hình đã huấn luyện
    
    def get_model(self):
        return self.model 
    
    def save_model(self , filepath):

        if self.model is not None :
            self.model.save(filepath)
            print(f'Mo hinh da duoc luu :{filepath}')
        else :
            print('chua co mo hinh')
    
    def load_model(self , filepath):
        self.model = Word2Vec.load(filepath)
        print(f'Mo hinh da duoc tai tu : {filepath}')
        return self.model 
