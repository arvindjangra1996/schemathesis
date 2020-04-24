
class Store_response():
    def __init__(self):
        self.store = {}
    
    def store_result(self,path,k,v):
        self.store[path] = {}
        self.store[path][k]=v
    
    def get_store_result(self,path,k):
        return self.store[path][k]