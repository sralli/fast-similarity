import faiss

class FAISS:
    def initialize_index(self, dim: 128):
        '''
            This function initializes the search index for the FAISS method

            Arguments: 
                - dim: Dimensionality of the space
               
            Returns:
                - faiss index: The initialized index for the HNSWLIB ANN method. 
        
        '''

        self.index = faiss.index.IndexFlatL2(dim)
        
        return self.index


    def add_items(self, data):
        '''
            This functions adds data to the index that was created
            Arguments: 
                - data (array/list): The data that needs to be added to the index, generally vectors
        '''

        self.index.add(data)

    