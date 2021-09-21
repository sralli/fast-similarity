from annoy import AnnoyIndex

class Annoy:
    def initialize_index(self, embedding_size, metric='angular'):
        '''
            This function initializes the search index for the ANNOY method

            Arguments: 
               - Embedding size: The size of the embeddigns, the vector
               - metric: choices: "angular", "euclidean", "manhattan", "hamming", or "dot".
               
               
            Returns:
                - faiss index: The initialized index for the HNSWLIB ANN method. 
        
        '''

        self.index = AnnoyIndex(embedding_size, metric)
      
        
        return self.index


    def add_items(self, data):
        '''
            This functions adds data to the index that was created
            Arguments: 
                - data (array/list): The data that needs to be added to the index, generally vectors
        '''

        for index, vector in enumerate(data):
            self.index.add_item(index, vector)

    
    def build_index(self, num_trees, annoy_index_path):
        '''
            This function is unique to the annoy method. It builds the annoy index which is used for querying

            - Arguments:
                - num_trees(int): Builds a forest of n tress. More trees give higher precision when querying. 
                - annoy_index_path(str): path where the index will be saved
            
                
                

        '''

        self.index.build(num_trees)
        self.index.save(annoy_index_path)

        return self.index

    def get_top_k_items(self, query_vector, top_k=10):
        '''
            This function is used to get the top 10 similar items to the query (in vector form) provided

            Arguments: 
                - query_vector (list/array): vector of the query that would be used to get the most similar items from the index
                - top_k(int): The number of top values to be returned by the index

            Returns:
                - Top k vectors in a list
        '''

        labels, distances = self.index.get_nns_by_vector(query_vector, top_k, include_distances=True)
        
        items = [{'label': id, 'similarity': 1-similarity} for id, similarity in zip(labels[0], distances[0])]
        items = sorted(items, key=lambda x: x['similarity'], reverse=True)

        return items