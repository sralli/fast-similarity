import hnswlib

class HNSWLIB:
    def initialize_index(self, dim: 128, max_elements: 2000, space='cosine', ef_construction=50, M=16):
        '''
            This function initializes the search index for the HNSWLIB search index

            Arguments: 
                - dim: Dimensionality of the space
                - space: name of the space can be one of l2, ip or cosine
                - M: parameter that defines the maximum number of outgoing connections in the graph.  M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
                - ef_construction: parameter that controls the speed/accuracy trade-off during the index construction
                - max_elements: current capacity of the index

            Returns:
                - hsnwlib_index: The initialized index for the HNSWLIB ANN method. 
        
        '''

        self.index = hnswlib.Index(space=space, dim=dim)

        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

        return self.index


    def change_ef(self, ef_value=50):
        '''
            This function changes the ef value. The ef value can be used to change the recall. 
            ef should always be: ef>k where: 
            k: number of closest elements

            higher ef leads to better accuracy, but slower search
            Arguments: 
                - ef_value(int): The ef_value to be set for the index, default=50
        '''

        self.index.set_ef(ef_value)

    
    def change_num_threads(self, num_threads):
        '''
            This function changes the num_threads for the index. By default, the index uses all the cores

            Arguments:
                - num_threads(int): The number of threads the index should use for batch/search and construction
        
        
        '''

        self.index.set_num_threads(num_threads)

    
    def add_items(self, data):
        '''
            This functions adds data to the index that was created
            Arguments: 
                - data (array/list): The data that needs to be added to the index, generally vectors
        '''

        self.index.add_items(data)


    def get_top_k_items(self, query_vector, top_k=10):
        '''
            This function is used to get the top 10 similar items to the query (in vector form) provided

            Arguments: 
                - query_vector (list/array): vector of the query that would be used to get the most similar items from the index
                - top_k(int): The number of top values to be returned by the index

            Returns:
                - Top k vectors in a list
        '''

        labels, distances = self.index.knn_query(query_vector, k=top_k)
        
        items = [{'label': id, 'similarity': 1-similarity} for id, similarity in zip(labels[0], distances[0])]
        items = sorted(items, key=lambda x: x['similarity'], reverse=True)

        return items

