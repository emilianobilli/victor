from database import init_db, get_session, Data, Vector
import requests
import numpy as np
import json


class VictorClient:
    """
    Python client for interacting with the Victor Vector Cache Database API.
    """

    def __init__(self, host="localhost", port=8080):
        self.base_url = f"http://{host}:{port}"
    
    def _send_request(self, method, endpoint, data=None, params=None):
        """
        Helper function to send an HTTP request to the API.
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.request(method, url, json=data, params=params, headers=headers)
            response.raise_for_status()
            if response.text.strip():  # Check if response is not empty
                return response.json()
            return {"message": "Empty response from server"}  # Return safe message if empty

        except Exception as e:
            return {"error", str(e)}

    def create_index(self, index_type=0, method=0, dims=128):
        """
        Creates a new index. If an index already exists, it will be replaced.
        """
        data = {
            "index_type": index_type,
            "method": method,
            "dims": dims
        }
        return self._send_request("POST", "/", data)

    def insert_vector(self, vector_id, vector):
        """
        Inserts a vector into the index.
        """
        data = {
            "id": vector_id,
            "vector": vector
        }
        return self._send_request("POST", "/index/vector", data)

    def search_vector(self, vector, dims):
        """
        Searches for the closest match to the given vector.
        """
        data = {
            "vector": vector,
            "dims": dims
        }
        return self._send_request("POST", "/search", data)

    def search_n_vectors(self, vector, dims, top_n):
        """
        Searches for the top N closest matches to the given vector.
        """
        data = {
            "vector": vector,
            "dims": dims,
            "top_n": top_n
        }
        return self._send_request("POST", "/search_n", data)

    def delete_vector(self, vector_id):
        """
        Deletes a vector from the index by its ID.
        """
        params = {"id": vector_id}
        return self._send_request("DELETE", "/index/vector", params=params)

    def destroy_index(self):
        """
        Destroys the current index.
        """
        return self._send_request("DELETE", "/index")


init_db()

class Rag(object):
    def __init__(self, index_type=0, method=0, dims=2):
        self.dims = dims
        try:
            session = get_session()
            self.vector = VictorClient(host="localhost", port=9000)
            self.vector.create_index(index_type, method, dims)

            # Cargar vectores previos desde la base de datos
            vectors = session.query(Vector).all()
            totalvs = len(vectors)
            print(f"Initializing RAG with {totalvs} vectors...")

            for vector in vectors:
                try:
                    result = self.vector.insert_vector(vector.id, vector.get_vector().tolist())
                    print(result)
                    if "error" in result:
                        raise ValueError(str(result["error"])) 
                except Exception as e:
                    print(f"Error inserting vector {vector.id}: {e}")
                    return None

        except Exception as e:
            print(f"Error initializing RAG: {e}")
            return None
        finally:
            session.close()  # Asegurar cierre de sesión

    @staticmethod
    def __insert_vector_in_database(vector):
        """Inserta un vector en la base de datos y retorna el ID."""
        vector_blob = np.array(vector, dtype=np.float32).tobytes()
        
        try:
            session = get_session()
            vec_entry = Vector(embedding=vector_blob)
            session.add(vec_entry)
            session.commit()  
            return vec_entry.id  # Retorna el ID después del commit
        
        except Exception as e:
            session.rollback()  # Deshacer cambios en caso de error
            print(f"Error inserting vector in database: {e}")
            return None
        
        finally:
            session.close()

    @staticmethod
    def __insert_data_in_database(data, vector_id):
        try:
            session = get_session()
            data_entry = Data(rawdata=json.dumps(data), vector_id=vector_id)
            session.add(data_entry)
            session.commit()  
            return data_entry.id  # Retorna el ID después del commit
        
        except Exception as e:
            session.rollback()  # Deshacer cambios en caso de error
            print(f"Error inserting data in database: {e}")
            return None
        
        finally:
            session.close()


    @staticmethod
    def __get_rawdata_by_vector_id(vector_id):
        session = get_session()
        result = session.query(Data.rawdata).filter(Data.vector_id == vector_id).first()
        return result[0] if result else None

    def insert_embedding(self, embedding, data):
        """Inserta un vector en la base de datos y en el índice vectorial."""
        if not isinstance(embedding, list) or len(embedding) != self.dims:
            raise ValueError(f"Invalid dimensions: expected {self.dims}, got {len(embedding)}")

        vector_id = self.__insert_vector_in_database(embedding)
        if vector_id is None:
            raise ValueError("Failed to insert vector in database")

        data_id = self.__insert_data_in_database(data, vector_id)
        
        response = self.vector.insert_vector(vector_id, embedding)
        if "error" in response:
            raise ValueError(response["error"])
        return True

    def search(self, embedding):
        if len(embedding) != self.dims:
            raise ValueError(f"Invalid dimensions: expected {self.dims}, got {len(embedding)}")
        
        response = self.vector.search_vector(embedding, self.dims)
        
        if "error" in response:
            raise ValueError(f"Vector search failed: {response['error']}")
        
        result = response.get("result")
        if not result:
            raise ValueError("Invalid response from vector search: missing 'result' field")

        vector_id = result.get("id")
        if not vector_id:
            raise ValueError("Vector search did not return a valid ID")

        data = self.__get_rawdata_by_vector_id(vector_id)
        if data is None:
            raise ValueError(f"Vector ID {vector_id} found but no associated data in the database")

        result["data"] = json.loads(data)
        return result
