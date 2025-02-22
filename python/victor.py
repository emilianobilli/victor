import requests
import json

class VictorDBClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

    def insert(self, record_id, embeddings, data):
        """Inserta un nuevo registro en la base de datos"""
        url = f"{self.base_url}/insert"
        payload = {"id": record_id, "embeddings": embeddings, "data": data}

        response = requests.post(url, json=payload)
        return self._handle_response(response)

    def search(self, vector, n=None):
        """Busca el mejor match o los n mejores resultados para un vector dado"""
        url = f"{self.base_url}/search"
        if n is not None:
            url += f"?n={n}"
        
        payload = {"vector": vector}
        response = requests.post(url, json=payload)
        return self._handle_response(response)

    def delete(self, record_id):
        """Elimina un registro de la base de datos por ID"""
        url = f"{self.base_url}/delete?id={record_id}"
        response = requests.delete(url)
        return self._handle_response(response)

    def _handle_response(self, response):
        """Maneja la respuesta del servidor"""
        if response.status_code in (200, 201):
            return response.json()
        elif response.status_code == 404:
            return {"error": "Not found"}
        elif response.status_code == 400:
            return {"error": "Bad request"}
        elif response.status_code == 500:
            return {"error": "Server error"}
        else:
            return {"error": f"Unexpected status code {response.status_code}"}


# Ejemplo de uso
if __name__ == "__main__":
    client = VictorDBClient()

    # Insertar un registro
    insert_response = client.insert("123", [[0.1, 0.2, 0.3, 0.4]], {"data": "ejemplo"})
    print("Insert Response:", insert_response)

    insert_response = client.insert("123", [[0.2, 0.2, 0.3, 0.4]], {"data": "ejemplo"})
    print("Insert Response:", insert_response)

    insert_response = client.insert("123", [[0.3, 0.2, 0.3, 0.4]], {"data": "ejemplo"})
    print("Insert Response:", insert_response)

    insert_response = client.insert("123", [[0.4, 0.2, 0.3, 0.4]], {"data": "ejemplo"})
    print("Insert Response:", insert_response)

    insert_response = client.insert("123", [[0.5, 0.2, 0.3, 0.4]], {"data": "ejemplo"})
    print("Insert Response:", insert_response)

    # Buscar el mejor match (default)
    search_response = client.search([0.1, 0.2, 0.3, 0.4])
    print("Search Response:", json.dumps(search_response, indent=2))

    # Buscar los 5 mejores matches
    search_n_response = client.search([0.1, 0.2, 0.3, 0.4], n=5)
    print("Search N Response:", json.dumps(search_n_response, indent=2))

