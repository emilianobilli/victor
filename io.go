package victor

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

type VictorDB struct {
	idMap map[int]string

	table   *table
	storage Storage

	Dims int
	Mode string
}

func Open(s Storage, dims int, smode string) (*VictorDB, error) {
	var mode int
	switch smode {
	case "L2NORM":
		mode = L2NORM
	case "COSINE":
		mode = COSINE
	default:
		return nil, fmt.Errorf("Invalid mode %s", smode)
	}

	db := &VictorDB{
		storage: s,
		idMap:   make(map[int]string),
		Mode:    smode,
		Dims:    dims,
	}

	records, err := s.list()
	if err != nil {
		return nil, err
	}

	db.table, err = newTable(db.Dims, mode)
	if err != nil {
		return nil, err
	}

	for _, id := range records {
		rec, err := s.load(id)
		if err != nil {
			fmt.Printf("Warning: skipping record %s: %v\n", id, err)
			continue // Skip this file if there's an error
		}

		for _, embadding := range rec.Embeddings {
			if len(embadding) != db.Dims {
				fmt.Printf("Warning: Invalid dims %d expected from %s has: %d\n", db.Dims, id, len(embadding))
				continue
			}
			if id, err := db.table.insertVector(embadding); err != nil {
				fmt.Printf("Error: Could not insert vector from: %s\n", id)
			} else {
				db.idMap[id] = *rec.ID
			}
		}
	}
	return db, nil
}

func (d *VictorDB) Insert(r *Record) error {

	if len(r.Embeddings) == 0 {
		return fmt.Errorf("invalid embaddings len, can't not be 0")
	}

	id := hashVector(r.Embeddings[0])
	r.ID = &id
	if d.storage.check(id) {
		return fmt.Errorf("duplicated entry")
	}

	if err := d.storage.save(r); err != nil {
		return err
	}

	for i, embadding := range r.Embeddings {
		if len(embadding) != d.Dims {
			d.storage.delete(*r.ID)
			return fmt.Errorf("invalid dims %d expected from %s has: [%d]%d\n", d.Dims, id, i, len(embadding))
		}
		if id, err := d.table.insertVector(embadding); err != nil {
			d.storage.delete(*r.ID)
			return fmt.Errorf("could not insert vector from: %s\n", id)
		} else {
			d.idMap[id] = *r.ID
		}
	}
	return nil
}

func (d *VictorDB) Delete(id string) error {
	for cid, did := range d.idMap {
		if did == id {
			d.table.deleteVector(cid)
			delete(d.idMap, cid)
			d.storage.delete(did)
			return nil
		}
	}
	return fmt.Errorf("not found")
}

func (d *VictorDB) Search(vector []float32) (*Record, float32, error) {
	match, err := d.table.searchBestMatch(vector)
	if err != nil {
		return nil, 0.0, err
	}
	id, ok := d.idMap[match.id]
	if ok {
		record, err := d.storage.load(id)
		if err != nil {
			return nil, 0.0, err
		}
		return record, match.distance, nil
	}
	return nil, 0.0, fmt.Errorf("invalid value")
}

func (d *VictorDB) SearchBestN(vector []float32, n int) ([]map[string]interface{}, error) {
	matches, err := d.table.searchBestNMatch(vector, n)
	if err != nil {
		return nil, err
	}

	results := make([]map[string]interface{}, 0, len(matches))
	for _, match := range matches {
		id, ok := d.idMap[match.id]
		if !ok {
			continue // Si el ID no está en el mapa, ignoramos este resultado
		}

		record, err := d.storage.load(id)
		if err != nil {
			continue // Si hay un error cargando, lo ignoramos y pasamos al siguiente
		}

		results = append(results, map[string]interface{}{
			"record":   record,
			"distance": match.distance,
		})
	}

	// Si no hay resultados válidos, devolvemos un error
	if len(results) == 0 {
		return nil, fmt.Errorf("no valid matches found")
	}

	return results, nil
}

// Insertar un nuevo registro
func (db *VictorDB) InsertHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	var record Record
	if err := json.NewDecoder(r.Body).Decode(&record); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if err := db.Insert(&record); err != nil {
		http.Error(w, fmt.Sprintf("Insert failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"id": *record.ID, "message": "Record inserted successfully"})
}

// Eliminar un registro
func (db *VictorDB) DeleteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "Missing 'id' parameter", http.StatusBadRequest)
		return
	}

	if err := db.Delete(id); err != nil {
		http.Error(w, "Record not found", http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "Record deleted successfully"})
}

func (db *VictorDB) SearchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Obtener el parámetro `n` desde la URL
	nParam := r.URL.Query().Get("n")
	var n int
	var err error

	if nParam != "" {
		n, err = strconv.Atoi(nParam)
		if err != nil || n <= 0 {
			http.Error(w, "Invalid 'n' parameter", http.StatusBadRequest)
			return
		}
	}

	// Decodificar el JSON con el vector
	var request struct {
		Vector []float32 `json:"vector"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Si `n` no fue especificado, usar el `Search` normal
	if n == 0 {
		record, distance, err := db.Search(request.Vector)
		if err != nil {
			http.Error(w, "No match found", http.StatusNotFound)
			return
		}

		response := map[string]interface{}{
			"record":   record,
			"distance": distance,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// Si `n` fue especificado, buscar los `n` mejores matches
	results, err := db.SearchBestN(request.Vector, n)
	if err != nil {
		http.Error(w, "No matches found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}
