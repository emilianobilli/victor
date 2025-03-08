package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"

	"victor"
)

// Global index instance and mutex for thread safety
var (
	indexInstance *victor.Index
	mutex         sync.Mutex
)

// Response structure
type Response struct {
	Message string      `json:"message"`
	Result  interface{} `json:"result,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// Index creation request structure
type CreateIndexRequest struct {
	IndexType int  `json:"index_type"`
	Method    int  `json:"method"`
	Dims      uint `json:"dims"`
}

// Vector insertion request structure
type InsertRequest struct {
	ID     uint64    `json:"id"`
	Vector []float32 `json:"vector"`
}

// Search request structure
type SearchRequest struct {
	Vector []float32 `json:"vector"`
	Dims   int       `json:"dims"`
	TopN   int       `json:"top_n,omitempty"`
}

// Logger middleware
func logRequest(r *http.Request) {
	log.Printf("%s %s", r.Method, r.URL.Path)
}

// Create an index (destroy existing one if necessary)
func createIndexHandler(w http.ResponseWriter, r *http.Request) {
	logRequest(r)
	mutex.Lock()
	defer mutex.Unlock()

	var req CreateIndexRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON input", http.StatusBadRequest)
		log.Println("Index creation failed: Invalid JSON input")
		return
	}

	// If an index already exists, destroy it before creating a new one
	if indexInstance != nil {
		indexInstance.DestroyIndex()
		indexInstance = nil
		log.Println("Previous index destroyed")
	}

	idx, err := victor.AllocIndex(req.IndexType, req.Method, uint16(req.Dims))
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create index: %v", err), http.StatusInternalServerError)
		log.Println("Error creating index:", err)
		return
	}

	indexInstance = idx
	log.Printf("Index created: Type=%d, Method=%d, Dims=%d\n", req.IndexType, req.Method, req.Dims)
	json.NewEncoder(w).Encode(Response{Message: "Index created successfully"})
}

// Search for the closest match
func searchVectorHandler(w http.ResponseWriter, r *http.Request) {
	logRequest(r)
	mutex.Lock()
	defer mutex.Unlock()

	if indexInstance == nil {
		http.Error(w, "Index not initialized", http.StatusNotFound)
		log.Println("Search failed: Index not initialized")
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON input", http.StatusBadRequest)
		log.Println("Search failed: Invalid JSON input")
		return
	}

	result, err := indexInstance.Search(req.Vector, req.Dims)
	if err != nil {
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		log.Println("Search failed:", err)
		return
	}

	log.Printf("Search successful: ID=%d, Distance=%.4f\n", result.ID, result.Distance)
	json.NewEncoder(w).Encode(Response{Message: "Search successful", Result: result})
}

// Search for the top N closest matches
func searchNVectorHandler(w http.ResponseWriter, r *http.Request) {
	logRequest(r)
	mutex.Lock()
	defer mutex.Unlock()

	if indexInstance == nil {
		http.Error(w, "Index not initialized", http.StatusNotFound)
		log.Println("SearchN failed: Index not initialized")
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON input", http.StatusBadRequest)
		log.Println("SearchN failed: Invalid JSON input")
		return
	}

	results, err := indexInstance.SearchN(req.Vector, req.Dims, req.TopN)
	if err != nil {
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		log.Println("SearchN failed:", err)
		return
	}

	// Convert +Inf distances to the max finite float
	for i := range results {
		if math.IsInf(float64(results[i].Distance), 1) {
			results = results[0:i]
			break
		}
	}

	if len(results) == 0 {
		log.Println("SearchN successful: No matches found")
		json.NewEncoder(w).Encode(Response{Message: "Search successful", Result: []victor.MatchResult{}})
		return
	}

	log.Printf("SearchN successful: Found %d results\n", len(results))
	json.NewEncoder(w).Encode(Response{Message: "Search successful", Result: results})
}

// Handles vector insertion (POST) and deletion (DELETE)
func vectorHandler(w http.ResponseWriter, r *http.Request) {
	logRequest(r)
	mutex.Lock()
	defer mutex.Unlock()

	if indexInstance == nil {
		http.Error(w, "Index not initialized", http.StatusNotFound)
		log.Println("Request failed: Index not initialized")
		return
	}

	switch r.Method {
	case "POST":
		// Insert vector
		var req InsertRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON input", http.StatusBadRequest)
			log.Println("Insert failed: Invalid JSON input")
			return
		}

		err := indexInstance.Insert(req.ID, req.Vector)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to insert vector: %v", err), http.StatusInternalServerError)
			log.Println("Insert failed:", err)
			return
		}

		log.Printf("Vector inserted: ID=%d\n", req.ID)
		json.NewEncoder(w).Encode(Response{Message: "Vector inserted successfully"})

	case "DELETE":
		// Delete vector
		idStr := r.URL.Query().Get("id")
		if idStr == "" {
			http.Error(w, "Missing vector ID", http.StatusBadRequest)
			log.Println("Delete failed: Missing vector ID")
			return
		}

		id, err := strconv.ParseUint(idStr, 10, 64)
		if err != nil {
			http.Error(w, "Invalid ID format", http.StatusBadRequest)
			log.Println("Delete failed: Invalid ID format")
			return
		}

		err = indexInstance.Delete(id)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to delete vector: %v", err), http.StatusInternalServerError)
			log.Println("Delete failed:", err)
			return
		}

		log.Printf("Vector deleted: ID=%d\n", id)
		json.NewEncoder(w).Encode(Response{Message: "Vector deleted successfully"})

	default:
		// Unsupported method
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		log.Println("Invalid HTTP method:", r.Method)
	}
}

// Destroy the index
func destroyIndexHandler(w http.ResponseWriter, r *http.Request) {
	logRequest(r)
	mutex.Lock()
	defer mutex.Unlock()

	if indexInstance == nil {
		http.Error(w, "Index not initialized", http.StatusNotFound)
		log.Println("Destroy failed: Index not initialized")
		return
	}

	indexInstance.DestroyIndex()
	indexInstance = nil
	log.Println("Index destroyed successfully")
	json.NewEncoder(w).Encode(Response{Message: "Index destroyed successfully"})
}

// Start the HTTP server
func main() {
	fmt.Println("Victor Cache Database v0.1")
	fmt.Println("==========================")

	// Command-line flags
	addr := flag.String("addr", "localhost", "Listening address")
	port := flag.String("port", "8080", "Listening port")
	flag.Parse()

	serverAddr := fmt.Sprintf("%s:%s", *addr, *port)
	log.Printf("Starting Victor API server on %s\n", serverAddr)

	// Define routes
	http.HandleFunc("/", createIndexHandler)
	http.HandleFunc("/index/vector", vectorHandler)
	http.HandleFunc("/search", searchVectorHandler)
	http.HandleFunc("/search_n", searchNVectorHandler)
	http.HandleFunc("/index", destroyIndexHandler)

	// Graceful shutdown
	go func() {
		if err := http.ListenAndServe(serverAddr, nil); err != nil {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Handle SIGINT (Ctrl+C) and SIGTERM
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig

	log.Println("Shutting down server...")
	if indexInstance != nil {
		indexInstance.DestroyIndex()
	}
	log.Println("Server stopped.")
}
