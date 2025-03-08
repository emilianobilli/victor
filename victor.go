package victor

/*
#cgo LDFLAGS: -L./lib -lvictor
#include "lib/index.h"
#include "lib/types.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ErrorCode maps C error codes to Go
type ErrorCode int

const (
	SUCCESS ErrorCode = iota
	INVALID_INIT
	INVALID_INDEX
	INVALID_VECTOR
	INVALID_RESULT
	INVALID_DIMENSIONS
	INVALID_ID
	INDEX_EMPTY
	SYSTEM_ERROR
)

// errorMessages maps error codes to human-readable messages
var errorMessages = map[ErrorCode]string{
	SUCCESS:            "Success",
	INVALID_INIT:       "Invalid initialization",
	INVALID_INDEX:      "Invalid index",
	INVALID_VECTOR:     "Invalid vector",
	INVALID_RESULT:     "Invalid result",
	INVALID_DIMENSIONS: "Invalid dimensions",
	INVALID_ID:         "Invalid ID",
	INDEX_EMPTY:        "Index is empty",
	SYSTEM_ERROR:       "System error",
}

// toError converts a C error code to a Go error
func toError(code C.int) error {
	if code == C.int(SUCCESS) {
		return nil
	}
	if msg, exists := errorMessages[ErrorCode(code)]; exists {
		return fmt.Errorf(msg)
	}
	return fmt.Errorf("Unknown error code: %d", code)
}

// MatchResult represents a search result in Go
type MatchResult struct {
	ID       int     `json:"id"`
	Distance float32 `json:"distance"`
}

// Index represents an index structure in Go
type Index struct {
	ptr *C.Index
}

// AllocIndex creates a new index
func AllocIndex(indexType, method int, dims uint16) (*Index, error) {
	idx := C.alloc_index(C.int(indexType), C.int(method), C.uint16_t(dims))
	if idx == nil {
		return nil, fmt.Errorf("Failed to allocate index")
	}
	return &Index{ptr: idx}, nil
}

// Insert adds a vector to the index with a given ID
func (idx *Index) Insert(id uint64, vector []float32) error {
	if idx.ptr == nil {
		return fmt.Errorf("Index not initialized")
	}
	if len(vector) == 0 {
		return fmt.Errorf("Empty vector")
	}

	cVector := (*C.float)(unsafe.Pointer(&vector[0]))
	return toError(C.insert(idx.ptr, C.uint64_t(id), cVector, C.uint16_t(len(vector))))
}

// Search finds the closest match for a given vector
func (idx *Index) Search(vector []float32, dims int) (*MatchResult, error) {
	if idx.ptr == nil {
		return nil, fmt.Errorf("Index not initialized")
	}

	var cResult C.MatchResult
	cVector := (*C.float)(unsafe.Pointer(&vector[0]))
	err := C.search(idx.ptr, cVector, C.uint16_t(dims), &cResult)
	if e := toError(err); e != nil {
		return nil, e
	}

	return &MatchResult{
		ID:       int(cResult.id),
		Distance: float32(cResult.distance),
	}, nil
}

func (idx *Index) SearchN(vector []float32, dims, n int) ([]MatchResult, error) {
	if idx == nil || idx.ptr == nil {
		return nil, fmt.Errorf("index is nil")
	}
	if n <= 0 {
		return nil, fmt.Errorf("invalid number of results: %d", n)
	}

	// Convertir el vector Go a un puntero C
	cVector := (*C.float)(unsafe.Pointer(&vector[0]))

	// Crear un buffer en C para almacenar los resultados
	var cResults *C.MatchResult

	// Llamar a la función C
	err := C.search_n(idx.ptr, cVector, C.uint16_t(dims), &cResults, C.int(n))
	if e := toError(err); e != nil {
		return nil, e
	}

	// Convertir los resultados de C a un slice de Go
	cResultsSlice := unsafe.Slice(cResults, n)
	results := make([]MatchResult, n)
	for i := 0; i < n; i++ {
		results[i] = MatchResult{
			ID:       int(cResultsSlice[i].id),
			Distance: float32(cResultsSlice[i].distance),
		}
	}

	C.free(unsafe.Pointer(cResults))
	fmt.Println(results)
	return results, nil
}

// Delete removes a vector from the index by its ID
func (idx *Index) Delete(id uint64) error {
	if idx.ptr == nil {
		return fmt.Errorf("Index not initialized")
	}
	return toError(C.delete(idx.ptr, C.uint64_t(id)))
}

// DestroyIndex releases index memory
func (idx *Index) DestroyIndex() {
	if idx.ptr != nil {
		C.destroy_index(&idx.ptr)
		idx.ptr = nil
	}
}
