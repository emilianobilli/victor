package victor

/*
#cgo LDFLAGS: -L./src -lvictor
#include "src/victor.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Table represents a vector table in C
type table struct {
	ptr *C.table_t
}

type matchResult struct {
	id       int
	distance float32
}

const (
	L2NORM = 1
	COSINE = 2
)

// NewTable creates a new vector table
func newTable(dims int, cmpmode int) (*table, error) {
	t := C.victor_table(C.int(dims), C.int(cmpmode))
	if t == nil {
		return nil, fmt.Errorf("failed to create table")
	}
	return &table{ptr: t}, nil
}

// InsertVector inserts a vector into the table
func (t *table) insertVector(vector []float32) (int, error) {
	if t.ptr == nil {
		return -1, fmt.Errorf("table is nil")
	}
	cVector := (*C.float)(unsafe.Pointer(&vector[0]))
	res := C.insert_vector(t.ptr, cVector)
	if res == -1 {
		return -1, fmt.Errorf("failed to insert vector")
	}
	return int(res), nil
}

// DeleteVector removes a vector from the table
func (t *table) deleteVector(id int) error {
	if t.ptr == nil {
		return fmt.Errorf("table is nil")
	}
	res := C.delete_vector(t.ptr, C.int(id))
	if res != 0 {
		return fmt.Errorf("failed to delete vector with ID %d", id)
	}
	return nil
}

// SearchBestMatch searches for the closest vector match
func (t *table) searchBestMatch(vector []float32) (matchResult, error) {
	if t.ptr == nil {
		return matchResult{}, fmt.Errorf("table is nil")
	}
	cVector := (*C.float)(unsafe.Pointer(&vector[0]))
	result := C.search_better_match(t.ptr, cVector)
	return matchResult{id: int(result.id), distance: float32(result.distance)}, nil
}

func (t *table) searchBestNMatch(vector []float32, n int) ([]matchResult, error) {
	if t.ptr == nil {
		return nil, fmt.Errorf("table is nil")
	}
	if n <= 0 {
		return nil, fmt.Errorf("invalid number of results: %d", n)
	}

	// Convertir el vector Go a un puntero C
	cVector := (*C.float)(unsafe.Pointer(&vector[0]))

	// Crear un buffer en C para almacenar los resultados
	var cResults *C.match_result_t

	// Llamar a la función C
	ret := C.search_better_n_match(t.ptr, cVector, &cResults, C.int(n))
	if ret != 0 {
		return nil, fmt.Errorf("search_better_n_match returned error code: %d", ret)
	}

	// Convertir los resultados de C a un slice de Go
	cResultsSlice := unsafe.Slice(cResults, n)
	results := make([]matchResult, n)
	for i := 0; i < n; i++ {
		results[i] = matchResult{
			id:       int(cResultsSlice[i].id),
			distance: float32(cResultsSlice[i].distance),
		}
	}

	C.free(unsafe.Pointer(cResults))

	return results, nil
}

// FreeTable deallocates the table
func (t *table) freeTable() {
	if t.ptr != nil {
		C.free_table(&t.ptr)
		t.ptr = nil
	}
}
