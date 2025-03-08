/*
* index.h - Index Structure and Management for Vector Database
* 
* Copyright (C) 2025 Emiliano A. Billi
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*
* Contact: emiliano.billi@gmail.com
*
* Purpose:
* This header defines the `Index` structure, which serves as an abstraction
* for various types of vector indices (e.g., Flat, HNSW, IVF). It provides
* function pointers for searching, inserting, deleting, and managing indices.
*/

#ifndef __INDEX_H
#define __INDEX_H 1

#include "types.h"

#define FLAT_INDEX 0x00
/**
 * Structure representing an abstract index for vector search.
 * It supports multiple indexing strategies through function pointers.
 */
typedef struct {
    char *name;     // Name of the indexing method (e.g., "Flat", "HNSW")
    void *data;        // Pointer to the specific index data structure
    void *context;     // Additional context for advanced indexing needs

    /**
     * Searches for the `n` closest matches to the given vector.
     * @param data The specific index data structure.
     * @param vector The input vector.
     * @param dims The number of dimensions.
     * @param results Output array to store the closest matches.
     * @param n The number of matches to retrieve.
     * @return The number of matches found, or -1 on error.
     */
    int (*search_n)(void *, float32_t *, uint16_t, MatchResult **, int);

    /**
     * Searches for the best match to the given vector.
     * @param data The specific index data structure.
     * @param vector The input vector.
     * @param dims The number of dimensions.
     * @param result Output structure to store the best match.
     * @return 0 if successful, or -1 on error.
     */
    int (*search)(void *, float32_t *, uint16_t, MatchResult *);

    /**
     * Inserts a new vector into the index.
     * @param data The specific index data structure.
     * @param id The unique identifier for the vector.
     * @param vector The input vector.
     * @param dims The number of dimensions.
     * @return 0 if successful, or -1 on error.
     */
    int (*insert)(void *, uint64_t, float32_t *, uint16_t);

    /**
     * Deletes a vector from the index using its ID.
     * @param data The specific index data structure.
     * @param id The unique identifier of the vector to delete.
     * @return 0 if successful, or -1 on error.
     */
    int (*delete)(void *, uint64_t);

    int (*_release)(void **);

} Index;

/**
 * Wrapper functions to call the corresponding method in `Index`.
 * These functions ensure safe access and provide a unified interface.
 */
extern int search_n(Index *index, float32_t *vector, uint16_t dims, MatchResult **results, int n);
extern int search(Index *index, float32_t *vector, uint16_t dims, MatchResult *result);
extern int insert(Index *index, uint64_t id, float32_t *vector, uint16_t dims);
extern int delete(Index *index, uint64_t id);

extern Index *alloc_index(int type, int method, uint16_t dims);
extern int destroy_index(Index **index);

#endif // __INDEX_H
