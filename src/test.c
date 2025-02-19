#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "victor.h"

#define DIMS 128  // Vector dimensions
#define NUM_VECTORS 200000  // Number of vectors to insert

float32_t vectors[NUM_VECTORS][DIMS];
int ids[NUM_VECTORS];

int main() {
    // Initialize random seed
    srand(time(NULL));

    // Create a vector table
    struct table *vec_table = victor_table(DIMS, COSINE);
    if (vec_table == NULL) {
        printf("Error: Failed to create vector table.\n");
        return EXIT_FAILURE;
    }

    printf("Vector table created with %d dimensions.\n", DIMS);

    // Allocate and insert vectors


    for (int i = 0; i < NUM_VECTORS; i++) {
        for (int j = 0; j < DIMS; j++) {
            vectors[i][j] = (float32_t)(rand() % 100) / 100.0f; // Random values 0.0 - 1.0
        }
        ids[i] = insert_vector(vec_table, vectors[i]);
        if (ids[i] == -1) {
            printf("Error: Failed to insert vector %d\n", i);
            return EXIT_FAILURE;
        }
        printf("Inserted vector %d with ID %d\n", i, ids[i]);
    }

    // Query vector
    float32_t query_vector[DIMS];
    for (int j = 0; j < DIMS; j++) {
        query_vector[j] = (float32_t)(rand() % 100) / 100.0f;
    }

    // Find the most similar vector without threshold
    match_result_t *result;
    search_better_n_match(vec_table, vectors[(NUM_VECTORS-1)/2], &result,5);
    printf("\n🔍 Closest vector found:\n");
    printf("  - ID: %d\n", result[1].id);
    printf("  - Distance: %f\n", result[1].distance);

    free_table(&vec_table);
    return EXIT_SUCCESS;
}
