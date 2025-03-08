#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "index.h"

#define DIMS 128  // Número de dimensiones del vector de prueba
#define TOP_N 5   // Número de mejores coincidencias a buscar

void print_vector(float *vector, int dims) {
    printf("[ ");
    for (int i = 0; i < dims; i++) {
        printf("%.2f ", vector[i]);
    }
    printf("]\n");
}

int main() {
    // Parámetros del índice
    int index_type = FLAT_INDEX;
    int method = 1; // Método de prueba, ajústalo si es necesario
    uint16_t dims = DIMS;

    // Crear el índice
    Index *index = alloc_index(index_type, method, dims);
    if (!index) {
        printf("Error: No se pudo asignar el índice.\n");
        return 1;
    }
    printf("Índice creado correctamente.\n");

    // Crear un vector de prueba
    float vector[DIMS];
    for (int i = 0; i < DIMS; i++) {
        vector[i] = (float)i * 0.1f; // Simulación de datos
    }

    // Mostrar el vector
    printf("Vector de prueba: ");
    print_vector(vector, DIMS);

    uint64_t id = 12345; // ID del vector

    // Insertar el vector en el índice
    if (insert(index, id, vector, dims) != 0) {
        printf("Error: No se pudo insertar el vector.\n");
        destroy_index(&index);
        return 1;
    }
    printf("Vector insertado correctamente.\n");

    // Buscar el mejor resultado
    MatchResult result;
    if (search(index, vector, dims, &result) != 0) {
        printf("Error: No se encontró el vector.\n");
        destroy_index(&index);
        return 1;
    }

    // Mostrar el mejor resultado
    printf("Mejor resultado encontrado: ID=%llu, Score=%.4f\n", 
           (unsigned long long)result.id, result.distance);

    // Buscar los N mejores resultados
    MatchResult *results[TOP_N];
    if (search_n(index, vector, dims, results, TOP_N) != -1) {
        printf("Top %d resultados:\n", TOP_N);
        for (int i = 0; i < TOP_N; i++) {
            if (results[i]) {
                printf(" - ID=%llu, Score=%.4f\n", 
                       (unsigned long long)results[i]->id, results[i]->distance);
                free(results[i]); // Liberar memoria asignada
            }
        }
    } else {
        printf("Error en la búsqueda de los mejores %d resultados.\n", TOP_N);
    }

    // Eliminar el vector
    if (delete(index, id) != 0) {
        printf("Error: No se pudo eliminar el vector.\n");
        destroy_index(&index);
        return 1;
    }
    printf("Vector eliminado correctamente.\n");

    // Destruir el índice
    destroy_index(&index);
    printf("Índice destruido correctamente.\n");

    return 0;
}
