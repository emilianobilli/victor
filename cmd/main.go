package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"victor"
)

func main() {
	var db *victor.VictorDB
	var err error

	// Definir los argumentos CLI
	listDBs := flag.Bool("list", false, "List all databases")
	deleteDB := flag.String("delete", "", "Delete a database by name")
	dbName := flag.String("db", "", "Database name to open")
	newDB := flag.String("new", "", "Create a new database with this name")
	dbType := flag.String("type", "", "Database type (e.g., filstorage, s3storage)")
	uri := flag.String("uri", "", "Database URI")
	dims := flag.Int("dims", 0, "Number of dimensions for the vectors (only for new databases)")
	mode := flag.String("mode", "", "Comparison mode (e.g., L2NORM | COSINE, only for new databases)")
	ip := flag.String("ip", "0.0.0.0", "IP address to listen on")
	port := flag.Int("port", 8080, "Port number to listen on")

	// Parsear los argumentos
	flag.Parse()
	// Cargar configuración existente
	conf, err := victor.LoadConfig()
	if err != nil {
		log.Fatalf("Error loading config: %v", err)
		return
	}

	// Comando: Listar todas las bases de datos
	if *listDBs {
		fmt.Println("Registered Databases:")
		for name, dbConfig := range *conf {
			fmt.Printf("- %s: Type=%s, Mode=%s, Dims=%d, URI=%s\n",
				name, dbConfig.Type, dbConfig.Mode, dbConfig.Dims, dbConfig.URI)
		}
		return
	}

	// Comando: Eliminar una base de datos
	if *deleteDB != "" {
		err := victor.DeleteDatabase(*deleteDB)
		if err != nil {
			log.Fatalf("Error deleting database: %v", err)
		}
		fmt.Printf("Database '%s' successfully deleted.\n", *deleteDB)
		return
	}
	// Si se especifica `-new`, se crea la base de datos y se usará automáticamente como `-db`

	if *newDB != "" {
		if *dbType == "" || *uri == "" || *dims == 0 || *mode == "" {
			log.Fatal("To create a new database, you must specify --type, --uri, --dims, and --mode.")
			return
		}

		// Usar `AppendDatabase` para agregar la base con validaciones
		conf, err = victor.AppendDatabase(*newDB, victor.DatabaseConfig{
			Type: *dbType,
			Mode: *mode,
			Dims: *dims,
			URI:  *uri,
		})
		if err != nil {
			log.Fatalf("Error creating database: %v", err)
			return
		}

		fmt.Printf("Database '%s' successfully created.\n", *newDB)

		// Asignar el nombre de la nueva base para que se abra automáticamente
		dbName = newDB
	}

	// Si después de todo no se especificó una base de datos a abrir, error
	if *dbName == "" {
		log.Fatal("You must specify a database to open using --db or create a new one with --new.")
		return
	}

	// Buscar la base de datos en la configuración
	dbConfig, exists := (*conf)[*dbName]
	if !exists {
		log.Fatalf("Database '%s' not found in configuration.", *dbName)
		return
	}

	storage, err := victor.NewFileStorage(dbConfig.URI)
	if err != nil {
		log.Fatal(err)
		return
	}
	db, err = victor.Open(storage, dbConfig.Dims, dbConfig.Mode)
	if err != nil {
		log.Fatal(err)
		return
	}

	// Configurar manejadores HTTP
	http.HandleFunc("/insert", db.InsertHandler)
	http.HandleFunc("/delete", db.DeleteHandler) // Usa query param: /delete?id=123
	http.HandleFunc("/search", db.SearchHandler)

	// Iniciar el servidor
	fmt.Printf("Server running on %s:%d\n", *ip, *port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf("%s:%d", *ip, *port), nil))
}
