package victor

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// DatabaseConfig representa la configuración de una base de datos
type DatabaseConfig struct {
	Type string `json:"type"`
	Mode string `json:"mode"`
	Dims int    `json:"dims"`
	URI  string `json:"uri"`
}

// Config es un mapa donde las claves son los nombres de las bases de datos
type Config map[string]DatabaseConfig

var (
	configFile string
)

func init() {
	home, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("Error getting home directory: %v", err)
		os.Exit(-1)
	}
	configFile = filepath.Join(home, ".victor.config")
}

// LoadConfig carga el archivo de configuración (lo crea si no existe)
func LoadConfig() (*Config, error) {
	var conf Config

	// Verificar si el archivo existe
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		// Si no existe, creamos un archivo vacío con un JSON válido
		defaultConfig := Config{}
		err := SaveConfig(&defaultConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create default config: %w", err)
		}
		return &defaultConfig, nil
	}

	// Leer el archivo
	data, err := os.ReadFile(configFile)
	if err != nil {
		return nil, err
	}

	// Decodificar el JSON
	err = json.Unmarshal(data, &conf)
	if err != nil {
		return nil, err
	}

	return &conf, nil
}

// SaveConfig guarda la configuración en el archivo
func SaveConfig(conf *Config) error {
	data, err := json.Marshal(conf)
	if err != nil {
		return err
	}
	return os.WriteFile(configFile, data, 0644)
}

// AppendDatabase agrega una nueva base de datos si no tiene la misma URI para el mismo tipo
func AppendDatabase(name string, dbConfig DatabaseConfig) (*Config, error) {
	conf, err := LoadConfig()
	if err != nil {
		return nil, err
	}

	// Verificar si ya existe una base de datos con el mismo tipo y URI
	for n, existingDB := range *conf {
		if n == name {
			return nil, fmt.Errorf("a database with the name '%s' already exists", name)
		}
		if existingDB.Type == dbConfig.Type && existingDB.URI == dbConfig.URI {
			return nil, fmt.Errorf("database of type '%s' with URI '%s' already exists", dbConfig.Type, dbConfig.URI)
		}
	}

	// Agregar la nueva base de datos
	(*conf)[name] = dbConfig

	// Guardar la configuración actualizada
	return conf, SaveConfig(conf)
}

// DeleteDatabase elimina una base de datos de la configuración
func DeleteDatabase(name string) error {
	conf, err := LoadConfig()
	if err != nil {
		return err
	}

	// Verificar si la base de datos existe
	if _, exists := (*conf)[name]; !exists {
		return fmt.Errorf("database '%s' not found", name)
	}

	// Eliminar la base de datos del mapa
	delete(*conf, name)

	// Guardar la configuración actualizada
	err = SaveConfig(conf)
	if err != nil {
		return fmt.Errorf("failed to save config after deletion: %w", err)
	}

	return nil
}
