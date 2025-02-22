package victor

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"
)

var FILEXT string = ".rec"

type FileStorage struct {
	Path string
}

// NewFileStorage crea una nueva instancia de FileStorage y se asegura de que el directorio exista
func NewFileStorage(path string) (Storage, error) {
	// Verificar si el directorio existe
	if _, err := os.Stat(path); os.IsNotExist(err) {
		// Si no existe, crearlo
		err := os.MkdirAll(path, 0755)
		if err != nil {
			return nil, fmt.Errorf("failed to create storage directory: %w", err)
		}
	}

	return &FileStorage{Path: path}, nil
}

func (f *FileStorage) save(record *Record) error {
	data, err := json.Marshal(record) // Pretty print
	if err != nil {
		return err
	}
	filename := path.Join(f.Path, *record.ID+FILEXT)
	return os.WriteFile(filename, data, 0644) // Save to file
}

func (f *FileStorage) load(id string) (*Record, error) {
	var record Record

	filename := path.Join(f.Path, id+FILEXT)
	// Read JSON file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Convert JSON to struct
	err = json.Unmarshal(data, &record)
	if err != nil {
		return nil, err
	}

	return &record, nil
}

func (f *FileStorage) delete(id string) error {
	filename := path.Join(f.Path, id+FILEXT)
	return os.Remove(filename)
}

func (f *FileStorage) check(id string) bool {
	_, err := os.Stat(path.Join(f.Path, id+FILEXT))
	return err == nil || !os.IsNotExist(err)
}

func (f *FileStorage) list() ([]string, error) {
	var ids []string
	files, err := os.ReadDir(f.Path)
	if err != nil {
		return nil, err
	}
	for _, file := range files {
		if !file.IsDir() && path.Ext(file.Name()) == FILEXT {
			base := path.Base(file.Name())
			name := strings.TrimSuffix(base, path.Ext(base))
			ids = append(ids, name)
		}
	}
	return ids, nil
}
