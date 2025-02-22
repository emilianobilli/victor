package victor

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
)

type Record struct {
	ID         *string                `json:"id,omitempty"`
	Embeddings [][]float32            `json:"embeddings"`
	Data       map[string]interface{} `json:"data"`
}

func hashVector(vector []float32) string {
	hasher := sha256.New()

	for _, value := range vector {
		bits := math.Float32bits(value)
		hasher.Write([]byte{
			byte(bits >> 24),
			byte(bits >> 16),
			byte(bits >> 8),
			byte(bits),
		})
	}

	hash := hex.EncodeToString(hasher.Sum(nil))
	return hash[:16]
}
